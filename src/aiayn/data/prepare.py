"""Download, verify and lay out parallel corpora (ADR-002: pinned, checksummed sources).

Layout under ``<root>`` (``translation.root``, repository-relative by default)::

    raw/<name>/<file from URL>          downloaded archives (checksummed)
    processed/<name>/<split>.<lang>     plain UTF-8 text, one sentence per line
    processed/<name>/MANIFEST.json      archive recipes only: source hash, counts, file hashes
"""

from __future__ import annotations

import gzip
import json
import shutil
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from aiayn.data.config import SPLITS, ArchiveSource, DataConfig, DownloadSpec, Split
from aiayn.data.datasets import read_parallel_text
from aiayn.utils.hashing import sha256_file
from aiayn.utils.paths import resolve_path

__all__ = [
    "ChecksumError",
    "DataPaths",
    "data_paths",
    "download",
    "prepare_translation_data",
]

_ALLOWED_SCHEMES = ("https", "file")


class ChecksumError(RuntimeError):
    """A downloaded file does not match its pinned SHA-256."""


@dataclass(frozen=True)
class DataPaths:
    """Filesystem locations derived from a translation ``DataConfig``."""

    root: Path
    raw_dir: Path
    processed_dir: Path
    tokenizer_prefix: Path

    def text_file(self, split: Split, lang: str) -> Path:
        return self.processed_dir / f"{split}.{lang}"

    @property
    def tokenizer_model(self) -> Path:
        return self.tokenizer_prefix.with_name(self.tokenizer_prefix.name + ".model")


def data_paths(config: DataConfig, base: str | Path | None = None) -> DataPaths:
    """Resolve the directory layout for ``config`` (a translation config)."""
    if config.translation is None:
        raise ValueError(f"{config.name}: not a translation config")
    t = config.translation
    root = resolve_path(t.root, base)
    tok = t.tokenizer
    return DataPaths(
        root=root,
        raw_dir=root / "raw" / config.name,
        processed_dir=root / "processed" / config.name,
        tokenizer_prefix=root / "tokenizers" / f"{config.name}_{tok.model_type}{tok.vocab_size}",
    )


def download(
    url: str,
    dest: Path,
    *,
    sha256: str,
    timeout: float = 60.0,
    mirror_dirs: Sequence[Path] = (),
) -> Path:
    """Fetch ``url`` to ``dest`` and verify its SHA-256.

    Lookup order: a verified cached ``dest``; then a file with the URL's basename in any
    of ``mirror_dirs`` (offline use, e.g. a Kaggle Dataset under ``/kaggle/input``);
    then the network. Every source is checksum-verified. On a mismatch nothing is left at
    ``dest`` (the partial ``.part`` file is removed) and ``ChecksumError`` is raised.
    """
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"refusing URL scheme {scheme!r} (allowed: {_ALLOWED_SCHEMES})")
    if dest.is_file() and sha256_file(dest) == sha256:
        return dest
    filename = Path(urllib.parse.urlparse(url).path).name
    mirrored = [Path(d) / filename for d in mirror_dirs if (Path(d) / filename).is_file()]
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        if mirrored:
            shutil.copyfile(mirrored[0], part)
        else:
            with urllib.request.urlopen(url, timeout=timeout) as response, part.open("wb") as fh:
                shutil.copyfileobj(response, fh)
        actual = sha256_file(part)
        if actual != sha256:
            raise ChecksumError(f"{url}: expected sha256 {sha256}, got {actual}")
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)
    return dest


def _extract(spec: DownloadSpec, archive: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    if spec.compression == "gzip":
        with gzip.open(archive, "rb") as src, tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        shutil.copyfile(archive, tmp)
    tmp.replace(out)


def _write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="")
    tmp.replace(path)


def _prepare_archive(config: DataConfig, paths: DataPaths, archive: Path) -> None:
    """Run the archive's recipe once; ``MANIFEST.json`` makes it idempotent (ADR-023)."""
    assert config.translation is not None and config.translation.archive is not None
    source: ArchiveSource = config.translation.archive
    manifest_path = paths.processed_dir / "MANIFEST.json"
    langs = (config.translation.src_lang, config.translation.tgt_lang)
    outputs = [paths.text_file(s, lang) for s in SPLITS for lang in langs]
    if manifest_path.is_file() and all(p.is_file() for p in outputs):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("archive_sha256") == source.sha256 and manifest.get("files") == {
            p.name: sha256_file(p) for p in outputs
        }:
            return
    from aiayn.data.iwslt import read_iwslt14  # recipe-specific (needs sacremoses)

    splits = read_iwslt14(archive)
    expected = {"train": source.train_lines, "valid": source.valid_lines, "test": source.test_lines}
    counts = splits.counts()
    wrong = {s: (counts[s], n) for s, n in expected.items() if n and counts[s] != n}
    if wrong:
        raise ChecksumError(f"{source.recipe}: split sizes (got, expected) {wrong}")
    for split in SPLITS:
        for lang in langs:
            _write_lines(paths.text_file(split, lang), splits.splits[split][lang])
    manifest = {
        "recipe": source.recipe,
        "archive_sha256": source.sha256,
        "counts": counts,
        "files": {p.name: sha256_file(p) for p in outputs},
    }
    _write_lines(manifest_path, [json.dumps(manifest, indent=2)])


def prepare_translation_data(
    config: DataConfig,
    base: str | Path | None = None,
    *,
    timeout: float = 60.0,
    mirror_dirs: Sequence[Path] = (),
) -> DataPaths:
    """Download (or copy from ``mirror_dirs``), verify, extract and check split alignment."""
    paths = data_paths(config, base)
    assert config.translation is not None
    t = config.translation
    if t.archive is not None:
        filename = Path(urllib.parse.urlparse(t.archive.url).path).name
        archive = download(
            t.archive.url,
            paths.raw_dir / filename,
            sha256=t.archive.sha256,
            timeout=timeout,
            mirror_dirs=mirror_dirs,
        )
        _prepare_archive(config, paths, archive)
    for spec in t.downloads:
        filename = Path(urllib.parse.urlparse(spec.url).path).name
        archive = download(
            spec.url,
            paths.raw_dir / filename,
            sha256=spec.sha256,
            timeout=timeout,
            mirror_dirs=mirror_dirs,
        )
        _extract(spec, archive, paths.text_file(spec.split, spec.lang))
    for split in SPLITS:
        read_parallel_text(paths.text_file(split, t.src_lang), paths.text_file(split, t.tgt_lang))
    return paths
