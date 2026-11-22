# Final report: outline

**Working title:** *Reproducing "Attention Is All You Need" from scratch: a fair, multi-seed comparison with attention RNNs*

- **Audience:** ML engineers and researchers, as a portfolio piece.
- **Length:** 8–10 pages plus appendix.
- **Rules for the report:**
  - every number comes from [results.md](results.md), and every claim cites an ADR or a results section;
  - nothing new is computed for the report;
  - figures are optional for v1.0, because the §11.3 set is deferred (ADR-023 Amendment 4). The report is built from tables.

---

## Abstract (≈ 150 words)
- **What was built:** a hand-written Transformer, policy-enforced and parity-tested against PyTorch, and two parameter-matched attention RNN baselines.
- **What was measured:**
  - IWSLT'14 DE→EN: 3 seeds, ≈ 36.6M parameters;
  - Multi30k EN→DE: 5 seeds, 7.6M parameters, ablations.
- **Headline:** 33.10 ± 0.07 vs 32.04 ± 0.10 BLEU (IWSLT'14). The lead is significant but ≈ 1 BLEU once each model's learning rate is tuned.
- **Secondary findings:**
  - attention is decisive (−19 to −24 BLEU without it);
  - Post-LN wins at 3+3 layers but failed at 6+6 in this recipe (declared Pre-LN deviation).

## 1. Introduction (≈ 0.75 page)
- **Goal:** test the paper's central claim (self-attention beats recurrence for NMT at equal cost) as a controlled experiment, not by quoting a BLEU number.
- **What "from scratch" means here:** banned APIs (ADR-001), parity tests (ADR-004), no pretrained weights.
- **Contributions:**
  1. the verified implementation;
  2. a fairness and statistics protocol;
  3. a two-benchmark result;
  4. honest reporting of failures (Post-LN at depth, GRU divergence, collapsed seeds).
- **Source:** README, DESIGN.md §1.

## 2. Implementation and verification (≈ 1.5 pages)
- **Architecture:** attention, multi-head, positional encoding, LayerNorm, Post-/Pre-LN, embedding tying and scaling. Sources: math.md, ADR-015.
- **Verification:** forward and backward parity with PyTorch references, and invariance tests. Sources: ADR-004, tests/.
- **Training system:** fp16 with explicit upcasts, resumable checkpoints, token-budget batching, gradient accumulation. Sources: ADR-014, ADR-017.
- **Decoding:** beam search with the GNMT length penalty, incremental decoding exact to 1e-10, checkpoint averaging. Sources: ADR-019, ADR-023 §6.
- **One table:** components → how each was verified.

## 3. Experimental protocol (≈ 1 page)
- **Fair comparison:**
  - same data, tokenizer, update budget, effective batch, schedule, decoding and scoring;
  - parameters matched within 1%;
  - declared differences (RNN clipping, GRU accumulation).

  Sources: ADR-021, ADR-023 Amendment 1.
- **Selection rules:** validation only (learning rate, checkpoint, decoding); the test set never drives a choice; protocol defaults are always reported next to tuned results. Sources: ADR-022, ADR-023 §7.
- **Statistics:** two-level paired bootstrap (sentences × seeds), Holm correction, seed permutation test and its floor. Source: results.md §2.9, §7.
- **Metrics:** sacreBLEU with signatures, chrF, and fairseq-style tok-lc BLEU for literature comparability. Sources: ADR-006, ADR-023 §3–4.

## 4. Results: IWSLT'14 DE→EN (≈ 2 pages)
- **4.1 Setup:** data and models. Table: models, configurations, parameters (results.md §2.1).
- **4.2 Main result:** the headline table with both protocol rows, ΔBLEU, CIs and Holm p (results.md §1, §2.2–2.3).
- **4.3 Robustness:**
  - per-seed table (§2.4);
  - decoding grid: corner selected, small effect (§2.3);
  - BLEU by source length, with a constant gap (§2.6).
- **4.4 Averaging, beam search, cost:** §2.7. Make the speed point: the cuDNN LSTM trains and decodes faster at this scale.
- **4.5 Literature context:** tok-lc 34.67–34.84 vs published ≈ 34–35, with the raw-text caveat.

## 5. Results: Multi30k EN→DE study (≈ 1.5 pages)
- **Main comparison, 5 seeds,** including the no-attention LSTM: results.md §3, §5.1.
- **Training sensitivity:** the learning rate dominates and narrows the gap to 1.6–1.8 (§5.3).
- **Ablations:** the paper's Table 3 rows; the significant and confounded ones (§6).

## 6. Normalization placement and depth (≈ 1 page)
- **Multi30k 3+3:** Post-LN > Pre-LN by 1.09 BLEU.
- **IWSLT'14 6+6:** Post-LN failed (pilot at 3 learning rates; 30k updates: NLL 3.03, BLEU 5.79) while Pre-LN reached 34.33. Table from results.md §2.8.
- **The bound:**
  - ruled out: learning rate in [5e-4, 1e-3], too few updates, fp16;
  - unlikely: an implementation fault;
  - remaining: fairseq recipe and initialization differences.

  Source: ADR-023 Amendment 4.
- **Lesson:** the gate at 6k updates was biased against slow starters, so architecture decisions are now made at the full budget.

## 7. Failures caught by the protocol (≈ 0.75 page)
- GRU late-training divergence (results.md §2.5).
- Silently collapsed BiLSTM seed (§5.4).
- Coin-flip NLL selection (§5.3).
- Harmful averaging window (§5.5).
- **For each:** symptom → which safeguard caught it → what changed in the code or protocol.

## 8. Limitations (≈ 0.5 page)
- Merge results.md §2.10 and §8: Pre-LN deviation, GRU learning rate not checked at full budget, unbracketed decoding grid, 3 seeds, one language pair per benchmark, no WMT'14.

## 9. Conclusions and future work (≈ 0.5 page)
- **Conclusions:** results.md §10, points 1–5.
- **Future work:** results.md §11. Post-LN investigation first.

## Appendix
- **A. Hyperparameters:** the frozen experiment configs (`configs/experiment/iwslt14_*.yaml`), as a table.
- **B. Reproduction:** README "Reproduce the IWSLT'14 benchmark", the experiments archive, the re-scoring command (archive NOTES.md).
- **C. ADR index:** one line per ADR (docs/decisions/README.md).
- **D. Compute:** 8.9 GPU-hours (IWSLT'14, RTX 5070 Ti); Multi30k on a Kaggle T4.

---

**Format:** write the report as a doc or PDF from this outline. Tables are copied from results.md unchanged, so the numbers stay identical across README, results.md, the report and the slides.
