# Final presentation: outline

- **Audience:** technical interviewers or an ML reading group.
- **Format:** 12 slides, about 15 minutes plus questions.
- **Rules:** one message per slide. All numbers come from [results.md](results.md). Tables are simplified from results.md; nothing is recomputed.

| # | Slide title (the message) | Content | Source |
|---|---|---|---|
| 1 | **Reproducing *Attention Is All You Need* from scratch** | Title; one line: "hand-written, verified, fairly compared"; repository link | README |
| 2 | **The question: does self-attention beat recurrence at equal cost?** | Paper claim vs what a fair test needs: same data, size, budget, decoding, scoring, and several seeds | DESIGN.md §1, ADR-021 |
| 3 | **Every component is hand-written and verified** | Banned APIs (nn.Transformer, MultiheadAttention, SDPA, LayerNorm); parity with PyTorch forward and backward; incremental decoding exact to 1e-10 | ADR-001, ADR-004, ADR-023 §6 |
| 4 | **A fair fight: matched models, shared code, validation-only choices** | Three models at ≈ 36.6M parameters (within 0.3%); shared trainer, beam search and scorer; per-model learning rate chosen at full budget; test never used for choices | results.md §2.1, ADR-022 |
| 5 | **Headline: the Transformer wins IWSLT'14 by ≈ 1 BLEU** | Bars or table: 33.10 ± 0.07 / 32.04 ± 0.10 / 28.88 ± 0.25; Δ −1.06 [−1.33, −0.82], Holm p = 0.002; tok-lc 34.84, in the published range | results.md §1 |
| 6 | **The margin depends on tuning** | Multi30k shared defaults: +2.7 to +4.3. Tuned learning rate: +1.6 to +1.8. IWSLT'14 with per-model learning rate: +1.06 | results.md §1, §5.3 |
| 7 | **Attention is the decisive ingredient** | No-attention LSTM 12.60 vs 32–37 for the attention models (Multi30k); recurrence vs self-attention is worth only 1–4 BLEU | results.md §3, §10 |
| 8 | **Robust across seeds, decoding and sentence length** | Seed std 0.05–0.19 (Transformer, BiLSTM); same ranking at every beam and α; constant gap across length buckets (+0.9 to +1.4) | results.md §2.3, §2.4, §2.6 |
| 9 | **Surprise: the paper's Post-LN did not train at 6+6** | Post-LN vs Pre-LN table (NLL 3.03 vs 1.62; BLEU 5.79 vs 34.33); yet Post-LN *wins* at 3+3 on Multi30k (+1.1) | results.md §2.8, §6 |
| 10 | **What we ruled out, and what's left** | Ruled out: learning rate in [5e-4, 1e-3], too few updates, fp16. Unlikely: a code bug (parity tests, 3+3 works). Left: recipe and initialization differences from fairseq. Hence the declared Pre-LN deviation | ADR-023 Amendment 4 |
| 11 | **The protocol caught failures a single number would hide** | GRU late divergence (averaged checkpoint 0.06 BLEU, best checkpoint 28.88); collapsed BiLSTM seed; coin-flip NLL selection; harmful averaging window. Each one changed the protocol | results.md §2.5, §5.4, §8 |
| 12 | **Takeaways and future work** | Takeaways: results.md §10, three bullets. Next: Post-LN investigation, GRU stability, wider decoding grid, figures, WMT'14 | results.md §10–11 |

## Backup slides (for questions)
- **B1 Statistics:** two-level paired bootstrap; why the 3-seed permutation test cannot go below 0.100 (results.md §2.9).
- **B2 Data pipeline:** pinned IWSLT'14 source, fairseq membership, 25-pair difference, Protocol A vs B (ADR-023 §1–4).
- **B3 Cost:** 8.9 GPU-hours for IWSLT'14; throughput and memory per model; why the cuDNN LSTM is faster (results.md §2.7).
- **B4 Multi30k ablations:** the paper's Table 3 rows; confounded learned PE and embedding scale (results.md §6).
- **B5 GRU memory fix:** Bahdanau activation memory, gradient accumulation at an identical effective update (ADR-023 Amendment 1).

## Speaker notes, key lines
- "Same everything except the architecture, and every choice made on validation."
- "The Transformer wins, but by about one BLEU once the baselines are tuned. Most of the textbook gap is tuning."
- "The paper's own normalization failed at depth in our recipe. We report it, we bound it, and we don't hide it behind Pre-LN."
