# Paper Notes: *Attention Is All You Need*

| | |
|---|---|
| **Paper** | Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin, *Attention Is All You Need*, NIPS 2017 |
| **Source used** | arXiv:1706.03762**v7** (2 Aug 2023). All numbers below were transcribed from that version's text and checked against it. |
| **Reference code** | `tensorflow/tensor2tensor` (named in §7 of the paper) |
| **Purpose** | Section-by-section study notes that turn the paper into reproduction decisions for this repository |
| **Status** | Week 1 deliverable. Living document; update it when an experiment changes our understanding. |

**How to read these notes.** Each section has the same structure:

- **Paper says**: a faithful summary (direct quotes are in quotation marks).
- **What it means**: interpretation and intuition.
- **Not specified**: details the paper leaves out, which we must decide.
- **Repro note**: what this repository will do, with a link to an ADR or a section of [math.md](math.md) where relevant.

---

## 0. TL;DR

1. The Transformer replaces recurrence and convolution with **attention only**. Any two positions interact in **O(1) sequential steps**, at **O(n²·d)** cost per layer.
2. Building blocks: **scaled dot-product attention** → **multi-head attention** → **position-wise FFN**, each wrapped in **residual + LayerNorm (Post-LN)**. Order information comes from **sinusoidal positional encodings**.
3. The training recipe matters as much as the architecture: **Adam (β₂ = 0.98)**, **linear warmup + inverse-sqrt decay**, **label smoothing 0.1**, **residual dropout**, **shared BPE vocabulary + three-way weight tying**, **checkpoint averaging**, **beam search with a length penalty**.
4. Results: **27.3 / 28.4 BLEU** (base / big) on WMT14 EN→DE and **38.1 / 41.8** on EN→FR. Training cost is a fraction of the previous best systems (base: 12 h on 8×P100).
5. The ablations (Table 3) show that multiple heads help but too many hurt, that a small d_k hurts, that bigger is better, that dropout is essential, and that **learned ≈ sinusoidal positional encodings**.

---

## 1. Abstract

**Main claims, in the authors' words:**
- "based solely on attention mechanisms, dispensing with recurrence and convolutions entirely"
- "superior in quality while being more parallelizable and requiring significantly less time to train"
- WMT14 EN→DE: **28.4 BLEU**, "improving over the existing best results, including ensembles, by over 2 BLEU"
- WMT14 EN→FR: "new single-model state-of-the-art BLEU score of **41.8** after training for 3.5 days on eight GPUs"
- Generalizes to English constituency parsing "both with large and limited training data"

**What it means.** The paper makes three separate claims: **quality**, **efficiency** (parallelism and training time) and **generality**. A good reproduction tests each one separately. Our plan tests the first two directly (BLEU; throughput against RNN baselines) and leaves the third (parsing) out of scope.

> **Inconsistency in the paper.** The abstract and Table 2 report **41.8** for EN→FR, but §6.1 says "our big model achieves a BLEU score of **41.0**". The 41.0 appears to be left over from the original NIPS version. We cite 41.8 (Table 2) and mention this discrepancy whenever EN→FR comes up.

---

## 2. §1 Introduction

**Paper says.**
- RNNs, LSTMs and GRUs are the state of the art for sequence modelling and transduction (language modelling, MT).
- Recurrent models "factor computation along the symbol positions": hₜ = f(hₜ₋₁, xₜ). "This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples."
- Factorization tricks and conditional computation (mixture of experts) improved efficiency, but "the fundamental constraint of sequential computation, however, remains."
- Attention already lets models capture dependencies "without regard to their distance", but almost always *together with* an RNN.
- Proposal: rely "entirely on an attention mechanism to draw global dependencies between input and output". State-of-the-art quality "after being trained for as little as twelve hours on eight P100 GPUs."

**What it means.** The motivation is primarily **hardware efficiency**: matrix multiplies over all positions at once suit GPUs and TPUs far better than a loop over time. Quality gains are presented as a welcome bonus. This framing explains why the paper reports **training FLOPs** next to BLEU.

**Repro note.** We measure the efficiency claim directly: tokens/s and time-to-BLEU for the Transformer against parameter-matched LSTM/GRU baselines (spec §7).

---

## 3. §2 Background

**Paper says.**
- Convolutional models (Extended Neural GPU, ByteNet, ConvS2S) also parallelize, but the number of operations needed to relate two distant positions grows **linearly (ConvS2S)** or **logarithmically (ByteNet)** with distance.
- In the Transformer this becomes **constant**, "albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention."
- Self-attention ("intra-attention") had already been used for reading comprehension, abstractive summarization, textual entailment and sentence embeddings. End-to-end memory networks used recurrent attention.
- Claim: "the first transduction model relying entirely on self-attention … without using sequence-aligned RNNs or convolution."

**What it means.** This is the clearest statement in the paper of **why multi-head attention exists**. One softmax produces *one* weighted average per query, which blurs information from different positions together. Several heads give several independent averages, which restores "resolution". This is the motivation to quote in interviews, rather than "it's like having several CNN filters".

---

## 4. §3 Model Architecture

### 4.0 Overview
**Paper says.** Standard encoder-decoder. The encoder maps (x₁…xₙ) → z = (z₁…zₙ). The decoder generates (y₁…yₘ) "one element at a time", and is **auto-regressive**: it consumes previously generated symbols.

**Figure 1, described in words (dataflow):**
```
Encoder (×N)                                   Decoder (×N)
inputs ─► Embedding·√d ─► +PE ─► Dropout       outputs (shifted right) ─► Embedding·√d ─► +PE ─► Dropout
   │                                              │
   ├─► [Multi-Head Self-Attn] ─► Add & Norm       ├─► [Masked Multi-Head Self-Attn] ─► Add & Norm
   └─► [Feed-Forward]         ─► Add & Norm       ├─► [Multi-Head Cross-Attn (K,V = encoder out)] ─► Add & Norm
                 │                                └─► [Feed-Forward] ─► Add & Norm
                 └──────── memory z ─────────────────────▲
                                                   Linear (tied) ─► Softmax ─► P(yₜ | y<ₜ, x)
```

### 4.1 §3.1 Encoder and decoder stacks
**Paper says.**
- **Encoder:** N = 6 identical layers. Each has two sub-layers: multi-head self-attention and a position-wise FFN. "We employ a residual connection around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is **LayerNorm(x + Sublayer(x))**."
- All sub-layers and the embedding layers output **d_model = 512**, "to facilitate these residual connections".
- **Decoder:** N = 6 identical layers with a **third** sub-layer doing multi-head attention "over the output of the encoder stack". Decoder self-attention is masked "to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i."

**What it means.**
- The **Post-LN** formula is stated explicitly. This is the main thing that separates a faithful reproduction from most tutorials (the Annotated Transformer uses Pre-LN "for code simplicity").
- A constant width throughout is what makes residual connections possible without projection layers.
- Causal correctness needs **two** mechanisms together: the causal mask *and* the one-position shift of decoder inputs. Getting either wrong produces a model that trains to near-zero loss and generates garbage.

**Not specified.** Whether there is a final LayerNorm after the last layer of each stack (with Post-LN there is none in the formula). Initialization scheme.

**Repro note.** Post-LN by default, Pre-LN as a configuration switch ([ADR-003](decisions/ADR-003-post-ln-default.md)). No extra stack-level LayerNorm in Post-LN mode. Pre-LN mode adds a final LayerNorm, as is standard.

### 4.2 §3.2 Attention (general definition)
**Paper says.** "An attention function can be described as mapping a query and a set of key-value pairs to an output … The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key."

**What it means.** This is a *soft dictionary lookup*. Hard lookup returns the value whose key equals the query. Attention returns a convex combination of all values, weighted by how well each key matches. Because the weights are differentiable, the lookup itself can be learned.

### 4.3 §3.2.1 Scaled dot-product attention
**Paper says.**
- Attention(Q, K, V) = softmax(QKᵀ / √d_k) V (Eq. 1).
- Compared with **additive attention** (Bahdanau: a feed-forward network with one hidden layer), dot-product attention is similar in theoretical complexity but "much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code."
- "While for small values of d_k the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of d_k. We suspect that for large values of d_k, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients."
- Footnote 4: if the components of q and k are independent with mean 0 and variance 1, then q·k has mean 0 and **variance d_k**.

**What it means.** The scaling is a *variance-preserving* fix, the same idea as Xavier/He initialization, applied to the softmax input. Without it, logits have standard deviation √d_k (8 for d_k = 64), softmax becomes nearly one-hot, and its Jacobian, diag(p) − ppᵀ, collapses towards zero. Full derivation: [math.md §2](math.md#2-why-divide-by-d_k).

**Not specified.** The exact masking value (the paper says "−∞"). Whether dropout is applied to the attention weights (see §4.5 below).

**Repro note.** Masking uses `masked_fill` with the dtype's most negative finite value (−∞ produces NaN on fully masked rows; −1e9 overflows fp16). Mask convention: **True = may attend**.

### 4.4 §3.2.2 Multi-head attention
**Paper says.**
- MultiHead(Q, K, V) = Concat(head₁, …, head_h) Wᴼ, where headᵢ = Attention(QWᵢᵠ, KWᵢᴷ, VWᵢⱽ).
- Wᵢᵠ, Wᵢᴷ ∈ ℝ^{d_model×d_k}, Wᵢⱽ ∈ ℝ^{d_model×d_v}, Wᴼ ∈ ℝ^{h·d_v×d_model}.
- h = 8, d_k = d_v = d_model / h = 64. "Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality."
- "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this."

**What it means.** The h per-head projections stack into **one d×d matrix**, so the implementation is a single matmul followed by a reshape into heads. Projection FLOPs are independent of h. The QKᵀ and AV FLOPs are also independent of h (h · n² · d_k = n² · d). What *does* change with h is the number of independent softmax distributions and the rank of each head's bilinear form (≤ d_k).

**Repro note.** The head split and merge are the most likely place for a "shapes correct, values wrong" bug, which is why [ADR-004](decisions/ADR-004-parity-testing.md) requires parity tests.

### 4.5 §3.2.3 Applications of attention in the model
**Paper says.** Attention is used in three ways:
1. **Encoder-decoder (cross) attention:** queries from the previous decoder layer; "memory keys and values come from the output of the encoder". Every decoder position attends over all input positions.
2. **Encoder self-attention:** Q, K and V all come from the previous encoder layer. Every position attends to every position.
3. **Decoder self-attention:** each position attends "to all positions in the decoder up to and including that position". Leftward information flow is prevented "inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections."

**What it means.** "Up to **and including**" means the causal mask keeps the **diagonal**. Cross-attention uses the *source* padding mask, not the target mask.

**Attention dropout.** §3 never mentions dropout on attention weights, but §6.3 refers to tuning "the dropout, both attention and residual (section 5.4)". So attention dropout existed in the authors' code even though §5.4 only describes residual dropout. **Repro note:** attention dropout is a separate configuration value, default 0.1 to match the reference implementations, and it is listed as an underspecified item (§12). *Update (Week 3):* the default is now **0.0** (paper-literal for MT) under [ADR-012](decisions/ADR-012-dropout-locations.md); 0.1 is an ablation.

### 4.6 §3.3 Position-wise feed-forward networks
**Paper says.** FFN(x) = max(0, xW₁ + b₁)W₂ + b₂ (Eq. 2), applied "to each position separately and identically". Parameters differ between layers. It can also be described "as two convolutions with kernel size 1". d_ff = 2048.

**What it means.** Attention is the only operation that mixes information *across* positions. The FFN is the per-token nonlinearity and holds about two-thirds of an encoder layer's parameters. Later interpretability work views FFN layers as key-value memories, though that is outside this paper.

**Not specified.** Whether dropout is applied inside the FFN (after the ReLU). tensor2tensor has a separate `relu_dropout` hyperparameter.

### 4.7 §3.4 Embeddings and softmax
**Paper says.** Learned embeddings for input and output tokens. "We share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [Press & Wolf]. In the embedding layers, we multiply those weights by √d_model."

**What it means.**
- **Three-way tying** (source embedding = target embedding = output projection) requires a **joint source-target vocabulary**, which is why §5.1 uses a shared BPE vocabulary.
- The √d_model factor only makes sense together with an initialization. If embeddings are initialized N(0, d^-½), each dimension has variance 1/d, and multiplying by √d gives unit variance, comparable to the positional encoding (each PE dimension has mean square 1/2). The *same* weights used *unscaled* in the output projection give logits with roughly unit variance. See [math.md §5.4](math.md#54-interaction-with-embedding-scaling).

**Not specified.** Embedding initialization. **Repro note:** N(0, d^-½), as in tensor2tensor and fairseq; Xavier-uniform for the other weight matrices; zero biases.

### 4.8 §3.5 Positional encoding
**Paper says.**
- No recurrence or convolution, so order must be injected explicitly. PEs are **added** to the input embeddings "at the bottoms of the encoder and decoder stacks".
- PE(pos, 2i) = sin(pos / 10000^{2i/d_model}) and PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model}).
- "The wavelengths form a geometric progression from 2π to 10000 · 2π." The function was chosen because "for any fixed offset k, PE_{pos+k} can be represented as a linear function of PE_pos."
- Learned positional embeddings "produced nearly identical results (see Table 3 row (E))". Sinusoids were chosen "because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training."

**What it means.** For each frequency, the (sin, cos) pair is a point on a circle, and shifting position by k *rotates* that point by a fixed angle. Attention, which works through linear maps and dot products, can therefore express "k positions to the left". Derivation in [math.md §4](math.md#4-positional-encoding).

**Critical reading.** The extrapolation claim is **a hypothesis the paper does not test**. Later work (for example ALiBi, Press et al. 2022) found that sinusoidal absolute encodings extrapolate poorly in practice. This is a cheap and interesting extension experiment (train on short sentences, test on long ones).

**Repro note.** The PE table is a non-trainable buffer. Dimensions are interleaved (sin at even indices, cos at odd), exactly as in the formula. Learned PE is an ablation switch.

---

## 5. §4 Why Self-Attention

**Paper says.** Three desiderata for a layer that maps (x₁…xₙ) → (z₁…zₙ):
1. Total computational complexity per layer.
2. Parallelizable computation, measured by the minimum number of **sequential operations**.
3. **Path length** between long-range dependencies: "the shorter these paths … the easier it is to learn long-range dependencies".

**Table 1 (transcribed):**

| Layer type | Complexity per layer | Sequential ops | Max path length |
|---|---|---|---|
| Self-attention | O(n² · d) | O(1) | O(1) |
| Recurrent | O(n · d²) | O(n) | O(n) |
| Convolutional | O(k · n · d²) | O(1) | O(log_k(n)) |
| Self-attention (restricted) | O(r · n · d) | O(1) | O(n/r) |

Further points:
- Self-attention is cheaper than recurrence "when the sequence length n is smaller than the representation dimensionality d, which is most often the case" for subword-level MT sentences.
- Restricting attention to a neighbourhood of size r would help very long sequences, at the cost of O(n/r) path length ("future work").
- One convolution with k < n does not connect all pairs. Doing so needs O(n/k) contiguous layers or O(log_k n) dilated layers. Separable convolutions cost O(k·n·d + n·d²), which "even with k = n" equals self-attention plus a point-wise FFN, "the approach we take in our model".
- Side benefit: interpretability. Heads "clearly learn to perform different tasks", some related to syntax and semantics.

**What it means.** Table 1 counts **per-layer** costs for a *single sequence*. The n < d argument held for 2017 sentence-level MT (n ≈ 30–100 subwords, d = 512). It fails badly for documents, code and images, which motivated a decade of efficient-attention research (sparse, linear, FlashAttention).

**Repro note.** We **measure** Table 1 empirically: forward/backward time and peak memory against n for our attention layer and an LSTM layer, with d fixed (spec §11.3, plot 4).

---

## 6. §5 Training

### 6.1 §5.1 Training data and batching
**Paper says.**
- WMT14 EN-DE: about **4.5M sentence pairs**, byte-pair encoding, **shared source-target vocabulary of about 37000 tokens**.
- WMT14 EN-FR: **36M sentences**, **32000 word-piece** vocabulary.
- "Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately **25000 source tokens and 25000 target tokens**."

**What it means.** Batches are defined by a **token budget**, not a sentence count. Bucketing by length keeps padding low. A step therefore sees a roughly constant number of tokens, which keeps gradient noise roughly constant.

**Back-of-envelope.** 100k steps × 25k target tokens ≈ **2.5B target tokens**. At about 30 subword tokens per sentence, one WMT14 EN-DE epoch is about 135M target tokens, so the base model sees roughly **15–20 epochs**.

### 6.2 §5.2 Hardware and schedule
**Paper says.** One machine with **8 NVIDIA P100**. Base: ~**0.4 s/step**, **100,000 steps ≈ 12 hours**. Big: ~**1.0 s/step**, **300,000 steps ≈ 3.5 days**.

**Repro note.** The 25k-token batch was spread over 8 GPUs (about 3k tokens per GPU). On one GPU we use gradient accumulation, and "step" in every formula means **optimizer update**.

### 6.3 §5.3 Optimizer
**Paper says.** Adam with **β₁ = 0.9, β₂ = 0.98, ε = 10⁻⁹**.
lrate = d_model^{-0.5} · min(step_num^{-0.5}, step_num · warmup_steps^{-1.5}) (Eq. 3), with **warmup_steps = 4000**. "Increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number."

**What it means.** The peak learning rate is **d^{-0.5} · warmup^{-0.5} ≈ 6.99 × 10⁻⁴** at step 4000 (derivation in [math.md §7](math.md#7-noam-learning-rate-schedule)). β₂ = 0.98 (rather than 0.999) gives the second-moment estimate a shorter memory (about 50 steps rather than 1000), so it adapts faster to the rapidly changing gradient scale of early training.

**Not specified.** Weight decay (none implied), gradient clipping (none mentioned), and whether `step_num` starts at 0 or 1. The formula is undefined at 0, so it must start at 1.

### 6.4 §5.4 Regularization
**Paper says.** "Three types of regularization", though only two paragraph headings appear:
- **Residual dropout:** "applied to the output of each sub-layer, before it is added to the sub-layer input and normalized", and to "the sums of the embeddings and the positional encodings in both the encoder and decoder stacks". P_drop = **0.1** (base).
- **Label smoothing:** ε_ls = **0.1**. "This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score."

**What it means.**
- "Three types" but two headings: the third is most plausibly the embedding+PE dropout (or attention dropout, see §4.5). Treat this as an ambiguity.
- The label smoothing remark is an important *metric* lesson: **perplexity and BLEU can move in opposite directions**. Never choose a model by perplexity alone when the target metric is BLEU. Report perplexity computed **without** smoothing.

---

## 7. §6 Results

### 7.1 §6.1 Machine translation

**Table 2 (transcribed; newstest2014; training cost in FLOPs):**

| Model | BLEU EN-DE | BLEU EN-FR | Cost EN-DE | Cost EN-FR |
|---|---|---|---|---|
| ByteNet | 23.75 | | | |
| Deep-Att + PosUnk | | 39.2 | | 1.0·10²⁰ |
| GNMT + RL | 24.6 | 39.92 | 2.3·10¹⁹ | 1.4·10²⁰ |
| ConvS2S | 25.16 | 40.46 | 9.6·10¹⁸ | 1.5·10²⁰ |
| MoE | 26.03 | 40.56 | 2.0·10¹⁹ | 1.2·10²⁰ |
| Deep-Att + PosUnk Ensemble | | 40.4 | | 8.0·10²⁰ |
| GNMT + RL Ensemble | 26.30 | 41.16 | 1.8·10²⁰ | 1.1·10²¹ |
| ConvS2S Ensemble | 26.36 | 41.29 | 7.7·10¹⁹ | 1.2·10²¹ |
| **Transformer (base)** | **27.3** | **38.1** | **3.3·10¹⁸** | |
| **Transformer (big)** | **28.4** | **41.8** | **2.3·10¹⁹** | |

**Inference and evaluation protocol (paper-exact):**
- Base: a single model obtained by **averaging the last 5 checkpoints**, "written at 10-minute intervals". Big: **last 20 checkpoints**.
- **Beam size 4**, **length penalty α = 0.6** (GNMT formula), "chosen after experimentation on the development set".
- **Maximum output length = input length + 50**, "terminate early when possible".
- Big EN-FR used P_drop = **0.1** instead of 0.3 (so big EN-DE used 0.3).

**Training cost methodology (footnote 5).** FLOPs = training time × number of GPUs × assumed sustained single-precision throughput, with **2.8, 3.7, 6.0 and 9.5 TFLOPS** for K80, K40, M40 and P100. Check: base = 12 h × 3600 s × 8 × 9.5·10¹² ≈ **3.3·10¹⁸** ✓. Big = 3.5 days × 86400 × 8 × 9.5·10¹² ≈ **2.3·10¹⁹** ✓.

**Observations.**
- **The base model beats every previous single model *and* ensemble on EN-DE at about 1/3 to 1/50 of their training cost.** This is the paper's most striking result and the one we approximate at smaller scale.
- On EN-FR the *base* model (38.1) does **not** beat earlier systems. Only the big model does. Larger datasets need more capacity.
- The cost numbers are **estimates from assumed peak-ish throughput, not measurements**. They are fine for order-of-magnitude comparisons but should not be treated as precise.
- **BLEU computation is not specified in the paper.** The tensor2tensor evaluation used tokenized BLEU with compound splitting for EN-DE, which is known to read about 1 point higher than detokenized sacreBLEU on the same output (Post, 2018, *A Call for Clarity in Reporting BLEU Scores*). → [ADR-002](decisions/ADR-002-iwslt14-before-wmt14.md) and the evaluation protocol.

### 7.2 §6.2 Model variations (Table 3)
**Setup.** EN-DE, **dev set newstest2013**, beam search as in §6.1 but **no checkpoint averaging**. Perplexities are **per-wordpiece** and "should not be compared to per-word perplexities". Unlisted values equal the base model.

**Table 3 (transcribed):**

| Row | N | d_model | d_ff | h | d_k | d_v | P_drop | ε_ls | Steps | PPL (dev) | BLEU (dev) | Params ×10⁶ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 6 | 512 | 2048 | 8 | 64 | 64 | 0.1 | 0.1 | 100K | 4.92 | 25.8 | 65 |
| (A) | | | | 1 | 512 | 512 | | | | 5.29 | 24.9 | |
| (A) | | | | 4 | 128 | 128 | | | | 5.00 | 25.5 | |
| (A) | | | | 16 | 32 | 32 | | | | 4.91 | 25.8 | |
| (A) | | | | 32 | 16 | 16 | | | | 5.01 | 25.4 | |
| (B) | | | | | 16 | | | | | 5.16 | 25.1 | 58 |
| (B) | | | | | 32 | | | | | 5.01 | 25.4 | 60 |
| (C) | 2 | | | | | | | | | 6.11 | 23.7 | 36 |
| (C) | 4 | | | | | | | | | 5.19 | 25.3 | 50 |
| (C) | 8 | | | | | | | | | 4.88 | 25.5 | 80 |
| (C) | | 256 | | | 32 | 32 | | | | 5.75 | 24.5 | 28 |
| (C) | | 1024 | | | 128 | 128 | | | | 4.66 | 26.0 | 168 |
| (C) | | | 1024 | | | | | | | 5.12 | 25.4 | 53 |
| (C) | | | 4096 | | | | | | | 4.75 | 26.2 | 90 |
| (D) | | | | | | | 0.0 | | | 5.77 | 24.6 | |
| (D) | | | | | | | 0.2 | | | 4.95 | 25.5 | |
| (D) | | | | | | | | 0.0 | | 4.67 | 25.3 | |
| (D) | | | | | | | | 0.2 | | 5.47 | 25.7 | |
| (E) | *learned positional embedding instead of sinusoids* | | | | | | | | | 4.92 | 25.7 | |
| big | 6 | 1024 | 4096 | 16 | | | 0.3 | | 300K | 4.33 | 26.4 | 213 |

**Authors' conclusions, and how to read them:**
- **(A) Heads, at constant compute:** a single head is **0.9 BLEU** worse than the best setting, and "quality also drops off with too many heads" (h = 32 → d_k = 16).
- **(B) Smaller d_k (d_v kept at 64) hurts.** The authors read this as "determining compatibility is not easy", suggesting a better compatibility function than the dot product may help. Note that (B) also *removes parameters*, so capacity and compatibility are confounded.
- **(C) Bigger is better.** Depth, width and d_ff all help, and d_ff = 4096 gives the best single-change BLEU (26.2).
- **(D) Dropout is essential** (0.0 → −1.2 BLEU). **Label smoothing:** ε = 0 gives the *best* perplexity (4.67) but lower BLEU (25.3). ε = 0.2 gives worse perplexity (5.47) but higher BLEU (25.7) than ε = 0. This illustrates the perplexity/BLEU tension.
- **(E) Learned ≈ sinusoidal** (25.7 vs 25.8).

**Critical reading.**
- These are **single runs** with no variance reported. Differences of ≤ 0.3 BLEU (for example (A) h = 16 vs base, or (E)) are within typical seed-to-seed noise for WMT (often ±0.2–0.5). Only the large effects (h = 1, N = 2, dropout = 0, d_model = 256) are clearly supported.
- **Parameter-count discrepancy.** By our analytic formula ([math.md §9](math.md#9-parameter-count-formulas)), the transformer layers of the base model hold **44.14M** parameters, so "65M" implies an embedding matrix of about 20.9M, i.e. **V ≈ 40.7k**, not "about 37000". Rows (C) d_model = 256 and 1024 imply V ≈ 41k, while the big row implies V ≈ 36k. Reported counts are rounded and the exact accounting (biases, vocabulary padding, possibly untied matrices) is not specified. **Our 37k-vocab base model will have about 63.1M parameters**, and we will say so rather than force-match 65M.

**Repro note.** We reproduce **trends**, not absolute numbers, for rows (A), (B), (D), (E), plus a Post-LN/Pre-LN row the paper doesn't have. Every row in our table gets ≥ 3 seeds where we claim an effect (spec §7.3 and §14).

### 7.3 §6.3 English constituency parsing
**Paper says.**
- The task is hard because "the output is subject to strong structural constraints and is significantly longer than the input", and RNN seq2seq models had failed to reach state of the art in small-data regimes.
- **4-layer** Transformer, **d_model = 1024**, on WSJ (Penn Treebank), about **40K** sentences. A semi-supervised variant uses about **17M** sentences. Vocabulary: **16K** (WSJ only) and **32K** (semi-supervised).
- Little tuning: only dropout ("both attention and residual"), learning rate and beam size, tuned on Section 22. Everything else is the EN-DE base configuration.
- Inference: max output length = input + **300**, **beam 21**, **α = 0.3**.

**Table 4 (Transformer rows; WSJ Section 23 F1):** WSJ only **91.3** (beats BerkeleyParser 90.4; below the RNN Grammar at 91.7). Semi-supervised **92.7**. Best listed overall: RNNG generative, 93.3.

**Repro note.** Out of scope. This section is the source of the **attention-dropout evidence** (§4.5 above).

---

## 8. §7 Conclusion and future work
The authors planned to (1) apply the Transformer to other modalities (images, audio, video), (2) study **local, restricted attention** for large inputs and outputs, and (3) make generation **less sequential**. All three later became major research areas: vision transformers, sparse and local attention, and non-autoregressive and speculative decoding.

## 9. Appendix: Attention visualizations
- **Figure 3:** encoder self-attention, **layer 5 of 6**. Many heads attend from "making" to the distant "more difficult", i.e. they follow a long-distance dependency.
- **Figure 4:** two heads in layer 5 apparently doing **anaphora resolution** ("its" → "Law"/"application"), with very sharp attention.
- **Figure 5:** different heads learn behaviour that looks related to sentence structure.

**Critical reading.** These are hand-picked qualitative examples. Later work (Jain & Wallace 2019, *Attention is not Explanation*; Wiegreffe & Pinter 2019) cautions against reading attention weights as explanations. **Repro note:** we reproduce these visualizations (spec §11.3) and pair them with a quantitative measure (per-head entropy), not stories alone.

---

## 10. Claims register

| ID | Claim | Evidence in paper | Strength | Tested here? | How |
|---|---|---|---|---|---|
| C1 | Higher translation quality than RNN/CNN systems | Table 2 | Strong (large margins) | **Yes** | BLEU vs parameter-matched LSTM/GRU baselines on IWSLT14 |
| C2 | Much lower training cost | Table 2 FLOP estimates | Moderate (estimated, not measured) | **Yes** | Measured tokens/s and time-to-target-BLEU |
| C3 | O(1) sequential ops → more parallelism | Table 1 (analytic) | Strong (by construction) | **Yes** | Throughput vs sequence length |
| C4 | Short path length helps long-range dependencies | Argued in §4; no direct experiment | Weak to moderate | **Yes** | BLEU by source-length bucket |
| C5 | Multi-head > single-head; too many heads hurts | Table 3 (A) | Moderate (single runs) | **Yes** | Ablation, 3 seeds |
| C6 | Reducing d_k hurts | Table 3 (B) | Moderate (confounded with parameters) | **Yes** | Ablation, 3 seeds |
| C7 | Learned PE ≈ sinusoidal PE | Table 3 (E) | Moderate | **Yes** | Ablation |
| C8 | Sinusoidal PE may extrapolate to longer sequences | **Not tested in the paper** | Hypothesis | Extension | Train on short, test on long |
| C9 | Label smoothing hurts perplexity, helps BLEU | Table 3 (D) | Moderate | **Yes** | Ablation |
| C10 | Dropout is essential | Table 3 (D) | Strong | **Yes** | Ablation |
| C11 | Generalizes to constituency parsing | Table 4 | Moderate | No | Out of scope |
| C12 | Heads learn interpretable roles | Figures 3–5 (qualitative) | Weak | Partially | Visualizations + entropy statistics |

---

## 11. Key observations from a critical reading

1. **Recipe vs architecture.** Many "Transformer" results depend on details outside the architecture: warmup, β₂, label smoothing, checkpoint averaging, beam search, token batching. A from-scratch implementation that skips them will underperform, and that will look like an architecture bug.
2. **No uncertainty estimates.** No seeds or confidence intervals are reported anywhere. That was normal for 2017, but our reproduction should do better.
3. **Dev-set ablations without checkpoint averaging.** Table 3 numbers (base 25.8, dev) are not comparable with Table 2 (27.3, test, averaged).
4. **The metric is underspecified.** No BLEU tool, tokenization or casing is stated in the paper.
5. **Post-LN works here because of warmup.** The paper does not discuss it, but later analysis (Xiong et al., 2020) showed Post-LN is unstable without warmup. This becomes one of our experiments.
6. **Internal inconsistencies:** 41.8 vs 41.0 (EN-FR); "about 37000" vocabulary vs parameter counts implying about 36–41k; "three types" of regularization with two described.

---

## 12. Underspecification register

| # | Detail | Paper says | Common reference practice | **Our decision** | Where recorded |
|---|---|---|---|---|---|
| U1 | Weight initialization | Nothing | Xavier-uniform for matrices; embeddings N(0, d^-½) | Same | **[ADR-011](decisions/ADR-011-initialization.md)** |
| U2 | Attention dropout | Mentioned only in §6.3 | 0.1 (tensor2tensor v2, Annotated Transformer); off in fairseq IWSLT | **Default 0.0, configurable, ablated** (Week 1's provisional 0.1 superseded) | **[ADR-012](decisions/ADR-012-dropout-locations.md)** |
| U3 | FFN inner dropout | Nothing | tensor2tensor `relu_dropout` | Default 0.0, configurable, ablated | **[ADR-012](decisions/ADR-012-dropout-locations.md)** |
| U4 | Final LayerNorm per stack | Not in the Post-LN formula | Present in Pre-LN implementations | None for Post-LN; present for Pre-LN | ADR-003 |
| U5 | Masking value | "−∞" | Large negative finite value | `torch.finfo(dtype).min` via `masked_fill` | math.md §1.3 |
| U6 | Label smoothing distribution | ε_ls = 0.1 only | ε/K (Szegedy, PyTorch) or ε/(K−2) (Annotated Transformer) | **ε/K over the full vocabulary**; pad positions excluded from the loss | math.md §8 |
| U7 | Loss normalization | Nothing | Per non-pad target token | Per non-pad target token | Week 4 |
| U8 | LR step indexing / accumulation | Formula only | Step counts optimizer updates, starting at 1 | Same | math.md §7 |
| U9 | Gradient clipping | Nothing | Usually off | Off by default; gradient norm logged | Spec §6.1 |
| U10 | Checkpoint interval | "10-minute intervals" | Step-based in most reproductions | **Step-based** (reproducible across hardware) | Week 4 |
| U11 | BLEU tool and tokenization | Nothing | tensor2tensor: tokenized + compound split | sacreBLEU (primary) + compatible tokenized BLEU (secondary) | ADR-002 |
| U12 | BPE details | "byte-pair encoding … about 37000" | subword-nmt / tensor2tensor subword | SentencePiece BPE, joint vocabulary | ADR-005 |
| U13 | Batching algorithm | "approximately 25000 source and 25000 target tokens" | Bucketed token-budget sampler | Same | Week 2 |
| U14 | LayerNorm ε | Nothing | 1e-5 (PyTorch) / 1e-6 (tensor2tensor) | 1e-5 (enables exact parity with `nn.LayerNorm` defaults) | ADR-004 |

---

## 13. Reproduction considerations

**What we can match exactly:** every architectural equation, the optimizer and schedule, the regularization, weight tying, the decoding hyperparameters, checkpoint averaging and the ablation structure.

**What we cannot match and why:**

| Gap | Reason | Mitigation |
|---|---|---|
| Dataset (WMT14 → IWSLT14) | About 100 P100-GPU-hours vs a single consumer GPU | ADR-002. Optional WMT14 stretch run. |
| Absolute BLEU | Different data, so numbers are not comparable | Compare with published IWSLT14 reference numbers under a matching protocol |
| Batch size | One GPU | Gradient accumulation |
| Exact tokenizer | Original BPE merges are unavailable | SentencePiece BPE (ADR-005) |
| Hardware numerics | P100 fp32 vs modern AMP | fp32 parity tests. AMP only after correctness is established. |

**Risks specific to this paper:**
- The Post-LN + warmup interaction: changing the batch size or warmup without care can cause divergence that looks like a bug.
- A small dataset with base dimensions overfits (that is why IWSLT configurations use dropout 0.3 and d_ff = 1024).
- Comparing BLEU across protocols is the most common way reproductions mislead people. ADR-002 handles it.

---

## 14. The 2017 Transformer vs modern Transformers

| Aspect | Paper (2017) | Typical modern practice (LLMs, ~2023–2026) | Why it changed |
|---|---|---|---|
| Overall shape | Encoder-decoder | **Decoder-only** for LLMs. Encoder-decoder survives in MT and speech. | One architecture for all tasks; simpler scaling |
| Normalization placement | **Post-LN** | **Pre-LN** | Stable gradients at depth; warmup is less critical |
| Normalization type | LayerNorm | **RMSNorm** | Cheaper; mean-centering found unnecessary |
| FFN activation | ReLU | **SwiGLU / GeGLU** (gated) | Better quality per FLOP |
| Positional information | Absolute sinusoidal, added | **RoPE** (rotary, applied to Q/K) or ALiBi | Relative position; better length generalization |
| Attention heads | Full multi-head | **GQA / MQA** (shared K/V heads) | Smaller KV cache, faster inference |
| Attention kernel | Plain matmul + softmax | **FlashAttention** (exact, IO-aware) | Memory O(n) instead of O(n²); speed |
| Biases | Everywhere | Often **removed** | Stability, simplicity |
| Optimizer | Adam, β₂ = 0.98, no weight decay | **AdamW** (decoupled weight decay) | Better regularization |
| LR schedule | Warmup + inverse-sqrt | Warmup + **cosine** or warmup-stable-decay | Better final loss for fixed budgets |
| Label smoothing | 0.1 | Rarely used in LLM pretraining | Hurts calibration and likelihood objectives |
| Dropout | 0.1–0.3 | Often **0** in large pretraining | Data ≫ parameters, so little overfitting |
| Vocabulary | 32–37k BPE / WordPiece | 100k–250k+ byte-level BPE | Multilingual, code, no unknown tokens |
| Context length | Sentence level (~100 tokens) | 8k → 1M+ tokens | Long-context applications |
| Weight tying | Three-way | Often untied in large models | Capacity is cheap at scale |
| Precision | fp32 on P100 | bf16 / fp8 mixed precision | Throughput |
| Inference | Beam search, full recomputation | KV cache, speculative decoding, sampling | Latency |
| Scale | 65M–213M parameters | 1B–1T+ parameters, MoE FFNs | Scaling laws |

**Repro note.** We implement the **2017** version faithfully first. The "2017 → modern" ablation ladder is a planned extension (spec §14.3, item 2).

---

## 15. Open questions to resolve in Week 2+

1. Pin an IWSLT14 DE-EN source (mirror + checksum) and its exact preprocessing; fall back to IWSLT2017 if needed. → Week 2.
2. Write the evaluation protocol ADR (sacreBLEU signature; reference-compatible tokenized BLEU). → Week 2.
3. Write the initialization ADR (U1). → Week 3.
4. Decide on a FFN-dropout default after a quick Multi30k check (U3). → Week 5.
