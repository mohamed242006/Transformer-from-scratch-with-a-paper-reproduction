# Mathematical Foundations

Derivations behind every component this repository implements. Each result is stated, derived, explained intuitively, and linked to the **implementation consequence** it creates. Implementation consequences are written as testable statements so they can become unit tests in Weeks 2–4.

> Rendering: this file uses GitHub-flavoured Markdown math (`$…$`, `$$…$$`).

---

## 0. Notation and conventions

| Symbol | Meaning |
|---|---|
| $n, m$ | Source and target sequence lengths |
| $d$ | Model width $d_\text{model}$ |
| $h$ | Number of heads; $d_k = d_v = d/h$ |
| $d_\text{ff}$ | FFN inner width |
| $N$ | Layers per stack |
| $V$ | Vocabulary size (joint) |
| $X \in \mathbb{R}^{n \times d}$ | Sequence of row vectors (one row per position) |
| $\mathbf{1}$ | All-ones vector |
| $\operatorname{softmax}$ | Applied **row-wise** (over the last axis) |

**Row-vector convention.** We write $XW$ with $W \in \mathbb{R}^{d_\text{in} \times d_\text{out}}$, matching the paper. PyTorch's `nn.Linear` stores $W^\top$ (shape `[d_out, d_in]`) and computes $XW^\top + b$. Every weight-copying parity test must account for this transpose.

**Mask convention.** A boolean mask $M$ with $M_{ij} = \texttt{True}$ means **query $i$ may attend to key $j$**.

---

## 1. Scaled dot-product attention

### 1.1 Definition
Given queries $Q \in \mathbb{R}^{n_q \times d_k}$, keys $K \in \mathbb{R}^{n_k \times d_k}$ and values $V \in \mathbb{R}^{n_k \times d_v}$:

$$
S = \frac{QK^\top}{\sqrt{d_k}} \in \mathbb{R}^{n_q \times n_k},
\qquad
A = \operatorname{softmax}(S),
\qquad
\operatorname{Attention}(Q,K,V) = AV \in \mathbb{R}^{n_q \times d_v}.
$$

Row $i$ of the output is

$$
o_i = \sum_{j=1}^{n_k} A_{ij}\, v_j,
\qquad
A_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}{\sum_{j'} \exp(q_i \cdot k_{j'} / \sqrt{d_k})}.
$$

### 1.2 Intuition: a differentiable dictionary lookup
Each $A_{i\cdot}$ is a probability distribution over keys, so $o_i$ is a **convex combination** of the value vectors and lies in their convex hull. A hard lookup would return the single $v_j$ whose key matches best. Softmax replaces that $\arg\max$ with a smooth approximation, which makes "which memory to read" learnable by gradient descent. It can also be seen as kernel smoothing with the kernel $\kappa(q,k) = \exp(q\cdot k/\sqrt{d_k})$.

### 1.3 Masking
Softmax is invariant to adding a constant to all logits in a row, $\operatorname{softmax}(s + c\mathbf{1}) = \operatorname{softmax}(s)$, but **not** to adding to individual entries. Setting $S_{ij} \to -\infty$ gives $A_{ij} = e^{-\infty}/Z = 0$ exactly, while the remaining weights renormalize over the allowed keys:

$$
A_{ij} = \frac{M_{ij}\, e^{S_{ij}}}{\sum_{j'} M_{ij'}\, e^{S_{ij'}}}.
$$

**Edge case:** if a row is fully masked ($M_{i\cdot} = \mathbf{0}$), the expression is $0/0$. With literal $-\infty$, PyTorch returns `NaN`.

**Implementation consequences**
- Use `masked_fill(~M, torch.finfo(dtype).min)`, not `-inf` and not `-1e9` (which overflows to `-inf` in fp16, whose maximum is 65504). Do **not** *add* a large negative number to the scores: $S + \text{finfo.min}$ can overflow to $-\infty$.
- **Test:** padding invariance. Changing the token at any masked key position leaves every output row unchanged.
- **Test:** causality. With a causal mask, perturbing input position $t$ leaves outputs at positions $< t$ unchanged.

**Causal mask.** $M^\text{causal}_{ij} = [\, j \le i \,]$: lower-triangular **including** the diagonal, since position $i$ may attend to itself. Combined with target padding, $M^\text{dec} = M^\text{causal} \wedge M^\text{pad}$, broadcast over the batch and head dimensions.

### 1.4 Softmax Jacobian
For $p = \operatorname{softmax}(s)$:

$$
\frac{\partial p_i}{\partial s_j} = p_i(\delta_{ij} - p_j)
\quad\Longleftrightarrow\quad
J = \operatorname{diag}(p) - pp^\top.
$$

$J$ is symmetric positive semi-definite with $J\mathbf{1} = 0$ (shift invariance). As $p \to e_k$ (one-hot), **$J \to 0$**: a saturated softmax passes back no gradient. Section 2 builds on this.

---

## 2. Why divide by √d_k

### 2.1 The variance argument (paper, footnote 4)
Assume at initialization that the components $q_i, k_i$ are independent, with $\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0$ and $\operatorname{Var}(q_i) = \operatorname{Var}(k_i) = 1$. Then

$$
\mathbb{E}[q\cdot k] = \sum_{i=1}^{d_k} \mathbb{E}[q_i]\,\mathbb{E}[k_i] = 0,
$$

$$
\operatorname{Var}(q\cdot k) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i)
= \sum_{i=1}^{d_k} \Big(\mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] - \mathbb{E}[q_i]^2\mathbb{E}[k_i]^2\Big)
= \sum_{i=1}^{d_k} (1\cdot 1 - 0) = d_k .
$$

(Independence of the terms across $i$ lets the variance of the sum split into a sum of variances.) So $\operatorname{std}(q\cdot k) = \sqrt{d_k}$, and

$$
\operatorname{Var}\!\left(\frac{q\cdot k}{\sqrt{d_k}}\right) = 1 .
$$

### 2.2 Why large logits hurt
With $d_k = 64$ and no scaling, the logits have standard deviation 8. Two logits two standard deviations apart (a gap of 16) give a probability ratio of $e^{16} \approx 8.9 \times 10^6$, so softmax is effectively one-hot. By §1.4, $J \approx 0$: almost no gradient reaches $Q$ and $K$, and the attention pattern stops learning.

More formally, write $s = \sigma u$ with fixed $u$ and a unique maximum $u_k$, and let $\Delta = u_k - \max_{j\ne k} u_j > 0$. Then $1 - p_k = O(e^{-\sigma\Delta})$ and every entry of $J$ is $O(e^{-\sigma\Delta})$, so **gradients decay exponentially with the logit scale**.

### 2.3 Why not some other constant?
The scale only needs to make the logits $O(1)$ at initialization, and $\sqrt{d_k}$ is the choice that does exactly that under the stated assumptions. It works as a **temperature** that the network can still adjust by growing or shrinking $W^Q, W^K$. Dot-product attention without scaling loses to additive attention as $d_k$ grows (paper §3.2.1), while additive attention $v^\top \tanh(W_q q + W_k k)$ is bounded by construction ($\tanh \in [-1,1]$) and has no such problem.

**Caveat.** The argument holds *at initialization*. During training, the norms of $q$ and $k$ drift, which is why later large-scale work adds QK-normalization. Not needed at our scale.

**Implementation consequences**
- **Test:** with $Q, K \sim \mathcal{N}(0, 1)$ and $d_k = 512$, the empirical variance of the scaled scores is $1 \pm 0.1$.
- Ablation idea (cheap, instructive): remove the scale and log attention entropy plus the gradient norm reaching $W^Q$.

---

## 3. Multi-head attention

### 3.1 Definition

$$
\operatorname{head}_i = \operatorname{Attention}(XW_i^Q,\; YW_i^K,\; YW_i^V),
\qquad
\operatorname{MultiHead}(X, Y) = \operatorname{Concat}(\operatorname{head}_1,\dots,\operatorname{head}_h)\,W^O
$$

with $W_i^Q, W_i^K \in \mathbb{R}^{d\times d_k}$, $W_i^V \in \mathbb{R}^{d \times d_v}$ and $W^O \in \mathbb{R}^{hd_v\times d}$. Here $X$ supplies the queries and $Y$ the keys and values: $Y = X$ for self-attention, and $Y$ = encoder memory for cross-attention.

### 3.2 Stacked projections: one matmul, then reshape
Concatenate the per-head matrices column-wise:

$$
W^Q = \big[\,W_1^Q \;\big|\; W_2^Q \;\big|\; \cdots \;\big|\; W_h^Q\,\big] \in \mathbb{R}^{d \times hd_k} = \mathbb{R}^{d\times d}.
$$

Then $XW^Q$ contains every head's queries side by side, and columns $[(i-1)d_k,\, i\,d_k)$ belong to head $i$. Reshaping $[n, d] \to [n, h, d_k]$ and transposing to $[h, n, d_k]$ separates the heads. **The transpose is essential:** reshaping directly to $[h, n, d_k]$ produces the right shape with **scrambled data**.

### 3.3 The output projection mixes heads
Partition $W^O$ into row blocks $W^O_i \in \mathbb{R}^{d_v \times d}$:

$$
\operatorname{Concat}(\operatorname{head}_1,\dots,\operatorname{head}_h)\,W^O = \sum_{i=1}^{h} \operatorname{head}_i\, W_i^O .
$$

So multi-head attention is a **sum of h independent attention read-outs**, each written back into the residual stream through its own $d_v \times d$ map.

### 3.4 Each head is a low-rank bilinear form
The logit for head $i$ between positions $a$ and $b$ is

$$
s^{(i)}_{ab} = \frac{1}{\sqrt{d_k}}\; x_a\, \underbrace{W_i^Q {W_i^K}^\top}_{\in\, \mathbb{R}^{d\times d},\ \operatorname{rank} \le d_k}\, y_b^\top .
$$

A single full-width head has rank $\le d$ but only **one** softmax. $h$ heads have $h$ separate softmaxes, each of rank $\le d/h$. Table 3 (A) of the paper is the empirical trade-off between these: $h=1$ blurs ("averaging inhibits this", §3.2.2), while $h=32$ ($d_k = 16$) makes each comparison too low-rank.

### 3.5 Cost is (almost) independent of h
| Operation | Multiply-accumulates (MACs) |
|---|---|
| Q, K, V projections | $3\, n d^2$ (for self-attention) |
| $QK^\top$ over all heads | $h \cdot n^2 d_k = n^2 d$ |
| $AV$ over all heads | $h \cdot n^2 d_v = n^2 d$ |
| Output projection | $n d^2$ |
| **Total** | $4nd^2 + 2n^2 d$ |

None of these terms depend on $h$, which confirms the paper's claim that the cost is "similar to that of single-head attention with full dimensionality". Memory for the attention weights is $h \cdot n^2$ per sequence, so it **does** grow with $h$.

---

## 4. Positional encoding

### 4.1 Definition
For $i = 0, \dots, d/2 - 1$, define frequencies $\omega_i = 10000^{-2i/d}$ and

$$
\mathrm{PE}(p, 2i) = \sin(\omega_i\, p), \qquad \mathrm{PE}(p, 2i+1) = \cos(\omega_i\, p).
$$

Computed stably in log space: $\omega_i = \exp\!\big(-\tfrac{2i}{d}\ln 10000\big)$.

### 4.2 Geometric wavelengths
The wavelength of pair $i$ is $\lambda_i = 2\pi/\omega_i = 2\pi \cdot 10000^{2i/d}$, a geometric progression from $\lambda_0 = 2\pi$ to $\lambda_{d/2-1} = 2\pi\cdot 10000^{1 - 2/d}$, which is just under $2\pi \cdot 10000$ (the paper rounds this to "$10000\cdot 2\pi$"). Low $i$ means fast oscillation and fine position resolution. High $i$ means slow oscillation and coarse, nearly monotone position over sentence-length ranges. This works like a continuous binary counter in which each bit flips at a different rate.

### 4.3 Relative positions are linear maps (the paper's key claim)
Look at one frequency pair $\phi_i(p) = \big(\sin\omega_i p,\; \cos\omega_i p\big)^\top$. By the angle-addition identities,

$$
\phi_i(p+k) =
\underbrace{\begin{pmatrix} \cos\omega_i k & \sin\omega_i k \\ -\sin\omega_i k & \cos\omega_i k \end{pmatrix}}_{R_i(k)}
\phi_i(p).
$$

*Check (row 1):* $\sin(\omega p + \omega k) = \sin\omega p \cos\omega k + \cos\omega p \sin\omega k$ ✓. *(Row 2):* $\cos(\omega p + \omega k) = \cos\omega p\cos\omega k - \sin\omega p \sin\omega k$ ✓.

Stacking all pairs gives $\mathrm{PE}(p+k) = R(k)\,\mathrm{PE}(p)$ with the block-diagonal rotation $R(k) = \operatorname{diag}(R_0(k),\dots,R_{d/2-1}(k))$. **$R(k)$ depends only on the offset $k$, not on $p$.** A linear layer can therefore implement "shift by $k$" uniformly at every position, which is what attention needs in order to look at a neighbour.

### 4.4 Dot products depend only on distance

$$
\mathrm{PE}(p)\cdot\mathrm{PE}(q) = \sum_{i=0}^{d/2-1} \big(\sin\omega_i p \sin\omega_i q + \cos\omega_i p\cos\omega_i q\big) = \sum_{i=0}^{d/2-1} \cos\big(\omega_i (p - q)\big).
$$

This is a function of $p - q$ only. It is symmetric ($p-q$ and $q-p$ give the same value), so raw PE similarity encodes **distance but not direction**. Direction has to come from the learned projections $W^Q, W^K$, which break the symmetry. In particular $\lVert \mathrm{PE}(p)\rVert^2 = d/2$ for every $p$: the encoding is bounded and has the same norm at every position.

### 4.5 Why add rather than concatenate?
Concatenation would cost extra width. Addition keeps the width at $d$ and relies on high-dimensional geometry: random-ish token embeddings and the PE manifold occupy nearly orthogonal subspaces, and the first linear maps can learn to separate them. The trade-off is that content and position are entangled from layer 1. RoPE (modern practice) instead applies $R(p)$ multiplicatively to $Q$ and $K$, so that $q_a\cdot k_b$ depends on content and on $a-b$ exactly.

**Implementation consequences**
- Store the table as a non-trainable `register_buffer`. **Test:** it does not appear in `model.parameters()`.
- **Test:** $\lVert\mathrm{PE}(p+k) - R(k)\,\mathrm{PE}(p)\rVert_\infty < 10^{-5}$ for random $p, k$.
- **Test:** $\mathrm{PE}(p)\cdot\mathrm{PE}(q)$ equals $\sum_i\cos(\omega_i(p-q))$.
- Require even $d$. `max_len` ≥ longest source + 50 (inference length bound).

---

## 5. Embeddings, weight tying and initialization

### 5.1 Three-way weight tying
One matrix $E\in\mathbb{R}^{V\times d}$ serves as the source embedding, the target embedding, and the output projection $\text{logits} = h E^\top$. It saves $2Vd$ parameters (37.9M for the base model with $V = 37{,}000$), and it ties the input and output representations of each token (Press & Wolf, 2017).

### 5.2 Initialization variance
With $E_{vj} \sim \mathcal{N}(0, d^{-1})$:
- **Input side:** $\sqrt{d}\,E_v$ has per-dimension variance $d \cdot d^{-1} = 1$.
- **Output side:** with decoder output $h$ whose dimensions have unit variance (after Post-LN with $\gamma = 1$, $\lVert h\rVert^2 \approx d$), logit $z_v = h \cdot E_v$ has variance $\lVert h \rVert^2 \cdot d^{-1} \approx 1$.

Both directions are well scaled, **which is why $\sqrt{d}$ is applied on the input side only**.

### 5.3 Expected initial loss
If $z_v \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2)$, then for large $V$

$$
\mathbb{E}[\mathcal{L}_0] = \mathbb{E}\Big[\log \textstyle\sum_v e^{z_v} - z_y\Big] \approx \log\!\big(V\,\mathbb{E}[e^{z}]\big) - 0 = \ln V + \frac{\sigma^2}{2}.
$$

With $\sigma^2 \approx 1$: **$\mathcal{L}_0 \approx \ln V + 0.5$** nats (≈ 9.71 for $V = 10{,}000$). A model whose initial loss is far from this, for example ≫ 15, has a scaling bug (a missing $\sqrt{d}$ in the wrong place, or the scale applied on the output side).

**Implementation consequence.** Initial-loss test: $\mathcal{L}_0 \in [\ln V,\; \ln V + 1.5]$, not equality with $\ln V$.

### 5.4 Interaction with embedding scaling
Each PE coordinate has mean square $\tfrac12$ (the average of $\sin^2$ and $\cos^2$). After scaling, each token-embedding coordinate has variance 1. Their sum therefore has signal-to-position power ratio ≈ 2:1, so neither drowns out the other. **Without** the $\sqrt{d}$ factor, the token signal would have variance $1/d \approx 0.002$ against PE's 0.5, and the model would initially "see" mostly position.

Remaining matrices (attention projections, FFN) use Xavier-uniform: $W_{ij} \sim \mathcal{U}\big(-\sqrt{6/(d_\text{in}+d_\text{out})},\, +\sqrt{6/(d_\text{in}+d_\text{out})}\big)$, which keeps activation variance roughly constant in both directions. Biases start at zero; LayerNorm starts at $\gamma = 1$, $\beta = 0$.

---

## 6. Layer normalization and residual connections

### 6.1 Definition
For one position's vector $x \in \mathbb{R}^d$:

$$
\mu = \frac1d\sum_{j} x_j,\qquad
\sigma^2 = \frac1d\sum_j (x_j - \mu)^2,\qquad
\hat x = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}},\qquad
\operatorname{LN}(x) = \gamma\odot \hat x + \beta .
$$

The statistics are taken **over features, per token**, independently of batch and position. The variance is **biased** ($1/d$, not $1/(d-1)$). $\epsilon = 10^{-5}$ (the PyTorch default, which allows exact parity tests).

**Contrast with BatchNorm:** BatchNorm normalizes each feature across the batch, which couples examples together, behaves differently in training and evaluation, and is ill-defined with variable-length padding. LayerNorm has none of these problems, which makes it the natural choice for sequences.

### 6.2 Invariances
For $\alpha > 0$ and scalar $c$ (ignoring $\epsilon$): $\operatorname{LN}(\alpha x + c\mathbf{1}) = \operatorname{LN}(x)$. The output does not depend on the input's scale or offset.

### 6.3 Gradient
Let $g = \gamma \odot \partial\mathcal{L}/\partial y$ and $s = \sqrt{\sigma^2+\epsilon}$. Then

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{1}{s}\Big(g - \bar g\,\mathbf{1} - \overline{(g\odot\hat x)}\;\hat x\Big),
\qquad \bar{a} := \tfrac1d \textstyle\sum_j a_j .
$$

*Derivation sketch:* $\hat x$ depends on $x$ directly, and through $\mu$ and $\sigma$. Differentiating gives $\partial \hat x_i/\partial x_j = \frac{1}{s}\big(\delta_{ij} - \frac1d - \frac1d \hat x_i \hat x_j\big)$. Contracting with $g$ gives the expression above.

**Consequences:** the gradient is orthogonal to $\mathbf{1}$ and to $\hat x$ (it removes exactly the directions LN is invariant to), and it is scaled by $1/s$. Large activations therefore receive proportionally smaller gradients. This is the source of LN's stabilizing effect, and also of its interaction with residual paths.

### 6.4 Residual paths: Post-LN vs Pre-LN

**Post-LN (paper):** $x_{l+1} = \operatorname{LN}\big(x_l + F_l(x_l)\big)$, so

$$
\frac{\partial x_{L}}{\partial x_l} = \prod_{k=l}^{L-1} J_{\mathrm{LN},k}\,\big(I + J_{F,k}\big).
$$

Every layer's LayerNorm Jacobian sits **on the residual path**, so there is no pure identity route from the loss to early layers.

**Pre-LN:** $x_{l+1} = x_l + F_l(\operatorname{LN}(x_l))$, so $x_L = x_l + \sum_{k=l}^{L-1} F_k(\operatorname{LN}(x_k))$ and

$$
\frac{\partial x_L}{\partial x_l} = I + \sum_{k} (\cdots),
$$

which contains an **exact identity term**: gradients reach every layer undiminished.

Xiong et al. (2020, *On Layer Normalization in the Transformer Architecture*) show that at initialization, the expected gradient norm of the parameters near the output is roughly independent of depth for Post-LN but shrinks like $1/\sqrt{L}$ for Pre-LN. Large updates to the top layers at the start of training explain **why Post-LN needs learning-rate warmup**, and why Pre-LN can often train without it. We test this directly (ADR-003).

---

## 7. Noam learning-rate schedule

### 7.1 Definition and piecewise form
With step $s \ge 1$ counting **optimizer updates** and warmup $w$:

$$
\eta(s) = d^{-1/2}\,\min\!\big(s^{-1/2},\; s\, w^{-3/2}\big)
= \begin{cases} d^{-1/2}\, w^{-3/2}\, s & s \le w \quad\text{(linear warmup)}\\[2pt] d^{-1/2}\, s^{-1/2} & s \ge w \quad\text{(inverse-sqrt decay)}\end{cases}
$$

The branches meet where $s^{-1/2} = s\,w^{-3/2} \iff s = w$.

### 7.2 Peak value and normalized form
$$
\eta^\star = \eta(w) = (d\,w)^{-1/2},
\qquad
\eta(s) = \eta^\star \cdot \min\!\Big(\frac{s}{w},\; \sqrt{\frac{w}{s}}\Big).
$$

The second form (peak × shape) is the one fairseq calls `inverse_sqrt`. It lets us set the peak directly for small datasets (spec §6.2).

### 7.3 Values for the base configuration ($d = 512$, $w = 4000$)
| Step $s$ | $\eta(s)$ |
|---|---|
| 1 | $1.75\times10^{-7}$ |
| 1,000 | $1.75\times10^{-4}$ |
| **4,000 (peak)** | $\mathbf{6.99\times10^{-4}}$ |
| 10,000 | $4.42\times10^{-4}$ |
| 100,000 | $1.40\times10^{-4}$ |

### 7.4 Why warmup?
1. **Adam's second moment is noisy early.** $\hat v_t$ is an average over an effective window of about $1/(1-\beta_2) = 50$ steps, but it is estimated from far fewer samples at the start. Its variance makes the per-parameter step size $\eta/\sqrt{\hat v_t}$ erratic (Liu et al., 2020, RAdam). Small early learning rates absorb this.
2. **Post-LN gradient scale** (§6.4): large gradients near the output at initialization, multiplied by a large learning rate, cause divergence.
3. **Adam's update is roughly scale-free.** $|\Delta\theta| \approx \eta$ per coordinate regardless of gradient magnitude, so the learning rate directly bounds how far parameters move per step.

**Why the $d^{-1/2}$ factor?** Wider models have more coordinates contributing to each pre-activation, so the same per-coordinate step changes outputs more. Scaling the learning rate by $d^{-1/2}$ roughly compensates. It is a heuristic, not a theorem.

**Implementation consequences**
- The schedule is a **pure function** $\eta(s)$, with the counter starting at 1 ($s = 0$ is a division by zero).
- With gradient accumulation of $a$ micro-batches, $s$ advances once per $a$ micro-batches.
- **Test:** values at $s \in \{1, w, 4w\}$ match the closed form; the maximum over $s$ occurs at $s = w$.

---

## 8. Label smoothing

### 8.1 Target distribution
For true class $y$, $K = V$ classes and smoothing $\varepsilon$:

$$
q'_k = (1-\varepsilon)\,[k = y] + \frac{\varepsilon}{K}
\qquad\Longrightarrow\qquad
q'_y = 1 - \varepsilon + \tfrac{\varepsilon}{K},\quad q'_{k\ne y} = \tfrac{\varepsilon}{K}.
$$

**Convention (our decision):** $\varepsilon/K$ over the **full** vocabulary, matching Szegedy et al. (2016) and `torch.nn.functional.cross_entropy(label_smoothing=ε)`, so PyTorch can serve as the reference implementation in tests. Positions whose *target* is padding are excluded from the loss entirely. Alternatives seen in the wild: $\varepsilon/(K-1)$ (excluding the true class) and $\varepsilon/(K-2)$ (also excluding padding, as in the Annotated Transformer). The differences are $O(\varepsilon/K)$ and immaterial for BLEU, but they matter for exact parity.

### 8.2 Loss decomposition
With $p = \operatorname{softmax}(z)$:

$$
\mathcal{L}_\varepsilon = -\sum_k q'_k \log p_k
= (1-\varepsilon)\,\underbrace{(-\log p_y)}_{\text{NLL}} + \varepsilon\,\underbrace{\Big(-\frac1K \sum_k \log p_k\Big)}_{\text{cross-entropy to uniform}} .
$$

So the smoothed loss is NLL plus a penalty that pushes the model's distribution towards uniform. Equivalently, $\mathcal{L}_\varepsilon = H(q') + \mathrm{KL}(q' \,\Vert\, p)$.

### 8.3 Gradient and optimum
$$
\frac{\partial \mathcal{L}_\varepsilon}{\partial z_k} = p_k - q'_k .
$$

The loss is minimized at $p = q'$, which gives finite logit gaps:

$$
z_y - z_k = \log\frac{q'_y}{q'_k} = \log\!\Big(1 + \frac{K(1-\varepsilon)}{\varepsilon}\Big) \approx 11.4 \quad (K = 10^4,\ \varepsilon = 0.1).
$$

Without smoothing, the optimum pushes $z_y - z_k \to \infty$. Smoothing **bounds the logits**: a regularizer against over-confidence that tends to improve calibration and beam search ranking.

### 8.4 The loss floor
$$
\min_p \mathcal{L}_\varepsilon = H(q') = -q'_y\log q'_y - (K-1)\,\frac{\varepsilon}{K}\log\frac{\varepsilon}{K}.
$$

| $K$ | $H(q')$ with $\varepsilon = 0.1$ |
|---|---|
| 8,000 | ≈ 1.22 nats |
| 10,000 | ≈ 1.25 nats |
| 37,000 | ≈ 1.38 nats |

**Implementation consequences (important for testing)**
- An **overfit-one-batch test with $\varepsilon = 0.1$ cannot reach zero loss.** It plateaus at about $H(q')$. Run overfit tests with $\varepsilon = 0$, or assert convergence to $H(q')$.
- **Report perplexity from the unsmoothed NLL**, $\mathrm{PPL} = \exp(\overline{\mathrm{NLL}})$, never $\exp(\mathcal{L}_\varepsilon)$. This is how label smoothing "hurts perplexity, … but improves accuracy and BLEU" (paper §5.4).
- **Test:** with $\varepsilon = 0$ our loss equals plain cross-entropy; for $\varepsilon > 0$ it equals `F.cross_entropy(..., label_smoothing=ε, ignore_index=pad)`.

---

## 9. Parameter count formulas

All counts include biases. Weight tying is on, the output projection has **no bias** (our decision), and the architecture is Post-LN, so there are no stack-final LayerNorms.

### 9.1 Components
| Component | Parameters |
|---|---|
| Multi-head attention ($W^Q, W^K, W^V, W^O$ + biases) | $4d^2 + 4d$ |
| FFN ($W_1, b_1, W_2, b_2$) | $2\,d\,d_\text{ff} + d_\text{ff} + d$ |
| LayerNorm ($\gamma, \beta$) | $2d$ |
| Shared embedding / output projection | $Vd$ |
| Positional encoding (sinusoidal) | $0$ (buffer) |

### 9.2 Layers
$$
P_\text{enc layer} = \underbrace{4d^2 + 4d}_{\text{self-attn}} + \underbrace{2dd_\text{ff} + d_\text{ff} + d}_{\text{FFN}} + \underbrace{4d}_{2\ \text{LN}} = 4d^2 + 2dd_\text{ff} + d_\text{ff} + 9d
$$

$$
P_\text{dec layer} = \underbrace{8d^2 + 8d}_{\text{self + cross}} + \underbrace{2dd_\text{ff} + d_\text{ff} + d}_{\text{FFN}} + \underbrace{6d}_{3\ \text{LN}} = 8d^2 + 2dd_\text{ff} + d_\text{ff} + 15d
$$

### 9.3 Full model
$$
P_\text{total} = N\big(12d^2 + 4dd_\text{ff} + 2d_\text{ff} + 24d\big) + Vd \;\;\big[+\,4d \text{ for Pre-LN final norms}\big]\;\;\big[+\,2Vd \text{ if untied}\big].
$$

### 9.4 Worked numbers
| Config | $N$ | $d$ | $d_\text{ff}$ | $V$ | Layers | Embedding | **Total** |
|---|---|---|---|---|---|---|---|
| Beginner | 3 | 256 | 1024 | 8,000 | 5,529,600 | 2,048,000 | **7,577,600** |
| Mid (IWSLT) | 6 | 512 | 1024 | 10,000 | 31,543,296 | 5,120,000 | **36,663,296** |
| Paper base | 6 | 512 | 2048 | 37,000 | 44,138,496 | 18,944,000 | **63,082,496** |
| Paper big | 6 | 1024 | 4096 | 37,000 | 176,357,376 | 37,888,000 | **214,245,376** |

Base, per layer: encoder 3,152,384; decoder 4,204,032.

**Compared with the paper:** base 63.1M vs reported **65M**; big 214.2M vs **213M**. The base gap implies the paper's effective vocabulary was nearer 41k (see [paper_notes.md §7.2](paper_notes.md#72-62-model-variations-table-3)). We report our own count.

**Implementation consequence.** **Test:** `count_parameters(model)` equals this formula exactly for several configurations. This catches missing biases, accidentally untied weights and layers shared by reference.

### 9.5 Where the parameters live (base)
Embedding 30% · decoder layers 40% · encoder layers 30%. Within an encoder layer: FFN 67%, attention 33%.

---

## 10. Complexity analysis

### 10.1 Per-layer cost (one sequence, length $n$)
| Layer | MACs | Sequential steps | Max path length |
|---|---|---|---|
| Self-attention (incl. projections) | $4nd^2 + 2n^2d$ | $O(1)$ | $O(1)$ |
| FFN | $2nd\,d_\text{ff}$ | $O(1)$ | n/a |
| LSTM layer (4 gates, input + recurrent) | $8nd^2$ | $O(n)$ | $O(n)$ |
| Convolution (kernel $k$) | $k\,n\,d^2$ | $O(1)$ | $O(\log_k n)$ dilated |

**Crossover.** Self-attention's mixing cost $2n^2d$ is below the projection cost $4nd^2$ when $n < 2d$. For $d = 512$ that is $n < 1024$, which is true for every sentence-level MT input. In that regime the Transformer's cost is dominated by **dense matmuls**, which GPUs handle very efficiently. The paper's "$n < d$" condition is the same argument with constants dropped.

### 10.2 Memory
Training stores the attention probabilities for backpropagation: $B \cdot h \cdot n^2$ values per attention module, and there are $3N$ such modules (encoder self, decoder self, decoder cross with $n\times m$). Example: $B = 64$ sentences, $n = 64$, $h = 8$ gives 2.1M floats ≈ 8.4 MB per module in fp32, about 150 MB across 18 modules. That is small at sentence length; the quadratic term dominates only for long documents.

### 10.3 Training compute (estimate)
Using the standard approximation **training FLOPs ≈ 6 × parameters × tokens** (forward 2×, backward 4×; Kaplan et al., 2020), with encoder parameters applied to source tokens and decoder plus output-projection parameters applied to target tokens:

$$
C \approx 6\big(P_\text{enc}\,T_\text{src} + (P_\text{dec} + Vd)\,T_\text{tgt}\big)
\approx 6 \times 63\text{M} \times 2.5\text{B} \approx 9.5\times10^{17}\ \text{FLOPs}
$$

for the base model (100k steps × 25k tokens per side). The paper's cost estimate is $3.3\times10^{18}$ (time × GPUs × 9.5 TFLOPS), which implies a **hardware utilization of about 29%**. This is the same idea as *model FLOPs utilization (MFU)*, and we will compute it for our own runs.

### 10.4 Autoregressive decoding
Generating $m$ tokens:

| Strategy | Cost of step $t$ | Total |
|---|---|---|
| Recompute the full decoder each step | $O(t\,d^2 + t^2 d)$ | $O(m^2 d^2 + m^3 d)$ |
| **KV cache** (store past K, V per layer) | $O(d^2 + t\,d)$ | $O(m\,d^2 + m^2 d)$ |

The encoder runs **once** per source sentence in both cases, and cross-attention keys and values are computed once from the memory and cached. Beam size $b$ multiplies the decoder cost by $b$. The KV cache trades $O(N\, m\, d)$ extra memory per sequence for an $O(m)$ speedup: extension item 1 in the spec.

---

## Appendix A: evaluation formulas

**Perplexity.** $\mathrm{PPL} = \exp\!\big(\frac{1}{T}\sum_{t} -\log p(y_t \mid y_{<t}, x)\big)$ over the $T$ non-pad target tokens, using natural logs and no label smoothing. Perplexities are comparable **only under an identical tokenizer**.

**BLEU** (corpus level). Let $p_n$ be the clipped $n$-gram precision summed over the corpus, $c$ the total hypothesis length and $r$ the total reference length. Then

$$
\mathrm{BLEU} = \mathrm{BP}\cdot\exp\Big(\sum_{n=1}^{4}\tfrac14\log p_n\Big),
\qquad
\mathrm{BP} = \min\!\big(1,\ e^{\,1 - r/c}\big).
$$

It is **not** the mean of sentence-level BLEU scores.

**GNMT length penalty** (used in beam search, $\alpha = 0.6$). Hypotheses are ranked by $\log P(Y\mid X) / \mathrm{lp}(Y)$, where $\mathrm{lp}(Y) = \frac{(5+|Y|)^\alpha}{(5+1)^\alpha}$. Without it, beam search favours short outputs, because every extra token adds a negative log-probability.

**Checkpoint averaging.** $\bar\theta = \frac1k\sum_{i=1}^{k}\theta_{(i)}$ over the last $k$ checkpoints (here $k = 5$). It is cheap and approximates the flatter solution that SWA/Polyak averaging reaches. It works because late checkpoints lie in the same basin of the loss.
