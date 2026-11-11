# Mathematical Formulations and Statistical Protocol

This document provides the formal mathematical definitions, algorithms, and statistical protocols defined in the research paper.

---

## 1. Grounding Metrics

### 1.1 Intersection-over-Union (IoU)
For a predicted candidate region bounding box $b_{\text{pred}}$ and a ground-truth annotated structure bounding box $b_{\text{gt}}$:
$$\text{IoU}(b_{\text{pred}}, b_{\text{gt}}) = \frac{\text{Area}(b_{\text{pred}} \cap b_{\text{gt}})}{\text{Area}(b_{\text{pred}} \cup b_{\text{gt}})}$$
When pixel masks $m_{\text{pred}}$ and $m_{\text{gt}}$ are available:
$$\text{IoU}_{\text{mask}}(m_{\text{pred}}, m_{\text{gt}}) = \frac{\sum_{x, y} (m_{\text{pred}}(x, y) \land m_{\text{gt}}(x, y))}{\sum_{x, y} (m_{\text{pred}}(x, y) \lor m_{\text{gt}}(x, y))}$$

### 1.2 Pointing Accuracy
A claim is defined as *correctly grounded* if its cited region ID $r_i$ achieves $\text{IoU} \ge 0.5$ against the ground-truth annotation for that finding:
$$\text{Pointing Accuracy} = \frac{1}{|F_{\text{grounded}}|} \sum_{f \in F_{\text{grounded}}} \mathbb{I}\left[\max_{j} \text{IoU}(b_{f}, b_{\text{gt}, j}) \ge 0.5\right]$$

### 1.3 Unsupported-Claim Rate (USR)
Let $N_{\text{claims}}$ be the total number of findings emitted by the VLM. Let $N_{\text{unsup}}$ be the number of claims that fail to cite a valid region in candidate set $R$ or whose cited region has $\text{IoU} < 0.5$ with the annotated lesion:
$$\text{USR} = \frac{N_{\text{unsup}}}{N_{\text{claims}}}$$
*Note*: USR is computed in two states:
- $\text{USR}_{\text{raw}}$: Before claim filtering (evaluating raw hallucination rate of the VLM).
- $\text{USR}_{\text{filtered}}$: After the verification stage drops invalid claims.

---

## 2. Confidence Estimation and Abstention Gate

### 2.1 Multi-Sample Agreement ($a$)
Given $N$ stochastic answers sampled from the VLM at temperature $T > 0$, let answers be grouped into semantic clusters $\{C_1, \dots, C_K\}$.
The majority answer agreement $a$ is the proportion of samples belonging to the modal semantic cluster $C_{\text{modal}}$:
$$a = \frac{|C_{\text{modal}}|}{N}, \quad a \in \left[\frac{1}{N}, 1\right]$$

### 2.2 Normalized Semantic Entropy ($H_n$)
Let $p_k = \frac{|C_k|}{N}$ be the empirical probability distribution across the $K$ answer clusters. The Shannon entropy is:
$$H = -\sum_{k=1}^{K} p_k \ln p_k$$
To normalize $H$ into $[0, 1]$ independent of sample count $N$:
$$H_n = \frac{H}{\ln N}$$
where $H_n = 0$ denotes complete consensus, and $H_n = 1$ denotes uniform dispersion.

### 2.3 Region Localizer Score ($\rho$)
Let $\rho \in [0, 1]$ be the detector's objectness/confidence score for the candidate region cited by the majority answer. If multiple findings are cited, $\rho = \min_{i} \rho_i$ (conservative lower bound). If the answer is negative and cites `NONE`, $\rho = 1.0$ (no localized lesion claimed).

### 2.4 Fused Confidence Score ($c$)
The scalar confidence score $c \in [0, 1]$ is:
$$c = w_1 a + w_2 (1 - H_n) + w_3 \rho$$
subject to constraints:
$$w_1 + w_2 + w_3 = 1, \quad w_1, w_2, w_3 \ge 0$$
Weights $(w_1, w_2, w_3)$ and abstention threshold $\tau \in [0, 1]$ are optimized via grid search on a held-out calibration split to maximize the selective AUROC.

### 2.5 Decision Rule
$$\hat{y} = \begin{cases} y_{\text{majority}} & \text{if } c \ge \tau \\ \text{"uncertain"} & \text{if } c < \tau \end{cases}$$

---

## 3. Calibration and Selective Classification Metrics

### 3.1 Expected Calibration Error (ECE)
Predictions are partitioned into $M$ equally spaced confidence bins $B_1, \dots, B_M \subset [0, 1]$.
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \big| \text{acc}(B_m) - \text{conf}(B_m) \big|$$
where:
$$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbb{I}(\hat{y}_i = y_i^*), \quad \text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} c_i$$

### 3.2 Selective Classification (Accuracy vs. Coverage)
- **Coverage ($\Phi$)**: Fraction of total test instances where $c \ge \tau$:
  $$\Phi(\tau) = \frac{1}{n} \sum_{i=1}^{n} \mathbb{I}(c_i \ge \tau)$$
- **Selective Accuracy ($A(\Phi)$)**: Accuracy evaluated strictly on answered cases:
  $$A(\tau) = \frac{\sum_{i=1}^{n} \mathbb{I}(\hat{y}_i = y_i^* \land c_i \ge \tau)}{\sum_{i=1}^{n} \mathbb{I}(c_i \ge \tau)}$$
- **AUROC**: Area under the Receiver Operating Characteristic curve where the binary label is correctness ($\mathbb{I}(\hat{y}_i = y_i^*)$) and the predictor is $c_i$.

---

## 4. Statistical Significance & Inter-Rater Agreement

### 4.1 Non-Parametric Bootstrap Resampling
For all reported metrics ($\theta \in \{\text{Accuracy}, \text{F1}, \text{USR}, \text{ECE}, \text{AUROC}\}$):
1. Sample $B = 1000$ bootstrap replicates with replacement from the test set of size $n$.
2. Compute $\hat{\theta}^{*b}$ on each replicate.
3. Compute 95% percentile confidence intervals: $[\hat{\theta}_{0.025}^*, \hat{\theta}_{0.975}^*]$.

### 4.2 Paired Statistical Hypothesis Testing
To compare Qwen2.5-VL-3B against MedGemma-4B under identical splits:
- McNemar's test for binary answer correctness on closed questions ($p < 0.05$).
- Wilcoxon signed-rank test for continuous confidence and calibration metrics across test folds.

### 4.3 Cohen's Kappa ($\kappa$) for Expert & Judge Review
To evaluate inter-rater agreement between clinical reviewers (or human vs. LLM judge):
$$\kappa = \frac{p_o - p_e}{1 - p_e}$$
where $p_o$ is the observed proportional agreement and $p_e$ is the expected hypothetical agreement by chance.
