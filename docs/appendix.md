# Technical Appendix

This document provides comprehensive technical details for the Dynamic Attention Mask (DAM) framework, extracted from the research paper's appendix.

## 📊 Attention Pattern Analysis

### Pattern Observation

Attention patterns in LLaMA 3.2 3B exhibit distinct behaviors as sequence length increases:

**1. Sliding Window Patterns**
- Local attention within fixed windows
- Persistent at short lengths (32-64 tokens)
- Example: Layer 11, Head 19

**2. Transient Diagonal Patterns** 
- Irregular diagonal tilts at intermediate lengths
- Emerge at 64-128 tokens, then disappear
- Unreliable for extrapolation

**3. Vertical Attention Structures** ⭐
- Column-wise attention on specific positions
- Consistent across increasing sequence lengths
- Most suitable for pattern matching (Layer 26, Head 13)

### Pattern Extension Framework

**Diagonal Pattern Definition:**
$$P_{\text{diag}, r}(i,j) = \begin{cases}
1, & \text{if } j = i - r \text{ and } i,j \geq 0 \\
0, & \text{otherwise}
\end{cases}$$

**Vertical Pattern Definition:**
$$P_{\text{vert}, c}(i,j) = \begin{cases}
1, & \text{if } j = c \text{ and } i \geq c \\
0, & \text{otherwise}
\end{cases}$$

**Pattern Pool:** $\mathcal{P} = \{P_{\text{diag}, r} : r \in [0, L-1]\} \cup \{P_{\text{vert}, c} : c \in [0, L-1]\}$

**Similarity Score:** $\gamma_k = \frac{\sum_{i,j} M_{\ell,h}^{(i,j)} \cdot P_k^{(i,j)}}{\sum_{i,j} P_k^{(i,j)}}$

## 🔧 Feature Amplification Methods

We analyzed **9 transformation methods** for attention pattern visualization:

### Method Comparison

| Method | Formula | Quality | Speed | Stability |
|--------|---------|---------|-------|-----------|
| **Box-Cox** ⭐ | $\frac{X^\lambda - 1}{\lambda}$ | Excellent | Medium | High |
| Square Root | $\sqrt{X}$ | Good | Fast | Medium |
| Logarithmic | $\log(X)$ | Fair | Fast | Low |
| Yeo-Johnson | Complex piecewise | Excellent | Slow | High |
| Z-Score | $\frac{X - \mu}{\sigma}$ | Fair | Fast | Medium |
| Min-Max | $\frac{X - \min}{\max - \min}$ | Fair | Fast | Low |
| Average | $\frac{A}{C + \epsilon}$ | Poor | Fast | High |
| Raw Sum | $A$ | Poor | Fastest | High |
| Arcsinh | $\sinh^{-1}(X)$ | Fair | Fast | Medium |

### Box-Cox Advantages (λ = 0.5)

- **Compact Range:** Max values ~2.0, means ~0.27
- **Enhanced Contrast:** Small/medium values become visible  
- **Stable Thresholding:** Consistent parameters across datasets
- **Pattern Preservation:** Maintains structural relationships

**Quantitative Comparison (Layer 25 Head 9):**

| Metric | Square Root | Box-Cox |
|--------|-------------|---------|
| Max Value | 149.95 | 2.00 |
| Mean (non-zero) | 13.57 | 0.27 |
| Std (non-zero) | 21.91 | 0.35 |

## 📈 Experimental Results

### LongEval Performance

| Method | Context Length | Accuracy (%) | Memory (GB) | Time (s) |
|--------|---------------|-------------|-------------|----------|
| Full Attention | 4K | 82.15 | 24.2 | 45.3 |
| Full Attention | 8K | 80.11 | **OOM** | **OOM** |
| **DAM** | 4K | 81.93 | 18.7 | 31.2 |
| **DAM** | 8K | **79.66** | **22.1** | **38.7** |
| **DAM** | 16K | **78.42** | **28.3** | **52.8** |

**Key Results:**
- **Memory Efficiency:** 23% reduction at 4K, enables 8K+ where full attention fails
- **Quality Retention:** Only 0.45% accuracy drop at 8K
- **Scalability:** Linear memory growth vs. quadratic in full attention

### Ablation Studies

**Pattern Capture Length (PCL):**

| PCL | Accuracy (%) | Pattern Coverage | Cost |
|-----|-------------|------------------|------|
| 64 | 78.91 | 78.9% | Medium |
| **128** | **79.66** | **85.2%** | **Medium** |
| 256 | 79.71 | 89.1% | High |

**Matching Threshold (μ):**

| Threshold | Sparsity (%) | Accuracy (%) | Pattern Preservation |
|-----------|-------------|-------------|-------------------|
| 0.7 | 58.7 | 79.12 | Medium |
| **0.8** | **69.4** | **79.66** | **High** |
| 0.9 | 82.1 | 79.31 | Very High |

## 🧮 Mathematical Framework

### Two-Stage Algorithm

**Stage 1: Attention Accumulation**
$$\bar{A}_{\ell,h,i,j} = \frac{A_{\ell,h,i,j}}{C_{\ell,h,i,j} + \epsilon}$$

**Stage 2: Mask Generation**

1. **Feature Amplification:** $\tilde{A}_{\ell,h,i,j} = \frac{(\max(\bar{A}_{\ell,h,i,j}, \epsilon))^{0.5} - 1}{0.5}$

2. **Binarization:** $M_{\ell,h,i,j}^{\text{thresh}} = \mathbf{1}[\tilde{A}_{\ell,h,i,j} \geq \tau]$

3. **Pattern Matching:** $k^* = \arg\max_{k : \gamma_k \geq \mu} \gamma_k$

4. **Extension:** $M_{\ell,h}^{\text{final}} = \text{ExtendPattern}(P_{k^*}, L_{\text{target}})$

### Computational Complexity

**Memory Reduction:** $O(S \cdot L^2 \cdot H)$ where $S \approx 0.3$ (70% reduction)

**Practical Speedup:** $S_{\text{practical}} \approx 2.25\times$ for $S = 0.3$

## ⚙️ Implementation Guidelines

### Recommended Configuration

```yaml
# Pattern Extraction
pattern_capture_length: 128
num_samples: 1000

# Transformation  
method: "box_cox"
box_cox_lambda: 0.5

# Mask Generation
target_length: 8192
matching_threshold: 0.8
percentile_threshold: 50
```

### Usage Pipeline

```bash
# Step 1: Extract patterns
python scripts/1_generate_attn_map.py --max_length 128 --num_samples 1000

# Step 2: Generate masks  
python scripts/2_generate_attn_mask.py --threshold 0.8 --transformation "box_cox"

# Step 3: Create DAM model
python scripts/3_generate_dam_model.py --mask_dir "./masks" --output_dir "./dam_model"
```

### Performance Tuning

**For Quality (Accuracy Priority):**
- PCL = 256, threshold = 0.7, lambda = 0.4

**For Speed (Efficiency Priority):**  
- PCL = 64, threshold = 0.9, lambda = 0.5

**For Memory (Resource Constrained):**
- target_length = 2048, threshold = 0.85, gradient_checkpointing = true

## 🎯 Use Cases and Limitations

### Successful Applications
- **Document QA:** 75-85% sparsity, <1% accuracy loss
- **Sequential Reasoning:** 60-70% sparsity, <2% accuracy loss  
- **Multi-Document Tasks:** 65-75% sparsity, <1.5% accuracy loss

### Known Limitations
- **Dense Global Dependencies:** 15-20% accuracy drop
- **Irregular Patterns:** 10-15% accuracy drop
- **Very Short Sequences (<1K):** Minimal benefit, possible overhead

### System Requirements

| Target Length | GPU Memory | Recommended Hardware |
|---------------|------------|---------------------|
| 4K | 16-24 GB | RTX 4090 |
| 8K | 24-32 GB | A100 40GB |
| 16K | 40-48 GB | A100 80GB |

---

## 📝 Citation

When using this technical appendix, please cite the original DAM paper:

```bibtex
@article{dam2024,
  title={Dynamic Attention Mask: Efficient Long-Context Inference via Pattern Extension},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
``` 