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
```
P_diag,r(i,j) = { 1, if j = i - r and i,j ≥ 0
                 { 0, otherwise
```

**Vertical Pattern Definition:**
```
P_vert,c(i,j) = { 1, if j = c and i ≥ c  
                 { 0, otherwise
```

**Pattern Pool:** 
```
P = {P_diag,r : r ∈ [0, L-1]} ∪ {P_vert,c : c ∈ [0, L-1]}
```

**Similarity Score:** 
```
γ_k = (Σ_i,j M_ℓ,h(i,j) · P_k(i,j)) / (Σ_i,j P_k(i,j))
```

## 🔧 Feature Amplification Methods

We analyzed **9 transformation methods** for attention pattern visualization:

### Method Comparison

| Method | Formula | Quality | Speed | Stability |
|--------|---------|---------|-------|-----------|
| **Box-Cox** ⭐ | `(X^λ - 1) / λ` | Excellent | Medium | High |
| Square Root | `√X` | Good | Fast | Medium |
| Logarithmic | `log(X)` | Fair | Fast | Low |
| Yeo-Johnson | Complex piecewise | Excellent | Slow | High |
| Z-Score | `(X - μ) / σ` | Fair | Fast | Medium |
| Min-Max | `(X - min) / (max - min)` | Fair | Fast | Low |
| Average | `A / (C + ε)` | Poor | Fast | High |
| Raw Sum | `A` | Poor | Fastest | High |
| Arcsinh | `sinh⁻¹(X)` | Fair | Fast | Medium |

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
```
Ā_ℓ,h,i,j = A_ℓ,h,i,j / (C_ℓ,h,i,j + ε)
```

**Stage 2: Mask Generation**

1. **Feature Amplification:** 
   ```
   Ã_ℓ,h,i,j = (max(Ā_ℓ,h,i,j, ε)^0.5 - 1) / 0.5
   ```

2. **Binarization:** 
   ```
   M_ℓ,h,i,j^thresh = 1[Ã_ℓ,h,i,j ≥ τ]
   ```

3. **Pattern Matching:** 
   ```
   k* = argmax{k : γ_k ≥ μ} γ_k
   ```

4. **Extension:** 
   ```
   M_ℓ,h^final = ExtendPattern(P_k*, L_target)
   ```

### Computational Complexity

**Memory Reduction:** `O(S · L² · H)` where `S ≈ 0.3` (70% reduction)

**Practical Speedup:** `S_practical ≈ 2.25×` for `S = 0.3`

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
@misc{zhang2025damdynamicattentionmask,
      title={DAM: Dynamic Attention Mask for Long-Context Large Language Model Inference Acceleration}, 
      author={Hanzhi Zhang and Heng Fan and Kewei Sha and Yan Huang and Yunhe Feng},
      year={2025},
      eprint={2506.11104},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.11104}, 
}
``` 