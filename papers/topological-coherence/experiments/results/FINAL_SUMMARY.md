# Toroidal Attention Topology: Complete Experimental Results

## Executive Summary

**Key Finding**: Applying Tonnetz (musical torus) topology to the final 1/3 of attention layers reduces hallucinations by **50-80%** in specific models (Phi-2, Mistral-7B) with **zero accuracy loss**.

**Critical Caveat**: Effect is highly model-specific. Only 2 of 8 tested models benefit.

---

## Complete Results Table

| Model | Company | Params | Architecture | Fine-tuning | Hallucination Change |
|-------|---------|--------|--------------|-------------|---------------------|
| TinyLlama | - | 1.1B | Llama | SFT | 0% |
| Qwen2 | Alibaba | 1.5B | Qwen | SFT | 0% |
| Gemma | Google | 2B | Gemma | IT | 0% |
| **Phi-2** | **Microsoft** | **2.7B** | **Phi** | **SFT** | **-50%** ✅ |
| OpenChat | - | 7B | Llama | C-RLFT | 0% |
| **Mistral-7B** | **Mistral AI** | **7B** | **Mistral** | **Instruct** | **-67% to -80%** ✅ |
| Zephyr | HuggingFace | 7B | Mistral+DPO | DPO+SFT | **+46%** ❌ |
| Yi-6B | 01.AI | 6B | Yi | Chat | 0% |

---

## Optimal Configuration

```
topology = "tonnetz"      # 12-tone musical torus
grid_size = 12            # Chromatic scale
radius = 2.0              # Tight local neighborhood
alpha = 1.0               # Moderate exponential decay
layers = "last_third"     # Only final 1/3 of attention layers
```

### Hyperparameter Sensitivity (Mistral-7B)

| Config | Hallucination | Notes |
|--------|---------------|-------|
| r=2, α=1 | **4%** | Optimal |
| r=2, α=0.5 | 10% | Too weak |
| r=3, α=1 | 10% | Too loose |
| r=2, α=2 | 22% | Too aggressive (worse than baseline 16%) |

---

## Key Insights

### 1. Model-Specific Effect
- Only Microsoft (Phi-2) and Mistral AI models benefit
- Llama-based models (TinyLlama, OpenChat) show zero effect
- Same architecture with different fine-tuning (Zephyr vs Mistral) shows OPPOSITE results

### 2. Fine-tuning Matters Critically
- Standard instruct fine-tuning: Compatible (Mistral, Phi-2)
- DPO/RLHF fine-tuning: Incompatible or harmful (Zephyr)
- Hypothesis: DPO optimizes attention patterns in ways that conflict with topological constraints

### 3. Layer Selectivity is Essential
- Full model application: Destroys accuracy
- Early layers only: Minimal effect
- **Late layers only**: Preserves accuracy + reduces hallucinations

### 4. The Tonnetz Connection
The 12-tone musical torus maps semantic relationships:
- Adjacent positions = harmonically related concepts
- Constraining attention to local neighborhoods forces semantic coherence
- Prevents "long-range hallucination jumps" between unrelated concepts

---

## Statistical Validation (Mistral-7B)

| Run | Samples | Baseline | layer_late | Reduction |
|-----|---------|----------|------------|-----------|
| 1 | 30 | 30% | 10% | 67% |
| 2 | 30 | 16.7% | 3.3% | 80% |
| 3 | 50 | 16% | 4% | 75% |
| 4 | 100 | 15% | 5% | 67% |

**Mean reduction: 72% ± 6%** - Highly consistent across runs.

---

## Future Directions

### 1. Native Toroidal Model (Rust Implementation)

Instead of retrofitting topology onto existing models, train from scratch with Tonnetz attention built-in:

```rust
struct ToroidalAttention {
    grid_size: usize,      // 12 for chromatic
    radius: f32,           // Local neighborhood
    alpha: f32,            // Decay strength
}

impl ToroidalAttention {
    fn forward(&self, q: Tensor, k: Tensor, v: Tensor) -> Tensor {
        let scores = q.matmul(&k.transpose(-2, -1));
        let topo_bias = self.tonnetz_bias(scores.size(-1));
        let masked = scores + topo_bias;
        softmax(masked, dim=-1).matmul(&v)
    }
}
```

### 2. BitNet Integration (1.58-bit Weights)

Combine toroidal topology with Microsoft's BitNet approach:
- Ternary weights: {-1, 0, +1}
- 10-20x smaller model size
- Faster inference (integer ops only)
- Native topology in attention

**Potential**: A 1B parameter BitNet model with toroidal attention could match 7B model quality with:
- 10x less memory
- 10x faster inference
- Built-in hallucination resistance

### 3. Mechanistic Understanding

Analyze attention patterns to understand WHY Mistral/Phi-2 respond differently:
- Visualize attention flows with/without topology
- Compare layer-wise activation patterns
- Identify what makes a model "topology-compatible"

---

## Technical Implementation

### Core Algorithm

```python
def toroidal_attention(query, key, value, radius=2.0, alpha=1.0, grid_size=12):
    # Standard attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) * scaling

    # Tonnetz distance on torus
    def torus_dist(i, j):
        xi, yi = i % grid_size, (i // grid_size) % grid_size
        xj, yj = j % grid_size, (j // grid_size) % grid_size
        dx = min(abs(xi - xj), grid_size - abs(xi - xj))
        dy = min(abs(yi - yj), grid_size - abs(yi - yj))
        return dx + dy

    # Build topology bias
    seq_len = query.size(-2)
    bias = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            d = torus_dist(i, j)
            bias[i, j] = 0.0 if d <= radius else -alpha * d

    # Apply topology + causal mask
    scores = scores + bias + causal_mask
    return softmax(scores).matmul(value)
```

### Integration Method

Register custom attention function in transformers:
```python
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
ALL_ATTENTION_FUNCTIONS["toroidal"] = toroidal_attention
model.config._attn_implementation = "toroidal"
```

---

## Reproducibility

All experiments conducted on:
- **Hardware**: NVIDIA A100 80GB PCIe
- **Software**: transformers 4.57.6, PyTorch with CUDA
- **Seeds**: torch.manual_seed(42), np.random.seed(42)

Code: `/papers/topological-coherence/experiments/gpu_layer_late_deep.py`
Results: `/papers/topological-coherence/experiments/results/`

---

## Conclusion

Toroidal (Tonnetz) attention topology is a **promising but model-specific** technique for reducing hallucinations. The 67-80% reduction in Mistral-7B with zero accuracy loss is significant, but the technique's limitation to specific model families (Microsoft, Mistral AI) suggests it exploits particular attention patterns created during training.

**Next step**: Build a native toroidal model in Rust with BitNet quantization to test if training with topology from scratch produces more universal benefits.
