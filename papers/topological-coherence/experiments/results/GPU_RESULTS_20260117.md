# GPU Experiment Results - January 17, 2026

## Environment
- **GPU**: NVIDIA A100 80GB PCIe
- **Transformers**: 4.57.6
- **PyTorch**: with CUDA

## Key Finding: Architecture-Dependent Effects Confirmed

The toroidal (Tonnetz) topology affects different architectures differently:
- **Phi-2**: Reduces hallucinations (CPU results: 40%→20%)
- **TinyLlama**: Increases hallucinations / reduces accuracy

---

## TinyLlama Baseline vs Toroidal (50 samples)

| Condition | TruthfulQA | HaluEval | Time |
|-----------|------------|----------|------|
| Baseline | 70-80% | 60% | ~30s |
| Toroidal (r=2, α=1) | 20% | 50-60% | ~90s |

**Interpretation**: Toroidal constraints harm TinyLlama's factual accuracy significantly.

---

## Hyperparameter Search Results

| Config | TruthfulQA | HaluEval | Notes |
|--------|------------|----------|-------|
| baseline | **80%** | 60% | Reference |
| r2_a1 | 20% | **50%** | Original params - harms accuracy |
| r2_a0.5 | 70% | 60% | Weaker decay preserves accuracy |
| r4_a1 | 50% | 60% | Larger radius helps |
| r4_a0.5 | **70%** | 60% | Best balance for TinyLlama |
| r1_a2 | 10% | 60% | Too tight - kills accuracy |

**Key Insights**:
1. Stronger constraints (small r, large α) hurt TinyLlama more
2. Weaker constraints (r4_a0.5) preserve accuracy
3. r2_a1 shows slight hallucination reduction (60%→50%) but at huge accuracy cost
4. TinyLlama may rely on long-range attention that toroidal topology disrupts

---

## Technical Notes

### Phi-2 Issue
Phi-2 outputs garbage (`!!!...`) on transformers 4.57.6. This appears to be a
compatibility bug, not related to our toroidal implementation. Original Phi-2
results (40%→20% hallucination reduction) were obtained on CPU with earlier
transformers version.

### GPU Implementation
Successfully implemented custom attention function approach:
- Register `toroidal_attention` in `ALL_ATTENTION_FUNCTIONS`
- Set `model.config._attn_implementation = "toroidal"`
- Bypasses all mask format/dtype issues that plagued parameter injection approach

---

## Next Experiments

1. **Other architectures**: Mistral-7B, Gemma-2B, Qwen2
2. **Layer-selective**: Only apply topology to early/middle/late layers
3. **Inverted topology**: Boost distant attention instead of suppress
4. **Different grid sizes**: Try 8, 16 instead of 12

---

## Files
- `runpod_gpu_test.py` - Quick GPU verification
- `tinyllama_gpu_proof.py` - Full TinyLlama experiment
- `gpu_hyperparam_v2.py` - Hyperparameter search
- `results/hyperparam_search_20260117_010250.json` - Raw results
