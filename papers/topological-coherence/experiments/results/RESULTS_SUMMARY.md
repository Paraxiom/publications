# Toroidal Topology Experiment Results
**Date:** January 15-16, 2026
**Paper:** "Topological Constraints for Coherent Language Models"

---

## Key Finding: Architecture-Dependent Effect

Toroidal (Tonnetz) attention constraints **help some models but hurt others**.

---

## Results Summary

### Phi-2 (2.78B params, NO RLHF) - POSITIVE RESULT

| Condition | TruthfulQA | HaluEval | Spectral CV |
|-----------|------------|----------|-------------|
| Baseline  | 0%         | 40%      | 7.39        |
| Toroidal  | 0%         | **20%**  | 7.68        |

**Effect: 50% hallucination reduction on HaluEval**

Parameters: radius=2.0, alpha=1.0, grid_size=12
Attention layers wrapped: 32

---

### TinyLlama (1.1B params, NO RLHF) - NEGATIVE RESULT

| Condition | TruthfulQA | HaluEval | Spectral CV |
|-----------|------------|----------|-------------|
| Baseline  | 0%         | 10%      | 7.55        |
| Toroidal  | 2%         | **28%**  | 7.60        |

**Effect: 180% hallucination INCREASE on HaluEval**

Parameters: radius=2.0, alpha=1.0, grid_size=12
Attention layers wrapped: 22

---

## Interpretation

1. **The toroidal constraint is not universally beneficial**
   - Helps Phi-2 (larger model, textbook data)
   - Hurts TinyLlama (smaller model, web data)

2. **Possible explanations:**
   - Model capacity: Larger models tolerate constraints better
   - Training data: Phi-2's structured data may align with geometric constraints
   - Tonnetz periodicity: 12-tone musical grid may not suit language

3. **Implications for the paper:**
   - Cannot claim universal hallucination reduction
   - Must investigate architecture-specific topologies
   - Opens research question: "What topology fits each architecture?"

---

## Files

| File | Description |
|------|-------------|
| `phi2_definitive_proof_20260115_120505.json` | Phi-2 results (n=10) |
| `phi2_definitive_proof_20260116_034745.json` | TinyLlama results (n=50) |
| `tinyllama_smoking_gun.log` | Full TinyLlama run log |
| `cross_architecture_20260115_155618.json` | Cross-architecture comparison |

---

## Reproducibility

```bash
# Phi-2 (positive result)
python phi2_definitive_proof.py --model phi-2 --mode quick

# TinyLlama (negative result)
python phi2_definitive_proof.py --model tinyllama --mode full --samples 50
```

---

## Next Steps

1. Test alternative topologies (linear distance, different grid sizes)
2. Test weaker constraints (larger radius) on smaller models
3. Investigate why Phi-2 benefits but TinyLlama doesn't
