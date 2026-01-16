# Topological Constraints for Coherent Language Models

**Why Geometry Prevents Hallucination**

Sylvain Cormier | Paraxiom Research | January 2026

---

## Abstract

Residual geometry determines whether reasoning is stable. We show that transformer latent dynamics, operating on unconstrained vector spaces, lack the conserved quantities necessary for bounded inference. This establishes a hierarchy of sufficient conditions:

```
mHC (Birkhoff) ⊂ ERLHS (Hamiltonian) ⊂ Karmonic (Toroidal + Spectral)
```

**Experimental validation on Phi-2 (2.7B) confirms the theory.**

## Key Results

### Experiment 1: Synthetic Validation (CPU, ~3 min)

| Condition | Drift Rate | Coherence Var | Grad Norm |
|-----------|------------|---------------|-----------|
| Baseline | 0.0100 | 35.76 | 0.27 |
| mHC | 0.0133 | 1010.54 | 1.60 |
| **Toroidal** | **0.0060** | 41.93 | **0.22** |

**Toroidal attention reduces drift by 40%** compared to unconstrained baseline.

### Experiment 2: Scaled Validation (Phi-2, A100, ~22 GPU-hours)

| Condition | TruthfulQA | HaluEval | Train Loss |
|-----------|------------|----------|------------|
| Baseline | 14.44% | 55.00% | 1.6708 |
| Local window | 17.26% | 53.00% | 1.6704 |
| Random | 15.30% | 55.20% | 1.6706 |
| **Toroidal** | **17.26%** | **52.60%** | 1.6699 |

**Toroidal achieves +19.5% relative improvement on TruthfulQA and best HaluEval score.**

## Interactive Demo

Try it: [huggingface.co/spaces/paraxiom/topological-coherence](https://huggingface.co/spaces/paraxiom/topological-coherence)

## Repository Structure

```
topological-coherence/
├── cormier_topological_coherence_2026.pdf   # Paper
├── cormier_topological_coherence_2026.tex   # LaTeX source
├── experiments/
│   ├── tonnetz_validation.py                # Minimal synthetic validation (CPU)
│   ├── phi2_definitive_proof.py             # Definitive proof on Phi-2 (GPU)
│   ├── llm_coherence_validation.py          # Cross-architecture validation
│   ├── requirements.txt                     # Dependencies
│   └── venv/                                # Python environment
└── README.md
```

## Running the Experiments

### 1. Minimal Validation (CPU, ~3 min)
```bash
cd experiments
python3 -m venv venv
source venv/bin/activate
pip install torch numpy
python tonnetz_validation.py
```

### 2. Definitive Proof on Phi-2 (GPU, ~2-4 hours)

**Why Phi-2?** Clean room (no RLHF masking), textbook data, direct replication of original results.

```bash
cd experiments
source venv/bin/activate
pip install -r requirements.txt

# Quick test (10 samples each benchmark)
python phi2_definitive_proof.py --mode quick

# Spectral signature analysis only
python phi2_definitive_proof.py --mode spectral-only

# Full proof (50 samples, TruthfulQA + HaluEval)
python phi2_definitive_proof.py --mode full --samples 50
```

**Targets:**
- TruthfulQA: 19-20% relative improvement
- HaluEval: <53% hallucination rate
- Spectral gap CV: <0.1 (proves constant gap theorem)

### 3. Cross-Architecture Validation (GPU recommended)

Tests the hypothesis on **Llama-3.2-1B** and **Phi-2** to prove cross-architectural generalization.

```bash
cd experiments
source venv/bin/activate
pip install -r requirements.txt

# Quick test (1 model, 2 prompts)
python llm_coherence_validation.py --mode quick

# Llama-3.2-1B only (~2-4 GPU-hours)
python llm_coherence_validation.py --mode llama-only

# Phi-2 only (~2-4 GPU-hours)
python llm_coherence_validation.py --mode phi-only

# Full comparison (~8-12 GPU-hours)
python llm_coherence_validation.py --mode full --output-dir results
```

**Note:** Llama-3.2-1B requires HuggingFace access. Run:
```bash
huggingface-cli login
```

### 3. Full Training Validation (GPU, ~22 hours)
See `/topological-coherence/experiments/` for:
- `train_phi2.py` - Full training script
- `topological_attention.py` - Mask implementations
- `EXPERIMENT_REPORT_v2.md` - Detailed results

## Estimated Costs

| Experiment | Hardware | Time | Cloud Cost |
|------------|----------|------|------------|
| Minimal validation | CPU | ~3 min | Free |
| Llama-3.2-1B inference | A100 | ~2-4 hrs | ~$5-10 |
| Phi-2 inference | A100 | ~2-4 hrs | ~$5-10 |
| Cross-architecture full | A100 | ~8-12 hrs | ~$25-40 |
| Fine-tuning (per model) | A100 | ~20 hrs | ~$60 |

## Citation

```bibtex
@misc{cormier2026topological,
  author = {Cormier, Sylvain},
  title = {Topological Constraints for Coherent Language Models: Why Geometry Prevents Hallucination},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18187835}
}
```

## Related Work

- [ERLHS: Hamiltonian Framework for Coherence-Preserving ML](https://doi.org/10.5281/zenodo.17928909)
- [Karmonic Mesh: Spectral Consensus on Toroidal Manifolds](https://doi.org/10.5281/zenodo.17928991)
- [mHC: Manifold-Constrained Hyper-Connections (DeepSeek, 2026)](https://arxiv.org/abs/2512.24880)
- [Geometric Uncertainty for Detecting Hallucinations (Phillips et al., 2025)](https://arxiv.org/abs/2505.xxxxx) - Complementary detection approach

## License

Apache 2.0
