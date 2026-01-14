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
│   ├── tonnetz_validation.py                # Minimal validation (CPU)
│   └── venv/                                # Python environment
└── README.md
```

## Running the Experiments

### Minimal Validation (CPU)
```bash
cd experiments
python3 -m venv venv
source venv/bin/activate
pip install torch numpy
python tonnetz_validation.py
```
Runs in ~3 minutes on CPU.

### Full Validation (GPU)
See `/topological-coherence/experiments/` for:
- `train_phi2.py` - Full training script
- `topological_attention.py` - Mask implementations
- `EXPERIMENT_REPORT_v2.md` - Detailed results

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
