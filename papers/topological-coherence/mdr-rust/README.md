# MDR - Meta Distributed Representations

A Rust implementation of sparse, topological, temporal neural representations.

## Overview

MDR combines four key concepts:

1. **Sparsity** (from SDR/HTM) - Only ~2% of units active
2. **Ternary Weights** (from BitNet) - Weights are {-1, 0, +1}
3. **Toroidal Topology** (from Tonnetz) - Attention constrained by musical pitch space
4. **Temporal Context** (from HTM) - Time-decaying memory of past states

## Empirical Basis

This implementation is based on experiments showing that toroidal topology reduces hallucinations by 67-80% in Mistral-7B-Instruct with zero accuracy loss.

## Building

```bash
cargo build --release
```

## Training

```bash
# Train a small model on synthetic data
cargo run --release --bin mdr-train -- --size small --steps 1000

# Train on custom data
cargo run --release --bin mdr-train -- --size medium --data training_data.txt --steps 10000

# Resume from checkpoint
cargo run --release --bin mdr-train -- --resume checkpoints/checkpoint_5000.bin
```

## Inference

```bash
# Interactive mode
cargo run --release --bin mdr-infer -- --model checkpoints/model_final.bin --interactive

# Single prompt
cargo run --release --bin mdr-infer -- --model checkpoints/model_final.bin --prompt "Hello, world"
```

## Model Sizes

| Size | Parameters | Memory | Topology Layers |
|------|------------|--------|-----------------|
| Small | ~20M | ~20 MB | 2/6 |
| Medium | ~100M | ~100 MB | 4/12 |
| Large | ~1B | ~1 GB | 8/24 |

## Key Features

### Ternary Weights

All weights are {-1, 0, +1}, enabling:
- Integer-only computation
- 10-20x memory reduction
- Efficient inference on CPU

### Toroidal Attention

Final 1/3 of layers use Tonnetz topology:
- Constrains attention to local neighborhoods on 12-tone torus
- Prevents "long-range hallucination jumps"
- Preserves factual knowledge in early layers

### Temporal Context

Hidden states are accumulated with exponential decay:
- Provides temporal coherence
- Enables sequence prediction
- Prevents inconsistency with prior context

## Architecture

```
Input Tokens
    ↓
Embedding (float)
    ↓
[Layer 1-N/3] - Standard attention
    ↓
[Layer N/3-2N/3] - Standard attention
    ↓
[Layer 2N/3-N] - Toroidal attention (topology applied)
    ↓
Output Projection (ternary)
    ↓
Logits
```

## Theory

See `../theory/MDR_THEORY.md` for the full theoretical framework.

## License

MIT

## Citation

```bibtex
@article{mdr2026,
  title={Meta Distributed Representations: Sparse, Topological, Temporal Neural Networks},
  author={Cormier, Sylvain},
  year={2026}
}
```
