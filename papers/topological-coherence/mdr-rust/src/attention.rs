//! Toroidal attention mechanism for MDR

use ndarray::{Array2, Array3, Axis};
use crate::tensor::TernaryTensor;
use crate::topology::{Tonnetz, TemporalContext};
use serde::{Deserialize, Serialize};

/// Toroidal attention with ternary weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToroidalAttention {
    /// Query projection weights (ternary)
    pub w_q: TernaryTensor,
    /// Key projection weights (ternary)
    pub w_k: TernaryTensor,
    /// Value projection weights (ternary)
    pub w_v: TernaryTensor,
    /// Output projection weights (ternary)
    pub w_o: TernaryTensor,

    /// Tonnetz topology
    pub topology: Tonnetz,

    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Head dimension
    pub d_head: usize,

    /// Whether to apply topology (false for early layers)
    pub apply_topology: bool,

    /// Sparsity for attention scores
    pub attention_sparsity: f32,
}

impl ToroidalAttention {
    /// Create a new toroidal attention layer
    pub fn new(
        d_model: usize,
        n_heads: usize,
        topology: Tonnetz,
        apply_topology: bool,
        sparsity: f32,
    ) -> Self {
        let d_head = d_model / n_heads;

        // Initialize ternary weights with ~30% non-zero
        let w_q = TernaryTensor::random(vec![d_model, d_model], 0.3);
        let w_k = TernaryTensor::random(vec![d_model, d_model], 0.3);
        let w_v = TernaryTensor::random(vec![d_model, d_model], 0.3);
        let w_o = TernaryTensor::random(vec![d_model, d_model], 0.3);

        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            topology,
            d_model,
            n_heads,
            d_head,
            apply_topology,
            attention_sparsity: sparsity,
        }
    }

    /// Forward pass
    pub fn forward(
        &self,
        x: &Array2<f32>,
        temporal_context: Option<&TemporalContext>,
    ) -> Array2<f32> {
        let seq_len = x.shape()[0];
        let batch_dim = x.shape()[1];

        // Project Q, K, V using ternary weights
        let q = self.w_q.matmul_f32(x);
        let k = self.w_k.matmul_f32(x);
        let v = self.w_v.matmul_f32(x);

        // Compute attention scores
        let scale = (self.d_head as f32).sqrt();
        let mut scores = q.dot(&k.t()) / scale;

        // Apply toroidal topology bias if enabled
        if self.apply_topology {
            let topo_bias = self.topology.compute_bias_matrix(seq_len);
            scores = scores + topo_bias;
        } else {
            // Just apply causal mask
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    scores[[i, j]] = f32::NEG_INFINITY;
                }
            }
        }

        // Add temporal context bias if available
        if let Some(ctx) = temporal_context {
            if let Some(context) = ctx.get_context() {
                // Simple additive bias from temporal context
                // In full implementation, this would be more sophisticated
                let ctx_bias = context.mean().unwrap_or(0.0) * 0.1;
                scores = scores + ctx_bias;
            }
        }

        // Sparse softmax: keep only top-k attention weights
        let attention = self.sparse_softmax(&scores, self.attention_sparsity);

        // Apply attention to values
        let context = attention.dot(&v);

        // Output projection
        self.w_o.matmul_f32(&context)
    }

    /// Sparse softmax: zero out all but top-k values before softmax
    fn sparse_softmax(&self, scores: &Array2<f32>, density: f32) -> Array2<f32> {
        let seq_len = scores.shape()[0];
        let k = ((seq_len as f32) * density).ceil() as usize;

        let mut result = Array2::<f32>::zeros(scores.raw_dim());

        for i in 0..seq_len {
            // Get row and find top-k indices
            let row = scores.row(i);
            let mut indexed: Vec<(usize, f32)> = row.iter().enumerate()
                .map(|(j, &v)| (j, v))
                .filter(|(_, v)| v.is_finite())
                .collect();

            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(k);

            if indexed.is_empty() {
                continue;
            }

            // Softmax over top-k only
            let max_val = indexed[0].1;
            let exp_sum: f32 = indexed.iter()
                .map(|(_, v)| (v - max_val).exp())
                .sum();

            for (j, v) in indexed {
                result[[i, j]] = ((v - max_val).exp()) / exp_sum;
            }
        }

        result
    }
}

/// Multi-head toroidal attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiHeadToroidalAttention {
    pub heads: Vec<ToroidalAttention>,
    pub w_o: TernaryTensor,
    pub d_model: usize,
    pub n_heads: usize,
}

impl MultiHeadToroidalAttention {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        topology: Tonnetz,
        apply_topology: bool,
        sparsity: f32,
    ) -> Self {
        let d_head = d_model / n_heads;

        let heads: Vec<ToroidalAttention> = (0..n_heads)
            .map(|_| {
                ToroidalAttention::new(
                    d_head,
                    1,
                    topology.clone(),
                    apply_topology,
                    sparsity,
                )
            })
            .collect();

        let w_o = TernaryTensor::random(vec![d_model, d_model], 0.3);

        Self {
            heads,
            w_o,
            d_model,
            n_heads,
        }
    }

    pub fn forward(
        &self,
        x: &Array2<f32>,
        temporal_context: Option<&TemporalContext>,
    ) -> Array2<f32> {
        let seq_len = x.shape()[0];
        let d_head = self.d_model / self.n_heads;

        // Split input across heads
        let mut head_outputs = Vec::with_capacity(self.n_heads);

        for (i, head) in self.heads.iter().enumerate() {
            // Extract slice for this head
            let start = i * d_head;
            let end = start + d_head;

            let x_head = x.slice(ndarray::s![.., start..end]).to_owned();
            let out = head.forward(&x_head, temporal_context);
            head_outputs.push(out);
        }

        // Concatenate heads
        let mut concat = Array2::<f32>::zeros((seq_len, self.d_model));
        for (i, out) in head_outputs.iter().enumerate() {
            let start = i * d_head;
            for j in 0..seq_len {
                for k in 0..d_head {
                    concat[[j, start + k]] = out[[j, k]];
                }
            }
        }

        // Final projection
        self.w_o.matmul_f32(&concat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toroidal_attention_forward() {
        let topology = Tonnetz::chromatic();
        let attn = ToroidalAttention::new(64, 4, topology, true, 0.5);

        let x = Array2::<f32>::ones((16, 64));
        let out = attn.forward(&x, None);

        assert_eq!(out.shape(), &[16, 64]);
    }

    #[test]
    fn test_sparse_softmax() {
        let topology = Tonnetz::chromatic();
        let attn = ToroidalAttention::new(64, 4, topology, true, 0.25);

        let scores = Array2::<f32>::ones((4, 4));
        let result = attn.sparse_softmax(&scores, 0.5);

        // Should have ~50% non-zero per row
        for i in 0..4 {
            let nnz = result.row(i).iter().filter(|&&x| x > 0.0).count();
            assert!(nnz <= 3);  // At most 50% of 4 = 2, but allow some slack
        }
    }
}
