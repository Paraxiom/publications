//! MDR transformer layer

use ndarray::Array2;
use crate::tensor::TernaryTensor;
use crate::attention::ToroidalAttention;
use crate::topology::{Tonnetz, TemporalContext};
use serde::{Deserialize, Serialize};

/// Layer normalization (simplified, no learnable params for now)
pub fn layer_norm(x: &Array2<f32>, eps: f32) -> Array2<f32> {
    let mean = x.mean().unwrap_or(0.0);
    let var = x.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(1.0);
    let std = (var + eps).sqrt();
    x.mapv(|v| (v - mean) / std)
}

/// GELU activation (approximation)
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// Feed-forward network with ternary weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryFFN {
    pub w1: TernaryTensor,
    pub w2: TernaryTensor,
    pub d_model: usize,
    pub d_ff: usize,
}

impl TernaryFFN {
    pub fn new(d_model: usize, d_ff: usize, sparsity: f32) -> Self {
        Self {
            w1: TernaryTensor::random(vec![d_model, d_ff], sparsity),
            w2: TernaryTensor::random(vec![d_ff, d_model], sparsity),
            d_model,
            d_ff,
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // Up projection
        let h = self.w1.matmul_f32(x);

        // GELU activation
        let h_gelu = h.mapv(gelu);

        // Down projection
        self.w2.matmul_f32(&h_gelu)
    }
}

/// Complete MDR transformer layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MDRLayer {
    pub attention: ToroidalAttention,
    pub ffn: TernaryFFN,
    pub layer_idx: usize,
    pub d_model: usize,
    pub apply_topology: bool,
}

impl MDRLayer {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        layer_idx: usize,
        total_layers: usize,
        topology: Tonnetz,
        weight_sparsity: f32,
        attention_sparsity: f32,
    ) -> Self {
        // Apply topology only to last 1/3 of layers
        let apply_topology = layer_idx >= (2 * total_layers / 3);

        let attention = ToroidalAttention::new(
            d_model,
            n_heads,
            topology,
            apply_topology,
            attention_sparsity,
        );

        let ffn = TernaryFFN::new(d_model, d_ff, weight_sparsity);

        Self {
            attention,
            ffn,
            layer_idx,
            d_model,
            apply_topology,
        }
    }

    pub fn forward(
        &self,
        x: &Array2<f32>,
        temporal_context: Option<&TemporalContext>,
    ) -> Array2<f32> {
        // Pre-norm architecture
        let normed = layer_norm(x, 1e-5);

        // Attention with residual
        let attn_out = self.attention.forward(&normed, temporal_context);
        let x_attn = x + &attn_out;

        // FFN with residual
        let normed_ffn = layer_norm(&x_attn, 1e-5);
        let ffn_out = self.ffn.forward(&normed_ffn);
        let x_ffn = x_attn + ffn_out;

        x_ffn
    }

    /// Get layer statistics
    pub fn stats(&self) -> LayerStats {
        LayerStats {
            layer_idx: self.layer_idx,
            apply_topology: self.apply_topology,
            w_q_sparsity: self.attention.w_q.sparsity(),
            w_k_sparsity: self.attention.w_k.sparsity(),
            w_v_sparsity: self.attention.w_v.sparsity(),
            ffn_w1_sparsity: self.ffn.w1.sparsity(),
            ffn_w2_sparsity: self.ffn.w2.sparsity(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayerStats {
    pub layer_idx: usize,
    pub apply_topology: bool,
    pub w_q_sparsity: f32,
    pub w_k_sparsity: f32,
    pub w_v_sparsity: f32,
    pub ffn_w1_sparsity: f32,
    pub ffn_w2_sparsity: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let normed = layer_norm(&x, 1e-5);

        // Should have mean ~0 and std ~1
        let mean = normed.mean().unwrap();
        assert!(mean.abs() < 0.01);
    }

    #[test]
    fn test_gelu() {
        assert!((gelu(0.0) - 0.0).abs() < 0.01);
        assert!(gelu(1.0) > 0.8);
        assert!(gelu(-1.0) < -0.1);
    }

    #[test]
    fn test_mdr_layer_forward() {
        let topology = Tonnetz::chromatic();
        let layer = MDRLayer::new(
            64,     // d_model
            4,      // n_heads
            256,    // d_ff
            5,      // layer_idx (last third of 6 layers)
            6,      // total_layers
            topology,
            0.3,    // weight sparsity
            0.5,    // attention sparsity
        );

        let x = Array2::<f32>::ones((16, 64));
        let out = layer.forward(&x, None);

        assert_eq!(out.shape(), &[16, 64]);
        assert!(layer.apply_topology);  // Should be true for layer 5 of 6
    }

    #[test]
    fn test_topology_application() {
        let topology = Tonnetz::chromatic();

        // Early layer - no topology
        let early = MDRLayer::new(64, 4, 256, 1, 12, topology.clone(), 0.3, 0.5);
        assert!(!early.apply_topology);

        // Late layer - with topology
        let late = MDRLayer::new(64, 4, 256, 10, 12, topology, 0.3, 0.5);
        assert!(late.apply_topology);
    }
}
