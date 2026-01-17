//! Complete MDR model

use ndarray::{Array1, Array2};
use crate::tensor::TernaryTensor;
use crate::topology::{Tonnetz, TemporalContext};
use crate::layer::{MDRLayer, layer_norm};
use serde::{Deserialize, Serialize};

/// MDR Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MDRConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Feed-forward dimension
    pub d_ff: usize,
    /// Number of layers
    pub n_layers: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Tonnetz grid size (12 for chromatic)
    pub grid_size: usize,
    /// Topology radius
    pub radius: f32,
    /// Topology decay
    pub alpha: f32,
    /// Weight sparsity (fraction non-zero)
    pub weight_sparsity: f32,
    /// Attention sparsity (fraction of positions attended)
    pub attention_sparsity: f32,
    /// Temporal context window
    pub temporal_window: usize,
    /// Temporal decay factor
    pub temporal_decay: f32,
}

impl Default for MDRConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 512,
            n_heads: 8,
            d_ff: 2048,
            n_layers: 12,
            max_seq_len: 2048,
            grid_size: 12,
            radius: 2.0,
            alpha: 1.0,
            weight_sparsity: 0.3,
            attention_sparsity: 0.5,
            temporal_window: 8,
            temporal_decay: 0.9,
        }
    }
}

impl MDRConfig {
    /// Small model for testing
    pub fn small() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 256,
            n_heads: 4,
            d_ff: 1024,
            n_layers: 6,
            max_seq_len: 512,
            ..Default::default()
        }
    }

    /// Medium model
    pub fn medium() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 512,
            n_heads: 8,
            d_ff: 2048,
            n_layers: 12,
            max_seq_len: 2048,
            ..Default::default()
        }
    }

    /// Large model (1B+ params)
    pub fn large() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 2048,
            n_heads: 16,
            d_ff: 8192,
            n_layers: 24,
            max_seq_len: 4096,
            ..Default::default()
        }
    }

    /// Estimate parameter count
    pub fn param_count(&self) -> usize {
        // Embedding
        let embed = self.vocab_size * self.d_model;

        // Per layer
        let qkv = 3 * self.d_model * self.d_model;
        let attn_out = self.d_model * self.d_model;
        let ffn = 2 * self.d_model * self.d_ff;
        let per_layer = qkv + attn_out + ffn;

        // Output
        let output = self.d_model * self.vocab_size;

        embed + (per_layer * self.n_layers) + output
    }

    /// Estimate model size in bytes (ternary = 2 bits per param, but stored as i8)
    pub fn size_bytes(&self) -> usize {
        self.param_count()  // 1 byte per ternary param
    }
}

/// Complete MDR model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MDRModel {
    pub config: MDRConfig,

    /// Token embeddings
    pub embeddings: Array2<f32>,

    /// Transformer layers
    pub layers: Vec<MDRLayer>,

    /// Output projection (tied with embeddings for efficiency)
    pub output_proj: TernaryTensor,

    /// Temporal context
    #[serde(skip)]
    pub temporal_context: TemporalContext,
}

impl MDRModel {
    /// Create a new randomly initialized MDR model
    pub fn new(config: MDRConfig) -> Self {
        let topology = Tonnetz::new(config.grid_size, config.radius, config.alpha);

        // Initialize embeddings (float for now, could be ternary)
        let mut embeddings = Array2::<f32>::zeros((config.vocab_size, config.d_model));
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for elem in embeddings.iter_mut() {
            *elem = rng.gen_range(-0.1..0.1);
        }

        // Create layers
        let layers: Vec<MDRLayer> = (0..config.n_layers)
            .map(|i| {
                MDRLayer::new(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    i,
                    config.n_layers,
                    topology.clone(),
                    config.weight_sparsity,
                    config.attention_sparsity,
                )
            })
            .collect();

        // Output projection
        let output_proj = TernaryTensor::random(
            vec![config.d_model, config.vocab_size],
            config.weight_sparsity,
        );

        let temporal_context = TemporalContext::new(
            config.temporal_window,
            config.temporal_decay,
        );

        Self {
            config,
            embeddings,
            layers,
            output_proj,
            temporal_context,
        }
    }

    /// Forward pass: tokens -> logits
    pub fn forward(&mut self, tokens: &[usize]) -> Array2<f32> {
        let seq_len = tokens.len();

        // Embed tokens
        let mut hidden = Array2::<f32>::zeros((seq_len, self.config.d_model));
        for (i, &token) in tokens.iter().enumerate() {
            if token < self.config.vocab_size {
                for j in 0..self.config.d_model {
                    hidden[[i, j]] = self.embeddings[[token, j]];
                }
            }
        }

        // Add positional encoding (sinusoidal)
        for i in 0..seq_len {
            for j in 0..self.config.d_model {
                let angle = (i as f32) / 10000_f32.powf((2 * (j / 2)) as f32 / self.config.d_model as f32);
                if j % 2 == 0 {
                    hidden[[i, j]] += angle.sin();
                } else {
                    hidden[[i, j]] += angle.cos();
                }
            }
        }

        // Pass through layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden, Some(&self.temporal_context));
        }

        // Update temporal context with final hidden state
        self.temporal_context.push(hidden.clone());

        // Final layer norm
        let normed = layer_norm(&hidden, 1e-5);

        // Project to vocab
        self.output_proj.matmul_f32(&normed)
    }

    /// Generate text autoregressively
    pub fn generate(
        &mut self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        temperature: f32,
    ) -> Vec<usize> {
        let mut tokens = prompt_tokens.to_vec();

        for _ in 0..max_new_tokens {
            // Get logits for last position
            let logits = self.forward(&tokens);
            let last_logits = logits.row(logits.shape()[0] - 1);

            // Sample next token
            let next_token = self.sample(&last_logits.to_owned(), temperature);
            tokens.push(next_token);

            // Stop at EOS (assuming token 2 is EOS)
            if next_token == 2 {
                break;
            }
        }

        tokens
    }

    /// Sample from logits with temperature
    fn sample(&self, logits: &Array1<f32>, temperature: f32) -> usize {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Apply temperature
        let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

        // Softmax
        let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|&x| x / sum).collect();

        // Sample
        let r: f32 = rng.gen();
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }

        probs.len() - 1
    }

    /// Reset temporal context (for new sequence)
    pub fn reset_context(&mut self) {
        self.temporal_context.reset();
    }

    /// Get model statistics
    pub fn stats(&self) -> ModelStats {
        let layer_stats: Vec<_> = self.layers.iter().map(|l| l.stats()).collect();

        let total_params = self.config.param_count();
        let topology_layers = layer_stats.iter().filter(|s| s.apply_topology).count();

        ModelStats {
            config: self.config.clone(),
            total_params,
            size_mb: self.config.size_bytes() as f32 / 1_000_000.0,
            topology_layers,
            total_layers: self.config.n_layers,
            layer_stats,
        }
    }

    /// Save model to file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let encoded = bincode::serialize(self).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e)
        })?;
        std::fs::write(path, encoded)
    }

    /// Load model from file
    pub fn load(path: &str) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        bincode::deserialize(&data).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e)
        })
    }
}

#[derive(Debug, Clone)]
pub struct ModelStats {
    pub config: MDRConfig,
    pub total_params: usize,
    pub size_mb: f32,
    pub topology_layers: usize,
    pub total_layers: usize,
    pub layer_stats: Vec<crate::layer::LayerStats>,
}

impl std::fmt::Display for ModelStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MDR Model Statistics")?;
        writeln!(f, "====================")?;
        writeln!(f, "Parameters: {}", self.total_params)?;
        writeln!(f, "Size: {:.2} MB", self.size_mb)?;
        writeln!(f, "Layers: {} ({} with topology)", self.total_layers, self.topology_layers)?;
        writeln!(f, "d_model: {}", self.config.d_model)?;
        writeln!(f, "n_heads: {}", self.config.n_heads)?;
        writeln!(f, "Topology: grid={}, r={}, α={}",
            self.config.grid_size, self.config.radius, self.config.alpha)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_sizes() {
        let small = MDRConfig::small();
        let medium = MDRConfig::medium();
        let large = MDRConfig::large();

        println!("Small: {} params ({:.2} MB)",
            small.param_count(), small.size_bytes() as f32 / 1e6);
        println!("Medium: {} params ({:.2} MB)",
            medium.param_count(), medium.size_bytes() as f32 / 1e6);
        println!("Large: {} params ({:.2} MB)",
            large.param_count(), large.size_bytes() as f32 / 1e6);

        assert!(small.param_count() < medium.param_count());
        assert!(medium.param_count() < large.param_count());
    }

    #[test]
    fn test_model_forward() {
        let config = MDRConfig::small();
        let mut model = MDRModel::new(config);

        let tokens = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens);

        assert_eq!(logits.shape()[0], 5);  // seq_len
        assert_eq!(logits.shape()[1], 32000);  // vocab_size
    }

    #[test]
    fn test_model_generate() {
        let config = MDRConfig::small();
        let mut model = MDRModel::new(config);

        let prompt = vec![1, 2, 3];
        let output = model.generate(&prompt, 5, 1.0);

        assert!(output.len() >= prompt.len());
        assert!(output.len() <= prompt.len() + 5);
    }

    #[test]
    fn test_model_stats() {
        let config = MDRConfig::small();
        let model = MDRModel::new(config);
        let stats = model.stats();

        println!("{}", stats);
        assert_eq!(stats.total_layers, 6);
        assert_eq!(stats.topology_layers, 2);  // Last 1/3 of 6 = 2
    }
}
