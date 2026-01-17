//! Training infrastructure for MDR models

use ndarray::{Array1, Array2};
use crate::model::{MDRModel, MDRConfig};
use crate::tensor::TernaryTensor;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Batch size
    pub batch_size: usize,
    /// Number of training steps
    pub num_steps: usize,
    /// Steps between evaluations
    pub eval_interval: usize,
    /// Steps between checkpoints
    pub checkpoint_interval: usize,
    /// Gradient accumulation steps
    pub grad_accum_steps: usize,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f32,
    /// Weight decay
    pub weight_decay: f32,
    /// Warmup steps
    pub warmup_steps: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            batch_size: 32,
            num_steps: 10000,
            eval_interval: 100,
            checkpoint_interval: 1000,
            grad_accum_steps: 1,
            max_grad_norm: 1.0,
            weight_decay: 0.01,
            warmup_steps: 100,
        }
    }
}

/// Training example
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub input_tokens: Vec<usize>,
    pub target_tokens: Vec<usize>,
}

/// Training batch
#[derive(Debug, Clone)]
pub struct Batch {
    pub inputs: Vec<Vec<usize>>,
    pub targets: Vec<Vec<usize>>,
}

/// Simple gradient for ternary weights
/// Since weights are discrete, we accumulate "votes" for direction changes
#[derive(Debug, Clone)]
pub struct TernaryGradient {
    pub votes: Vec<f32>,  // Positive = should increase, negative = should decrease
    pub shape: Vec<usize>,
}

impl TernaryGradient {
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            votes: vec![0.0; size],
            shape,
        }
    }

    pub fn accumulate(&mut self, other: &TernaryGradient) {
        for (v, o) in self.votes.iter_mut().zip(other.votes.iter()) {
            *v += o;
        }
    }

    /// Apply gradient to ternary tensor using "straight-through estimator"
    pub fn apply(&self, tensor: &mut TernaryTensor, lr: f32, threshold: f32) {
        for (i, &vote) in self.votes.iter().enumerate() {
            let scaled_vote = vote * lr;

            // If vote is strong enough, flip the weight
            if scaled_vote > threshold {
                // Increase: -1 -> 0 -> 1
                match tensor.data[i] {
                    -1 => tensor.data[i] = 0,
                    0 => tensor.data[i] = 1,
                    _ => {}
                }
            } else if scaled_vote < -threshold {
                // Decrease: 1 -> 0 -> -1
                match tensor.data[i] {
                    1 => tensor.data[i] = 0,
                    0 => tensor.data[i] = -1,
                    _ => {}
                }
            }
        }
    }
}

/// Trainer for MDR models
pub struct Trainer {
    pub model: MDRModel,
    pub config: TrainingConfig,
    pub step: usize,
    pub loss_history: VecDeque<f32>,
}

impl Trainer {
    pub fn new(model: MDRModel, config: TrainingConfig) -> Self {
        Self {
            model,
            config,
            step: 0,
            loss_history: VecDeque::with_capacity(100),
        }
    }

    /// Compute cross-entropy loss
    pub fn compute_loss(&mut self, batch: &Batch) -> f32 {
        let mut total_loss = 0.0;
        let mut total_tokens = 0;

        for (inputs, targets) in batch.inputs.iter().zip(batch.targets.iter()) {
            let logits = self.model.forward(inputs);

            // Cross-entropy loss
            for (t, &target) in targets.iter().enumerate() {
                if target < self.model.config.vocab_size {
                    let row = logits.row(t);

                    // Log-softmax
                    let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let log_sum_exp: f32 = row.iter()
                        .map(|&x| (x - max_val).exp())
                        .sum::<f32>()
                        .ln() + max_val;

                    let log_prob = row[target] - log_sum_exp;
                    total_loss -= log_prob;
                    total_tokens += 1;
                }
            }
        }

        if total_tokens > 0 {
            total_loss / total_tokens as f32
        } else {
            0.0
        }
    }

    /// Training step with straight-through gradient estimation
    pub fn train_step(&mut self, batch: &Batch) -> f32 {
        // Forward pass and compute loss
        let loss = self.compute_loss(batch);

        // For ternary networks, we use a simplified update rule:
        // - Compute loss gradient w.r.t. outputs
        // - Propagate "votes" back through ternary weights
        // - Update weights based on accumulated votes

        // This is a simplified version - full implementation would need
        // proper backprop with straight-through estimator

        // Update learning rate with warmup
        let lr = if self.step < self.config.warmup_steps {
            self.config.learning_rate * (self.step as f32 / self.config.warmup_steps as f32)
        } else {
            self.config.learning_rate
        };

        // Simplified weight update: random perturbation with loss-guided acceptance
        // (This is a placeholder - real implementation needs proper gradients)
        self.perturb_weights_if_better(batch, lr);

        self.step += 1;

        // Track loss
        if self.loss_history.len() >= 100 {
            self.loss_history.pop_front();
        }
        self.loss_history.push_back(loss);

        loss
    }

    /// Simple perturbation-based update (placeholder for proper gradients)
    fn perturb_weights_if_better(&mut self, batch: &Batch, lr: f32) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Randomly select a layer and weight to potentially flip
        let layer_idx = rng.gen_range(0..self.model.layers.len());
        let layer = &mut self.model.layers[layer_idx];

        // Choose which weight matrix to perturb
        let matrix_choice: u8 = rng.gen_range(0..4);

        let weights = match matrix_choice {
            0 => &mut layer.attention.w_q,
            1 => &mut layer.attention.w_k,
            2 => &mut layer.attention.w_v,
            _ => &mut layer.ffn.w1,
        };

        if weights.data.is_empty() {
            return;
        }

        // Select random position
        let idx = rng.gen_range(0..weights.data.len());
        let old_val = weights.data[idx];

        // Try flipping
        let new_val = match old_val {
            -1 => if rng.gen_bool(0.5) { 0 } else { -1 },
            0 => if rng.gen_bool(0.5) { 1 } else { -1 },
            1 => if rng.gen_bool(0.5) { 0 } else { 1 },
            _ => old_val,
        };

        weights.data[idx] = new_val;

        // Check if loss improved (with some randomness to escape local minima)
        let new_loss = self.compute_loss(batch);
        let accept_prob = if new_loss < self.loss_history.back().copied().unwrap_or(f32::INFINITY) {
            1.0
        } else {
            // Simulated annealing-like acceptance
            (-(new_loss - self.loss_history.back().copied().unwrap_or(0.0)) / lr).exp().min(0.1)
        };

        if !rng.gen_bool(accept_prob as f64) {
            // Revert
            weights.data[idx] = old_val;
        }
    }

    /// Get average recent loss
    pub fn avg_loss(&self) -> f32 {
        if self.loss_history.is_empty() {
            0.0
        } else {
            self.loss_history.iter().sum::<f32>() / self.loss_history.len() as f32
        }
    }

    /// Save checkpoint
    pub fn save_checkpoint(&self, path: &str) -> std::io::Result<()> {
        let checkpoint = Checkpoint {
            model: self.model.clone(),
            step: self.step,
            config: self.config.clone(),
        };

        let encoded = bincode::serialize(&checkpoint).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e)
        })?;
        std::fs::write(path, encoded)
    }

    /// Load checkpoint
    pub fn load_checkpoint(path: &str) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        let checkpoint: Checkpoint = bincode::deserialize(&data).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e)
        })?;

        Ok(Self {
            model: checkpoint.model,
            config: checkpoint.config,
            step: checkpoint.step,
            loss_history: VecDeque::new(),
        })
    }
}

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    model: MDRModel,
    step: usize,
    config: TrainingConfig,
}

/// Simple data loader
pub struct DataLoader {
    pub examples: Vec<TrainingExample>,
    pub batch_size: usize,
    current_idx: usize,
}

impl DataLoader {
    pub fn new(examples: Vec<TrainingExample>, batch_size: usize) -> Self {
        Self {
            examples,
            batch_size,
            current_idx: 0,
        }
    }

    pub fn next_batch(&mut self) -> Option<Batch> {
        if self.current_idx >= self.examples.len() {
            self.current_idx = 0;
            return None;  // Epoch complete
        }

        let end = (self.current_idx + self.batch_size).min(self.examples.len());
        let batch_examples = &self.examples[self.current_idx..end];

        let inputs: Vec<Vec<usize>> = batch_examples.iter()
            .map(|e| e.input_tokens.clone())
            .collect();
        let targets: Vec<Vec<usize>> = batch_examples.iter()
            .map(|e| e.target_tokens.clone())
            .collect();

        self.current_idx = end;

        Some(Batch { inputs, targets })
    }

    pub fn shuffle(&mut self) {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        self.examples.shuffle(&mut rng);
        self.current_idx = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_creation() {
        let model_config = MDRConfig::small();
        let model = MDRModel::new(model_config);
        let train_config = TrainingConfig::default();

        let trainer = Trainer::new(model, train_config);
        assert_eq!(trainer.step, 0);
    }

    #[test]
    fn test_loss_computation() {
        let model_config = MDRConfig::small();
        let model = MDRModel::new(model_config);
        let train_config = TrainingConfig::default();

        let mut trainer = Trainer::new(model, train_config);

        let batch = Batch {
            inputs: vec![vec![1, 2, 3, 4]],
            targets: vec![vec![2, 3, 4, 5]],
        };

        let loss = trainer.compute_loss(&batch);
        assert!(loss > 0.0);  // Should be positive
        assert!(loss < 20.0);  // Should be reasonable (log vocab_size ~= 10)
    }
}
