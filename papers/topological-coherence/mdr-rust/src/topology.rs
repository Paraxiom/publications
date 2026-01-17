//! Toroidal topology based on the Tonnetz (musical pitch space)

use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Tonnetz - a toroidal representation of pitch relationships
///
/// The Tonnetz is a 2D torus where:
/// - Horizontal axis: Perfect fifths (7 semitones)
/// - Vertical axis: Major thirds (4 semitones)
/// - Wraps around in both dimensions (12-tone equal temperament)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tonnetz {
    /// Grid size (12 for chromatic scale)
    pub grid_size: usize,
    /// Radius for local neighborhood
    pub radius: f32,
    /// Decay strength for distant positions
    pub alpha: f32,
    /// Cached bias matrices for different sequence lengths
    #[serde(skip)]
    cache: std::collections::HashMap<usize, Array2<f32>>,
}

impl Tonnetz {
    /// Create a new Tonnetz with given parameters
    pub fn new(grid_size: usize, radius: f32, alpha: f32) -> Self {
        Self {
            grid_size,
            radius,
            alpha,
            cache: std::collections::HashMap::new(),
        }
    }

    /// Default Tonnetz for chromatic scale
    pub fn chromatic() -> Self {
        Self::new(12, 2.0, 1.0)
    }

    /// Map linear position to 2D torus coordinates
    #[inline]
    pub fn to_torus(&self, pos: usize) -> (usize, usize) {
        let x = pos % self.grid_size;
        let y = (pos / self.grid_size) % self.grid_size;
        (x, y)
    }

    /// Toroidal Manhattan distance between two positions
    #[inline]
    pub fn distance(&self, i: usize, j: usize) -> f32 {
        let (xi, yi) = self.to_torus(i);
        let (xj, yj) = self.to_torus(j);

        let dx = {
            let d = (xi as i32 - xj as i32).abs() as usize;
            d.min(self.grid_size - d)
        };

        let dy = {
            let d = (yi as i32 - yj as i32).abs() as usize;
            d.min(self.grid_size - d)
        };

        (dx + dy) as f32
    }

    /// Compute attention bias for position pair
    #[inline]
    pub fn bias(&self, i: usize, j: usize) -> f32 {
        let d = self.distance(i, j);
        if d <= self.radius {
            0.0
        } else {
            -self.alpha * d
        }
    }

    /// Get or compute bias matrix for given sequence length
    pub fn get_bias_matrix(&mut self, seq_len: usize) -> Array2<f32> {
        if let Some(cached) = self.cache.get(&seq_len) {
            return cached.clone();
        }

        let mut bias = Array2::<f32>::zeros((seq_len, seq_len));

        for i in 0..seq_len {
            for j in 0..seq_len {
                bias[[i, j]] = self.bias(i, j);
            }
        }

        // Add causal mask
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                bias[[i, j]] = f32::NEG_INFINITY;
            }
        }

        self.cache.insert(seq_len, bias.clone());
        bias
    }

    /// Compute bias matrix without caching (for parallel use)
    pub fn compute_bias_matrix(&self, seq_len: usize) -> Array2<f32> {
        let mut bias = Array2::<f32>::zeros((seq_len, seq_len));

        for i in 0..seq_len {
            for j in 0..seq_len {
                bias[[i, j]] = self.bias(i, j);
            }
        }

        // Add causal mask
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                bias[[i, j]] = f32::NEG_INFINITY;
            }
        }

        bias
    }
}

/// Inverted Tonnetz - boosts distant attention instead of suppressing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvertedTonnetz {
    pub inner: Tonnetz,
}

impl InvertedTonnetz {
    pub fn new(grid_size: usize, radius: f32, alpha: f32) -> Self {
        Self {
            inner: Tonnetz::new(grid_size, radius, alpha),
        }
    }

    /// Inverted bias: suppress nearby, allow distant
    pub fn bias(&self, i: usize, j: usize) -> f32 {
        let d = self.inner.distance(i, j);
        if d <= self.inner.radius {
            -self.inner.alpha * (self.inner.radius - d + 1.0)
        } else {
            0.0
        }
    }

    pub fn compute_bias_matrix(&self, seq_len: usize) -> Array2<f32> {
        let mut bias = Array2::<f32>::zeros((seq_len, seq_len));

        for i in 0..seq_len {
            for j in 0..seq_len {
                bias[[i, j]] = self.bias(i, j);
            }
        }

        // Add causal mask
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                bias[[i, j]] = f32::NEG_INFINITY;
            }
        }

        bias
    }
}

/// Temporal context buffer for MDR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalContext {
    /// History of hidden states
    pub history: Vec<Array2<f32>>,
    /// Maximum history length
    pub max_len: usize,
    /// Decay factor per timestep
    pub decay: f32,
}

impl TemporalContext {
    pub fn new(max_len: usize, decay: f32) -> Self {
        Self {
            history: Vec::with_capacity(max_len),
            max_len,
            decay,
        }
    }

    /// Add a new state to history
    pub fn push(&mut self, state: Array2<f32>) {
        if self.history.len() >= self.max_len {
            self.history.remove(0);
        }
        self.history.push(state);
    }

    /// Get weighted sum of historical states
    pub fn get_context(&self) -> Option<Array2<f32>> {
        if self.history.is_empty() {
            return None;
        }

        let shape = self.history[0].shape();
        let mut context = Array2::<f32>::zeros((shape[0], shape[1]));

        for (i, state) in self.history.iter().rev().enumerate() {
            let weight = self.decay.powi(i as i32);
            context = context + &(state * weight);
        }

        Some(context)
    }

    /// Clear history
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tonnetz_distance() {
        let t = Tonnetz::chromatic();

        // Same position
        assert_eq!(t.distance(0, 0), 0.0);

        // Adjacent positions
        assert_eq!(t.distance(0, 1), 1.0);

        // Wrapping distance
        assert_eq!(t.distance(0, 11), 1.0);  // Wraps around

        // Diagonal
        assert_eq!(t.distance(0, 13), 2.0);  // (0,0) to (1,1)
    }

    #[test]
    fn test_tonnetz_bias() {
        let t = Tonnetz::new(12, 2.0, 1.0);

        // Within radius
        assert_eq!(t.bias(0, 1), 0.0);
        assert_eq!(t.bias(0, 2), 0.0);

        // Outside radius
        assert!(t.bias(0, 3) < 0.0);
        assert!(t.bias(0, 6) < t.bias(0, 3));  // Further = more negative
    }

    #[test]
    fn test_temporal_context() {
        let mut ctx = TemporalContext::new(3, 0.9);

        ctx.push(Array2::ones((2, 4)));
        ctx.push(Array2::ones((2, 4)) * 2.0);

        let context = ctx.get_context().unwrap();

        // Most recent has weight 1.0, previous has weight 0.9
        // Expected: 2.0 * 1.0 + 1.0 * 0.9 = 2.9
        assert!((context[[0, 0]] - 2.9).abs() < 0.01);
    }
}
