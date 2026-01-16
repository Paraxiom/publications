//! Sinkhorn-Knopp Validator Weight Normalization
//!
//! Prototype implementation of conservation-constrained validator aggregation.
//! This enforces doubly-stochastic properties on validator influence matrices.
//!
//! Reference: "Conservation-Constrained Aggregation in Distributed Trust Networks"
//! Paraxiom Research Technical Note, January 2026

use sp_std::vec::Vec;

/// Number of Sinkhorn-Knopp iterations (20 is sufficient for practical accuracy)
const SINKHORN_ITERATIONS: usize = 20;

/// Represents a validator's raw influence metrics
#[derive(Clone, Debug)]
pub struct ValidatorMetrics {
    /// Raw stake weight (before normalization)
    pub stake: u128,
    /// Raw reputation score (before normalization)
    pub reputation: u64,
    /// Coherence score from QBER measurements
    pub coherence: u64,
}

/// Doubly-stochastic influence matrix
/// Row sums = 1 (outgoing influence conserved)
/// Column sums = 1 (incoming trust conserved)
pub struct InfluenceMatrix {
    /// n×n matrix where n = validator count
    data: Vec<Vec<f64>>,
    /// Number of validators
    size: usize,
}

impl InfluenceMatrix {
    /// Create from raw affinity scores
    pub fn from_raw_affinities(raw: Vec<Vec<f64>>) -> Self {
        let size = raw.len();
        Self { data: raw, size }
    }

    /// Apply Sinkhorn-Knopp to make doubly-stochastic
    ///
    /// This is the core conservation constraint:
    /// - Row normalization: each validator's outgoing influence sums to 1
    /// - Column normalization: each validator's incoming trust sums to 1
    ///
    /// Properties preserved:
    /// - Spectral norm ≤ 1 (non-expansive)
    /// - Compositional closure (A₁·A₂ remains doubly-stochastic)
    /// - Convex mixing (bounded output)
    pub fn apply_sinkhorn(&mut self) {
        // Make all entries positive via exp
        for i in 0..self.size {
            for j in 0..self.size {
                self.data[i][j] = self.data[i][j].exp();
            }
        }

        // Alternating row/column normalization
        for _ in 0..SINKHORN_ITERATIONS {
            // Row normalization
            for i in 0..self.size {
                let row_sum: f64 = self.data[i].iter().sum();
                if row_sum > 0.0 {
                    for j in 0..self.size {
                        self.data[i][j] /= row_sum;
                    }
                }
            }

            // Column normalization
            for j in 0..self.size {
                let col_sum: f64 = (0..self.size).map(|i| self.data[i][j]).sum();
                if col_sum > 0.0 {
                    for i in 0..self.size {
                        self.data[i][j] /= col_sum;
                    }
                }
            }
        }
    }

    /// Verify doubly-stochastic properties (for testing)
    pub fn verify_doubly_stochastic(&self, epsilon: f64) -> bool {
        // Check row sums
        for i in 0..self.size {
            let row_sum: f64 = self.data[i].iter().sum();
            if (row_sum - 1.0).abs() > epsilon {
                return false;
            }
        }

        // Check column sums
        for j in 0..self.size {
            let col_sum: f64 = (0..self.size).map(|i| self.data[i][j]).sum();
            if (col_sum - 1.0).abs() > epsilon {
                return false;
            }
        }

        // Check non-negativity
        for i in 0..self.size {
            for j in 0..self.size {
                if self.data[i][j] < 0.0 {
                    return false;
                }
            }
        }

        true
    }

    /// Apply the influence matrix to aggregate validator outputs
    ///
    /// result[i] = Σⱼ A[i][j] * inputs[j]
    ///
    /// This is a convex combination, ensuring bounded output
    pub fn aggregate(&self, inputs: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.size];
        for i in 0..self.size {
            for j in 0..self.size {
                result[i] += self.data[i][j] * inputs[j];
            }
        }
        result
    }
}

/// Build raw affinity matrix from validator metrics
///
/// This creates the initial (unconstrained) affinity scores
/// which will then be projected onto the doubly-stochastic manifold
pub fn build_raw_affinity(validators: &[ValidatorMetrics]) -> Vec<Vec<f64>> {
    let n = validators.len();
    let mut raw = vec![vec![0.0; n]; n];

    // Total metrics for normalization
    let total_stake: u128 = validators.iter().map(|v| v.stake).sum();
    let total_reputation: u64 = validators.iter().map(|v| v.reputation).sum();

    for i in 0..n {
        for j in 0..n {
            // Self-affinity based on own metrics
            if i == j {
                let stake_ratio = validators[i].stake as f64 / total_stake as f64;
                let rep_ratio = validators[i].reputation as f64 / total_reputation as f64;
                let coherence = validators[i].coherence as f64 / 100.0;

                // Weighted combination
                raw[i][j] = 0.4 * stake_ratio + 0.3 * rep_ratio + 0.3 * coherence;
            } else {
                // Cross-validator affinity (symmetric for simplicity)
                let coherence_avg = (validators[i].coherence + validators[j].coherence) as f64 / 200.0;
                raw[i][j] = coherence_avg * 0.5; // Lower than self-affinity
            }
        }
    }

    raw
}

/// Main entry point: compute conservation-constrained validator weights
///
/// This replaces naive stake-weighted aggregation with doubly-stochastic
/// normalization that prevents:
/// - Cartel dominance (no single validator can dominate)
/// - Marginalization (no validator gets zero influence)
/// - Amplification over time (composition remains bounded)
pub fn compute_constrained_weights(validators: &[ValidatorMetrics]) -> InfluenceMatrix {
    let raw = build_raw_affinity(validators);
    let mut matrix = InfluenceMatrix::from_raw_affinities(raw);
    matrix.apply_sinkhorn();

    // Verify constraints are satisfied
    debug_assert!(matrix.verify_doubly_stochastic(1e-6));

    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinkhorn_produces_doubly_stochastic() {
        let validators = vec![
            ValidatorMetrics { stake: 1000, reputation: 80, coherence: 95 },
            ValidatorMetrics { stake: 2000, reputation: 90, coherence: 85 },
            ValidatorMetrics { stake: 500, reputation: 70, coherence: 90 },
            ValidatorMetrics { stake: 1500, reputation: 85, coherence: 88 },
        ];

        let matrix = compute_constrained_weights(&validators);

        // Should be doubly stochastic within numerical tolerance
        assert!(matrix.verify_doubly_stochastic(1e-6));
    }

    #[test]
    fn test_aggregation_is_convex_combination() {
        let validators = vec![
            ValidatorMetrics { stake: 1000, reputation: 80, coherence: 95 },
            ValidatorMetrics { stake: 2000, reputation: 90, coherence: 85 },
        ];

        let matrix = compute_constrained_weights(&validators);

        // Input values
        let inputs = vec![10.0, 20.0];
        let output = matrix.aggregate(&inputs);

        // Output should be within input range (convex combination)
        for val in output {
            assert!(val >= 10.0 && val <= 20.0);
        }
    }

    #[test]
    fn test_no_single_validator_dominates() {
        // Even with vastly different stakes, no validator dominates
        let validators = vec![
            ValidatorMetrics { stake: 1, reputation: 50, coherence: 90 },
            ValidatorMetrics { stake: 1_000_000, reputation: 50, coherence: 90 },
        ];

        let matrix = compute_constrained_weights(&validators);

        // Each row and column sums to 1, so max influence is bounded
        for i in 0..2 {
            for j in 0..2 {
                assert!(matrix.data[i][j] < 1.0);
                assert!(matrix.data[i][j] > 0.0);
            }
        }
    }
}
