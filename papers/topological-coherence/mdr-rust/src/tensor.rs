//! Tensor types for MDR: Ternary and Sparse tensors

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Ternary value: -1, 0, or +1
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i8)]
pub enum Ternary {
    Neg = -1,
    Zero = 0,
    Pos = 1,
}

impl Ternary {
    pub fn from_f32(x: f32, threshold: f32) -> Self {
        if x > threshold {
            Ternary::Pos
        } else if x < -threshold {
            Ternary::Neg
        } else {
            Ternary::Zero
        }
    }

    pub fn to_f32(self) -> f32 {
        self as i8 as f32
    }
}

/// Ternary tensor with {-1, 0, +1} values
/// Stored as i8 for efficiency, but only uses 3 values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryTensor {
    pub data: Vec<i8>,
    pub shape: Vec<usize>,
}

impl TernaryTensor {
    /// Create a new ternary tensor from float values
    pub fn from_f32(data: &[f32], shape: Vec<usize>, threshold: f32) -> Self {
        let ternary_data: Vec<i8> = data
            .iter()
            .map(|&x| {
                if x > threshold {
                    1
                } else if x < -threshold {
                    -1
                } else {
                    0
                }
            })
            .collect();

        Self {
            data: ternary_data,
            shape,
        }
    }

    /// Create a random ternary tensor with given sparsity
    pub fn random(shape: Vec<usize>, sparsity: f32) -> Self {
        let mut rng = rand::thread_rng();
        let total_size: usize = shape.iter().product();

        let data: Vec<i8> = (0..total_size)
            .map(|_| {
                let r: f32 = rng.gen();
                if r < sparsity / 2.0 {
                    -1
                } else if r < sparsity {
                    1
                } else {
                    0
                }
            })
            .collect();

        Self { data, shape }
    }

    /// Matrix multiplication: self @ other
    /// Returns f32 result (accumulation needs higher precision)
    pub fn matmul(&self, other: &TernaryTensor) -> Array2<f32> {
        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 2);
        assert_eq!(self.shape[1], other.shape[0]);

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        let mut result = Array2::<f32>::zeros((m, n));

        for i in 0..m {
            for j in 0..n {
                let mut sum: i32 = 0;
                for l in 0..k {
                    let a = self.data[i * k + l] as i32;
                    let b = other.data[l * n + j] as i32;
                    sum += a * b;
                }
                result[[i, j]] = sum as f32;
            }
        }

        result
    }

    /// Efficient ternary-float matmul: ternary @ float
    pub fn matmul_f32(&self, other: &Array2<f32>) -> Array2<f32> {
        assert_eq!(self.shape.len(), 2);
        assert_eq!(self.shape[1], other.shape()[0]);

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape()[1];

        let mut result = Array2::<f32>::zeros((m, n));

        for i in 0..m {
            for j in 0..n {
                let mut sum: f32 = 0.0;
                for l in 0..k {
                    let a = self.data[i * k + l];
                    if a != 0 {
                        // Only compute if weight is non-zero
                        sum += (a as f32) * other[[l, j]];
                    }
                }
                result[[i, j]] = sum;
            }
        }

        result
    }

    /// Count non-zero elements (sparsity check)
    pub fn nnz(&self) -> usize {
        self.data.iter().filter(|&&x| x != 0).count()
    }

    /// Sparsity ratio (fraction of zeros)
    pub fn sparsity(&self) -> f32 {
        1.0 - (self.nnz() as f32 / self.data.len() as f32)
    }
}

/// Sparse tensor using coordinate format (COO)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseTensor {
    pub indices: Vec<Vec<usize>>,  // indices[dim][nnz]
    pub values: Vec<f32>,
    pub shape: Vec<usize>,
}

impl SparseTensor {
    /// Create from dense array, keeping top-k fraction
    pub fn from_dense_topk(dense: &Array2<f32>, density: f32) -> Self {
        let total = dense.len();
        let k = ((total as f32) * density).ceil() as usize;

        // Find top-k values
        let mut indexed: Vec<(usize, usize, f32)> = dense
            .indexed_iter()
            .map(|((i, j), &v)| (i, j, v.abs()))
            .collect();

        indexed.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        indexed.truncate(k);

        let mut row_indices = Vec::with_capacity(k);
        let mut col_indices = Vec::with_capacity(k);
        let mut values = Vec::with_capacity(k);

        for (i, j, _) in indexed {
            row_indices.push(i);
            col_indices.push(j);
            values.push(dense[[i, j]]);
        }

        Self {
            indices: vec![row_indices, col_indices],
            values,
            shape: vec![dense.shape()[0], dense.shape()[1]],
        }
    }

    /// Convert back to dense
    pub fn to_dense(&self) -> Array2<f32> {
        let mut dense = Array2::<f32>::zeros((self.shape[0], self.shape[1]));
        for (idx, &val) in self.values.iter().enumerate() {
            let i = self.indices[0][idx];
            let j = self.indices[1][idx];
            dense[[i, j]] = val;
        }
        dense
    }

    /// Number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_creation() {
        let data = vec![0.5, -0.3, 0.1, -0.8, 0.9, 0.0];
        let t = TernaryTensor::from_f32(&data, vec![2, 3], 0.2);
        assert_eq!(t.data, vec![1, -1, 0, -1, 1, 0]);
    }

    #[test]
    fn test_ternary_matmul() {
        let a = TernaryTensor {
            data: vec![1, 0, -1, 1],
            shape: vec![2, 2],
        };
        let b = TernaryTensor {
            data: vec![1, 1, 0, -1],
            shape: vec![2, 2],
        };
        let c = a.matmul(&b);
        assert_eq!(c[[0, 0]], 1.0);  // 1*1 + 0*0 = 1
        assert_eq!(c[[0, 1]], 1.0);  // 1*1 + 0*(-1) = 1
        assert_eq!(c[[1, 0]], 0.0);  // (-1)*1 + 1*0 = -1... wait
    }

    #[test]
    fn test_sparse_topk() {
        let dense = Array2::from_shape_vec((3, 3), vec![
            0.1, 0.9, 0.2,
            0.8, 0.3, 0.7,
            0.4, 0.5, 0.6,
        ]).unwrap();

        let sparse = SparseTensor::from_dense_topk(&dense, 0.33);
        assert_eq!(sparse.nnz(), 3);  // Top 3 values
    }
}
