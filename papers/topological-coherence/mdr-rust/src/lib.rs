//! # MDR - Meta Distributed Representations
//!
//! A Rust implementation of sparse, topological, temporal neural representations.
//!
//! ## Core Concepts
//!
//! - **Sparsity**: Only ~2% of units active (from SDR)
//! - **Ternary Weights**: {-1, 0, +1} (from BitNet)
//! - **Toroidal Topology**: Tonnetz-based attention constraints
//! - **Temporal Context**: Time-decaying memory

pub mod tensor;
pub mod topology;
pub mod attention;
pub mod layer;
pub mod model;
pub mod training;
pub mod tokenizer;

pub use tensor::{TernaryTensor, SparseTensor};
pub use topology::Tonnetz;
pub use attention::ToroidalAttention;
pub use layer::MDRLayer;
pub use model::MDRModel;
