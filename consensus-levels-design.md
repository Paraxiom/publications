# Multi-Level Adaptive Consensus System

## Design Document
**Status:** Draft
**Author:** Sylvain Cormier
**Date:** December 2025

---

## Problem

Current log output shows:
```
✨ Finalized #89438 (PoC: simulated mode, no quantum hardware connected)
```

This creates a negative first impression for node operators, suggesting the system is incomplete or running in a degraded state.

---

## Solution: Adaptive Consensus Levels

### Level Definitions

| Level | Name | Display String | Requirements |
|-------|------|----------------|--------------|
| 1 | Classical BFT | `BFT-Classical` | Base Substrate |
| 2 | Post-Quantum BFT | `PQ-BFT` | SPHINCS+ keys loaded |
| 3 | Coherence-Weighted | `PQ-BFT + Coherence` | Level 2 + coherence pallet |
| 4 | QRNG-Enhanced | `PQ-BFT + QRNG` | Level 3 + hardware QRNG |
| 5 | Full PoC | `Proof-of-Coherence` | Level 4 + QKD links |

### Upgrade Path

```
Level 1 → Level 2 → Level 3 → Level 4 → Level 5
   ↑         ↑         ↑         ↑         ↑
  Base    Runtime   Runtime   Hardware  Hardware
          Upgrade   Upgrade   + Node    + Node
```

---

## Implementation

### Phase 1: Runtime Pallet (No Node Restart)

```rust
// pallets/consensus-level/src/lib.rs

#![cfg_attr(not(feature = "std"), no_std)]

pub use pallet::*;

#[frame_support::pallet]
pub mod pallet {
    use frame_support::pallet_prelude::*;
    use frame_system::pallet_prelude::*;

    #[pallet::pallet]
    pub struct Pallet<T>(_);

    #[pallet::config]
    pub trait Config: frame_system::Config {
        type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;
    }

    /// Consensus level (1-5)
    #[derive(Clone, Encode, Decode, TypeInfo, MaxEncodedLen, PartialEq, Eq, Debug)]
    pub enum ConsensusLevel {
        ClassicalBFT = 1,
        PostQuantumBFT = 2,
        CoherenceWeighted = 3,
        QRNGEnhanced = 4,
        FullProofOfCoherence = 5,
    }

    impl Default for ConsensusLevel {
        fn default() -> Self {
            ConsensusLevel::PostQuantumBFT // We already have SPHINCS+
        }
    }

    #[pallet::storage]
    #[pallet::getter(fn current_level)]
    pub type CurrentLevel<T> = StorageValue<_, ConsensusLevel, ValueQuery>;

    #[pallet::storage]
    #[pallet::getter(fn qrng_available)]
    pub type QRNGAvailable<T> = StorageValue<_, bool, ValueQuery>;

    #[pallet::storage]
    #[pallet::getter(fn qkd_links_active)]
    pub type QKDLinksActive<T> = StorageValue<_, u32, ValueQuery>;

    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        /// Consensus level changed
        LevelChanged { old: ConsensusLevel, new: ConsensusLevel },
        /// QRNG hardware detected
        QRNGDetected,
        /// QKD link established
        QKDLinkEstablished { peer_count: u32 },
    }

    #[pallet::hooks]
    impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
        fn on_initialize(_n: BlockNumberFor<T>) -> Weight {
            // Auto-detect and upgrade consensus level each block
            Self::evaluate_and_upgrade();
            Weight::from_parts(10_000, 0)
        }
    }

    impl<T: Config> Pallet<T> {
        /// Evaluate current capabilities and upgrade level if possible
        fn evaluate_and_upgrade() {
            let current = Self::current_level();
            let qrng = Self::qrng_available();
            let qkd_links = Self::qkd_links_active();

            let new_level = match (qrng, qkd_links) {
                (true, n) if n >= 2 => ConsensusLevel::FullProofOfCoherence,
                (true, _) => ConsensusLevel::QRNGEnhanced,
                (false, _) => ConsensusLevel::CoherenceWeighted,
            };

            if new_level != current {
                CurrentLevel::<T>::put(new_level.clone());
                Self::deposit_event(Event::LevelChanged {
                    old: current,
                    new: new_level
                });
            }
        }

        /// Get display string for current level
        pub fn level_display_string() -> &'static str {
            match Self::current_level() {
                ConsensusLevel::ClassicalBFT => "BFT-Classical",
                ConsensusLevel::PostQuantumBFT => "PQ-BFT",
                ConsensusLevel::CoherenceWeighted => "PQ-BFT + Coherence",
                ConsensusLevel::QRNGEnhanced => "PQ-BFT + QRNG",
                ConsensusLevel::FullProofOfCoherence => "Proof-of-Coherence",
            }
        }
    }

    #[pallet::call]
    impl<T: Config> Pallet<T> {
        /// Register QRNG hardware availability (called by node on startup)
        #[pallet::call_index(0)]
        #[pallet::weight(10_000)]
        pub fn register_qrng(origin: OriginFor<T>, available: bool) -> DispatchResult {
            ensure_root(origin)?;
            QRNGAvailable::<T>::put(available);
            if available {
                Self::deposit_event(Event::QRNGDetected);
            }
            Ok(())
        }

        /// Register QKD link status (called by quantum-p2p layer)
        #[pallet::call_index(1)]
        #[pallet::weight(10_000)]
        pub fn register_qkd_link(origin: OriginFor<T>, active_links: u32) -> DispatchResult {
            ensure_root(origin)?;
            QKDLinksActive::<T>::put(active_links);
            Self::deposit_event(Event::QKDLinkEstablished { peer_count: active_links });
            Ok(())
        }
    }
}
```

### Phase 2: RPC Extension (No Node Restart with Runtime Upgrade)

```rust
// rpc/consensus-level/src/lib.rs

use jsonrpsee::{core::RpcResult, proc_macros::rpc};

#[rpc(server)]
pub trait ConsensusLevelApi {
    /// Get current consensus level (1-5)
    #[method(name = "consensusLevel_current")]
    fn current_level(&self) -> RpcResult<u8>;

    /// Get display string for current level
    #[method(name = "consensusLevel_displayString")]
    fn display_string(&self) -> RpcResult<String>;

    /// Get full status including hardware detection
    #[method(name = "consensusLevel_status")]
    fn status(&self) -> RpcResult<ConsensusStatus>;
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ConsensusStatus {
    pub level: u8,
    pub level_name: String,
    pub display_string: String,
    pub qrng_available: bool,
    pub qkd_links_active: u32,
    pub next_level_requirements: Option<String>,
}
```

### Phase 3: Node Logging Update (One-Time Restart)

```rust
// client/consensus/src/finality.rs

// Change from:
log::info!(
    "✨ Finalized #{} (PoC: simulated mode, no quantum hardware connected)",
    block_number
);

// To:
let level_string = runtime_api.consensus_level_display_string(block_hash)?;
log::info!(
    "✨ Finalized #{} (Consensus: {})",
    block_number,
    level_string
);
```

---

## Deployment Plan

### Step 1: Runtime Upgrade (No Restart)
1. Add `pallet-consensus-level` to runtime
2. Set initial level to `CoherenceWeighted` (Level 3)
3. Deploy via runtime upgrade extrinsic

**Result:** Storage now tracks consensus level

### Step 2: Node Update (Scheduled Restart)
1. Update logging to read from runtime
2. Add RPC methods
3. Coordinate restart with validators

**Result:** Logs now show "Consensus: PQ-BFT + Coherence"

### Step 3: Dashboard Update
1. Query `consensusLevel_status` RPC
2. Display current level prominently
3. Show upgrade path to operators

---

## Log Output Examples

**Current (problematic):**
```
✨ Finalized #89438 (PoC: simulated mode, no quantum hardware connected)
```

**After Phase 1+2:**
```
✨ Finalized #89438 (Consensus: PQ-BFT + Coherence)
```

**With QRNG detected:**
```
✨ Finalized #89438 (Consensus: PQ-BFT + QRNG)
```

**Full PoC:**
```
✨ Finalized #89438 (Consensus: Proof-of-Coherence)
```

---

## Coherence Scoring (Level 3)

Even without quantum hardware, coherence-weighted selection provides value:

```rust
pub struct CoherenceScore {
    /// Uptime percentage (0-100)
    pub uptime: u8,
    /// Blocks produced vs expected (0-100)
    pub block_production: u8,
    /// Peer connectivity score (0-100)
    pub connectivity: u8,
    /// Recent slash events (negative factor)
    pub slash_penalty: u8,
}

impl CoherenceScore {
    pub fn total(&self) -> u32 {
        let base = (self.uptime as u32 + self.block_production as u32 + self.connectivity as u32) / 3;
        base.saturating_sub(self.slash_penalty as u32)
    }
}
```

Leader selection weighted by coherence score means:
- Reliable validators produce more blocks
- Flaky validators naturally rotate out
- No quantum hardware needed
- Same algorithm as full PoC, just with classical metrics

---

## Summary

| Change | Restart Required | Timeline |
|--------|------------------|----------|
| Add pallet-consensus-level | No (runtime upgrade) | Immediate |
| Coherence scoring logic | No (runtime upgrade) | Immediate |
| RPC methods | Yes (once) | Scheduled |
| Log message format | Yes (once) | Scheduled |
| QRNG integration | Yes (when hardware ready) | Future |
| QKD integration | Yes (when hardware ready) | Future |

The key insight: **Level 3 (Coherence-Weighted) can be enabled immediately via runtime upgrade**, giving operators a positive message while maintaining upgrade path to full PoC.
