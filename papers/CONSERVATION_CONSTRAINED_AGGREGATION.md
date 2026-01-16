# Conservation-Constrained Aggregation in Distributed Trust Networks

**Paraxiom Research Technical Note**
**January 2026**

Sylvain Cormier
sylvain@paraxiom.org

---

## Abstract

We formalize the principle that stable distributed aggregation requires composite mappings to remain within a norm-preserving manifold. This framing unifies recent advances in deep learning architecture (manifold-constrained hyper-connections) with Paraxiom's coherence-based consensus mechanisms. We show that validator weighting, oracle routing, and multi-agent coordination share a common failure mode: unconstrained composition drift. The doubly-stochastic constraint provides a principled solution applicable across domains.

---

## 1. The Conservation Principle

### 1.1 Identity Mapping as Conservation

In residual networks, the identity mapping `x_{l+1} = x_l + F(x_l)` preserves signal magnitude across layers. He et al. (2016) demonstrated that this property is essential for training deep networks: without it, signals either explode or vanish during forward and backward propagation.

Recent work on Hyper-Connections (HC) by Zhu et al. (2024) extended this paradigm by expanding the residual stream width and introducing learnable mixing matrices. While yielding performance gains, unconstrained mixing compromises the identity mapping property. The DeepSeek mHC paper (Xie et al., 2026) formalizes this as:

> **Instability emerges when composite mappings leave the manifold that preserves identity.**

The solution: project mixing matrices onto the Birkhoff polytope (doubly stochastic matrices), ensuring:
1. Row sums = 1 (outgoing signal conserved)
2. Column sums = 1 (incoming gradient conserved)
3. Non-negative entries (no signal cancellation)

### 1.2 Generalization to Distributed Systems

The same principle applies to distributed trust networks:

| ML Context | Distributed Systems Context |
|------------|----------------------------|
| Residual stream | Validator influence stream |
| Layer mixing matrix | Trust/reputation aggregation |
| Signal explosion | Cartel dominance |
| Signal vanishing | Validator marginalization |
| Composite mapping stability | Long-term governance stability |

**Key insight**: Validator influence aggregation is a composite mapping problem. Without conservation constraints, repeated aggregation causes unbounded amplification (cartel effects) or attenuation (marginalization).

---

## 2. Coherence as Constrained Mixing

### 2.1 Reframing Paraxiom's Coherence

Paraxiom's coherence framework can be formally expressed as:

> **Coherence = constrained mixing on a conservation manifold**
> **Instability = unconstrained composition drift**

The Coherence Gadget (QuantumHarmony's finality mechanism) implicitly enforces conservation:
- 2/3 supermajority threshold prevents minority signal explosion
- Falcon1024 signatures provide non-repudiation (no vote cancellation)
- QBER thresholds constrain entropy quality bounds

### 2.2 Formal Definition

Let V = {v_1, ..., v_n} be the validator set with influence weights w_i. Define the influence aggregation matrix A where A_{ij} represents how validator j's actions affect validator i's effective influence.

**Conservation constraint**: A must be doubly stochastic:
```
A ∈ M_DS = {A ∈ R^{n×n} | A·1 = 1, 1^T·A = 1^T, A ≥ 0}
```

**Properties preserved**:
1. **Norm preservation**: ||A||_2 ≤ 1 (non-expansive)
2. **Compositional closure**: A_1 · A_2 ∈ M_DS (stable under repeated application)
3. **Convex mixing**: A·x is a convex combination (bounded output)

---

## 3. Applications

### 3.1 Validator Weighting

**Current approach**: Stake-weighted or reputation-weighted aggregation
```
influence_i = stake_i / Σ stake_j
```

**Conservation-constrained approach**: Doubly-stochastic normalization
```
A = Sinkhorn(raw_weights)  // Row and column sums = 1
influence = A · base_influence
```

**Benefits**:
- Outgoing influence conserved (no single validator dominates)
- Incoming trust conserved (no validator marginalized)
- Provable bounded amplification over time
- Resistance to cartel effects

### 3.2 Oracle/Agent Routing

For oracle routing and federated learning:

**Constraint**: No agent can amplify signal beyond global budget
```
routing_matrix = Sinkhorn(affinity_scores)
aggregated_output = routing_matrix · agent_outputs
```

**Properties**:
- Repeated composition remains bounded
- Information flow is conservative
- No single oracle dominates aggregation

### 3.3 Cross-Domain Aggregation

The 512-segment toroidal mesh in QuantumHarmony performs cross-segment communication. Each segment-to-segment routing can be constrained:

```
segment_routing ∈ M_DS
```

This ensures load balancing without runaway concentration.

---

## 4. Implementation Guidance

### 4.1 Sinkhorn-Knopp Projection

Given raw affinity matrix M^(0):
```
for t in 1..T:
    M^(t) = normalize_rows(normalize_cols(M^(t-1)))
```

Converges to doubly stochastic matrix. T=20 iterations sufficient for practical accuracy.

### 4.2 Integration Points in QuantumHarmony

| Component | Current Implementation | Conservation Enhancement |
|-----------|----------------------|-------------------------|
| Validator selection | substrate-validator-set | Sinkhorn-normalized influence |
| Coherence scoring | QBER thresholds | Doubly-stochastic vote weights |
| Toroidal routing | pallet-runtime-segmentation | Constrained segment affinity |
| Oracle aggregation | pallet-oracle-feeds | Conservative feed mixing |

### 4.3 What NOT to Do

- **Do not** implement Sinkhorn everywhere indiscriminately
- **Do not** claim architectural novelty from applying known techniques
- **Do** frame as: "Recent large-scale training failures independently confirm the necessity of constrained residual topology—something we've been enforcing at the protocol level."

---

## 5. Theoretical Foundations

### 5.1 Connection to Existing Work

| Paraxiom Paper | Conservation Principle |
|----------------|----------------------|
| ERLHS (Cormier, 2025) | Hamiltonian energy conservation for state transitions |
| Karmonic Mesh | Spectral conservation on toroidal topology |
| Proof of Coherence | QBER bounds as entropy conservation |
| Toroidal Governance | Tonnetz manifold as mixing constraint surface |

### 5.2 The Unified View

All these papers address the same underlying problem:

> **How do we ensure that composite operations over distributed state remain stable?**

The mHC paper provides the cleanest formal statement: **project onto a conservation manifold**.

For ML: doubly stochastic matrices on residual mixing.
For consensus: doubly stochastic matrices on validator influence.
For governance: doubly stochastic matrices on voting weight propagation.

**Same math. Bigger surface.**

---

## 6. Strategic Positioning

### 6.1 How to Frame This

**Do say**:
> "Recent large-scale training failures independently confirm the necessity of constrained residual topology—something we've been enforcing at the protocol level."

**Don't say**:
> "We do what DeepSeek does."

### 6.2 Academic Legitimacy

This note establishes:
1. Formal connection between ML architecture and distributed consensus
2. Prior art on conservation-constrained aggregation (existing Paraxiom papers)
3. Independent validation from DeepSeek's empirical findings
4. Clear implementation pathway

---

## 7. Conclusion

Conservation-constrained aggregation is not a new idea—it is implicit in any stable distributed system. The contribution here is making it explicit and providing a unified framework applicable to:

- Validator influence in blockchain consensus
- Agent routing in federated systems
- Trust propagation in multi-party computation
- Any domain where repeated aggregation must remain bounded

The doubly-stochastic constraint (row/column sums = 1) is the minimal sufficient condition for stability under composition. Paraxiom's coherence framework already embodies this principle; this note makes the mathematics explicit.

---

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity mappings in deep residual networks. ECCV.

2. Zhu, D., et al. (2024). Hyper-Connections. arXiv:2409.19606.

3. Xie, Z., et al. (2026). mHC: Manifold-Constrained Hyper-Connections. arXiv:2512.24880.

4. Cormier, S. (2025). ERLHS: A Hamiltonian Framework for Coherence-Preserving Machine Intelligence. Zenodo. DOI: 10.5281/zenodo.17928909

5. Cormier, S. (2025). Karmonic Mesh: Spectral Consensus on Toroidal Manifolds. Zenodo. DOI: 10.5281/zenodo.17928991

6. Cormier, S. (2025). Proof of Coherence: QKD-Based Distributed Consensus. Zenodo. DOI: 10.5281/zenodo.17929054

7. Sinkhorn, R., & Knopp, P. (1967). Concerning nonnegative matrices and doubly stochastic matrices. Pacific Journal of Mathematics.

---

*Paraxiom Research Technical Note. January 2026.*
