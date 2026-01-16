# Updated Abstract for Paper v2

**Paper:** "Topological Constraints for Coherent Language Models: Bounding Latent Drift via Spectral Gaps on Toroidal Manifolds"

**Changes:** Incorporates architecture-dependent effect discovery

---

## Proposed Abstract (v2)

We argue that hallucination in large language models is driven in part by unconstrained latent dynamics: residual updates evolve in high-dimensional Euclidean space without contractive structure that bounds drift. Recent work on Hyper-Connections shows that unconstrained residual mixing can destabilize training, and that projecting mixing matrices onto the Birkhoff polytope (doubly-stochastic) restores stability.

We unify these observations within an ERLHS-style coherence framework: doubly-stochastic projection is a special case of non-expansive evolution on a constrained set. The constraint ensures bounded mixing but does not, by itself, impose a neighborhood graph or Laplacian spectrum that filters incoherent modes. We provide a constructive topology—the 2D torus (Tonnetz)—as an architectural prior that introduces a spectral gap λ₁ = Θ(1) for fixed side length, enabling explicit suppression of high-frequency drift.

This establishes a hierarchy of sufficient conditions for coherence: mHC (Birkhoff contractivity) ⊂ ERLHS (Hamiltonian-inspired bounds) ⊂ Karmonic (Toroidal + Spectral filtering).

**Experimental validation**: We test toroidal attention constraints on models without RLHF alignment to isolate architectural effects. On Phi-2 (2.78B), the constraint reduces HaluEval hallucination rate by 50% (40% → 20%). However, we report a critical negative result: on TinyLlama (1.1B), the identical constraint *increases* hallucination by 180% (10% → 28%). **The effect is architecture-dependent.** This divergence invalidates universal applicability claims and opens fundamental questions about architecture-topology compatibility. We hypothesize that model capacity, training data structure, and attention pattern distribution determine whether a given topology helps or harms. Code and reproducibility details at https://github.com/Paraxiom/topological-coherence.

---

## Key Changes from v1

1. **Added negative result**: TinyLlama shows opposite effect
2. **Removed universal claim**: Changed from "achieves +19.5% relative improvement" to architecture-specific results
3. **Added warning**: "The effect is architecture-dependent"
4. **Reframed contribution**: From "this works" to "this works conditionally, and understanding the conditions is the real research question"

---

## Suggested Title Update (Optional)

**Original:**
"Topological Constraints for Coherent Language Models: Bounding Latent Drift via Spectral Gaps on Toroidal Manifolds"

**Proposed v2:**
"Topological Constraints for Coherent Language Models: Architecture-Dependent Effects of Spectral Gaps on Toroidal Manifolds"

or

"Topological Constraints for Coherent Language Models: When Geometry Helps and When It Harms"
