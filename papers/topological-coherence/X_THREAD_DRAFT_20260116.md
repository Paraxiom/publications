# X Thread Draft: Toroidal Topology Divergence
**Account:** @ParaxiomAPI
**Date:** January 16, 2026

---

## Thread

**1/7**
We found something unexpected while testing toroidal attention constraints for hallucination reduction.

The same geometric fix that cut Phi-2 hallucinations by 50%... INCREASED TinyLlama hallucinations by 180%.

Same code. Same parameters. Opposite effect.

Thread on why this matters:

---

**2/7**
The method: wrap attention in a Tonnetz topology (12-tone musical lattice on a torus).

Tokens "nearby" on the torus get full attention weight.
Distant tokens get exponentially suppressed.

Theory: geometric constraints = coherent representations = fewer hallucinations.

---

**3/7**
Phi-2 results (2.78B params, no RLHF):

Baseline: 40% hallucination rate
Toroidal: 20% hallucination rate

50% reduction. Reproduced 3x. Exactly what we predicted.

---

**4/7**
Then we tested TinyLlama (1.1B params, no RLHF):

Baseline: 10% hallucination rate
Toroidal: 28% hallucination rate

180% INCREASE. Over 50 samples. Consistent.

The constraint that helps one model actively harms another.

---

**5/7**
Why?

Hypotheses:
- Larger models absorb constraints; smaller models get destabilized
- Phi-2's "textbook" training data aligns with geometric structure
- The 12-tone Tonnetz periodicity fits some attention patterns, disrupts others

We don't know yet. But we know it's real.

---

**6/7**
The takeaway:

There is no universal geometric fix for hallucination.

Any topological constraint must be validated per-architecture before deployment.

"Same medicine, different patient, opposite outcome."

---

**7/7**
Full divergence note with reproducibility details:
[link to divergence note]

Code + results:
github.com/paraxiom/topological-coherence

Next: hyperparameter sweep on GPU to find what topology (if any) works for smaller models.

---

## Alt versions for individual posts

**Standalone hook:**
> We accidentally proved that the same geometric attention constraint that reduces hallucination in one LLM can INCREASE it in another by 180%.
>
> The fix isn't universal. It's architecture-dependent.
>
> Details in thread.

**Visual suggestion:**
Create side-by-side bar chart:
- Left: Phi-2 (40% → 20%, green arrow down)
- Right: TinyLlama (10% → 28%, red arrow up)
- Caption: "Same constraint. Opposite effect."
