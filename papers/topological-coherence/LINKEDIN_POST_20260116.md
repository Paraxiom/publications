# LinkedIn Post: Divergence Note

---

**When the same fix helps one AI model and harms another**

We tested toroidal attention constraints to reduce hallucination in language models. The theory: geometric structure in attention patterns should improve coherence.

Results on Phi-2 (2.78B): 50% hallucination reduction. Exactly as predicted.

Then we tested TinyLlama (1.1B) with identical parameters.

Result: 180% hallucination *increase*.

Same constraint. Same code. Opposite effect.

This isn't a failure—it's a finding. It means:

1. There is no universal geometric fix for hallucination
2. Architecture and topology must be matched
3. Any intervention requires per-model validation

The implication for AI safety: "works on GPT-4" doesn't mean "works on Llama" doesn't mean "works on your fine-tune." Interventions that improve one system can actively degrade another.

We've published a divergence note with full reproducibility details:
https://doi.org/10.5281/zenodo.18267913

Code: https://github.com/Paraxiom/topological-coherence

Next step: hyperparameter sweep to understand *why* the divergence occurs and whether alternative topologies can help smaller models.

#AI #MachineLearning #LLM #Research #AIAlignment #Hallucination

---

**Character count:** ~1,150 (LinkedIn limit is 3,000)
