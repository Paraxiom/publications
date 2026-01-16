"""
GPU Hyperparameter Search: Find optimal toroidal parameters per architecture
=============================================================================
Run on runpod/lambda/vast.ai with: python gpu_hyperparameter_search.py

Tests multiple radius/alpha combinations to find what works for each model.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from pathlib import Path
import json
import time
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")


class TonnetzTopology:
    def __init__(self, grid_size: int = 12):
        self.grid_size = grid_size
        self._cache = {}

    def toroidal_distance(self, i: int, j: int) -> int:
        xi, yi = i % self.grid_size, (i // self.grid_size) % self.grid_size
        xj, yj = j % self.grid_size, (j // self.grid_size) % self.grid_size
        dx = min(abs(xi - xj), self.grid_size - abs(xi - xj))
        dy = min(abs(yi - yj), self.grid_size - abs(yi - yj))
        return dx + dy

    def create_attention_mask(self, seq_len: int, radius: float, alpha: float, device=DEVICE):
        cache_key = (seq_len, radius, alpha, self.grid_size)
        if cache_key in self._cache:
            return self._cache[cache_key].to(device)

        mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(seq_len):
                d = self.toroidal_distance(i, j)
                mask[i, j] = 1.0 if d <= radius else np.exp(-alpha * d)

        self._cache[cache_key] = mask
        return mask.to(device)


class LinearTopology:
    """Alternative: Simple linear distance (no wraparound)"""
    def __init__(self):
        self._cache = {}

    def create_attention_mask(self, seq_len: int, radius: float, alpha: float, device=DEVICE):
        cache_key = (seq_len, radius, alpha)
        if cache_key in self._cache:
            return self._cache[cache_key].to(device)

        mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(seq_len):
                d = abs(i - j)
                mask[i, j] = 1.0 if d <= radius else np.exp(-alpha * d)

        self._cache[cache_key] = mask
        return mask.to(device)


def load_model(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        device_map="auto" if DEVICE.type == "cuda" else None,
        trust_remote_code=True,
    )

    if DEVICE.type != "cuda":
        model = model.to(DEVICE)

    model.eval()
    print(f"Loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")
    return model, tokenizer


def apply_topology_constraint(model, topology, radius: float, alpha: float):
    """Apply topological constraint to all attention layers."""
    n_wrapped = 0
    for name, module in model.named_modules():
        if 'self_attn' in name and hasattr(module, 'q_proj'):
            original_forward = module.forward

            def make_wrapper(orig_fwd, topo, r, a):
                def wrapper(hidden_states, position_embeddings, attention_mask=None,
                           past_key_values=None, cache_position=None, **kwargs):
                    seq_len = hidden_states.shape[1]
                    batch_size = hidden_states.shape[0]
                    dtype = hidden_states.dtype
                    device = hidden_states.device

                    # Create fresh contiguous tensor for SDPA compatibility
                    new_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=dtype, device=device)

                    # Fill with toroidal bias + causal mask
                    topo_mask = topo.create_attention_mask(seq_len, r, a, device=device)
                    topo_bias = torch.log(topo_mask + 1e-10).to(dtype)
                    causal_bias = torch.triu(torch.full((seq_len, seq_len), float('-inf'), dtype=dtype, device=device), diagonal=1)
                    combined = topo_bias + causal_bias

                    # Broadcast to all batches
                    new_mask[:, 0, :, :] = combined

                    if attention_mask is not None:
                        attention_mask = attention_mask + new_mask
                    else:
                        attention_mask = new_mask

                    return orig_fwd(hidden_states, position_embeddings, attention_mask,
                                   past_key_values, cache_position, **kwargs)
                return wrapper

            module.forward = make_wrapper(original_forward, topology, radius, alpha)
            n_wrapped += 1

    return n_wrapped


def load_benchmarks(n_samples: int = 20):
    """Load benchmark data."""
    from datasets import load_dataset

    # TruthfulQA
    print("Loading TruthfulQA...")
    tqa = load_dataset("truthful_qa", "generation", split="validation")
    truthfulqa = [{"prompt": item["question"],
                   "correct": item.get("correct_answers", []),
                   "incorrect": item.get("incorrect_answers", [])}
                  for item in list(tqa)[:n_samples]]

    # HaluEval
    print("Loading HaluEval...")
    halu = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    halueval = [{"prompt": item["question"],
                 "knowledge": item.get("knowledge", ""),
                 "right": item.get("right_answer", ""),
                 "hallucinated": item.get("hallucinated_answer", "")}
                for item in list(halu)[:n_samples]]

    return truthfulqa, halueval


def evaluate(model, tokenizer, truthfulqa, halueval, max_tokens=50):
    """Quick evaluation."""
    # TruthfulQA
    tqa_correct = 0
    for item in tqdm(truthfulqa, desc="TruthfulQA"):
        inputs = tokenizer(item["prompt"], return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)

        response = tokenizer.decode(out[0], skip_special_tokens=True).lower()
        has_correct = any(a.lower() in response for a in item["correct"])
        has_incorrect = any(a.lower() in response for a in item["incorrect"])
        if has_correct and not has_incorrect:
            tqa_correct += 1

    # HaluEval (detect hallucination)
    halu_score = 0
    for item in tqdm(halueval, desc="HaluEval"):
        prompt = f"Knowledge: {item['knowledge']}\nQuestion: {item['prompt']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)

        response = tokenizer.decode(out[0], skip_special_tokens=True).lower()

        # Check for hallucination keywords
        predicts_hallucination = any(w in response for w in ["incorrect", "wrong", "false", "no,"])
        predicts_correct = any(w in response for w in ["correct", "right", "true", "yes,"])

        if predicts_correct and not predicts_hallucination:
            halu_score += 1

    return {
        "truthfulqa": tqa_correct / len(truthfulqa) if truthfulqa else 0,
        "halueval": halu_score / len(halueval) if halueval else 0
    }


def run_hyperparameter_search():
    """Main search function."""

    MODELS = {
        "phi-2": "microsoft/phi-2",
        "tinyllama": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    }

    TOPOLOGIES = {
        "tonnetz_12": TonnetzTopology(grid_size=12),
        "tonnetz_6": TonnetzTopology(grid_size=6),
        "linear": LinearTopology(),
    }

    PARAMS = [
        {"radius": 2.0, "alpha": 1.0},   # Original
        {"radius": 4.0, "alpha": 0.5},   # Weaker
        {"radius": 6.0, "alpha": 0.3},   # Much weaker
        {"radius": 1.0, "alpha": 2.0},   # Stronger
    ]

    N_SAMPLES = 10  # Reduced for faster GPU runs

    # Load benchmarks once
    truthfulqa, halueval = load_benchmarks(N_SAMPLES)

    all_results = {}

    for model_key, model_name in MODELS.items():
        print(f"\n{'='*60}")
        print(f"TESTING: {model_key}")
        print(f"{'='*60}")

        all_results[model_key] = {}

        # Baseline (no constraint)
        model, tokenizer = load_model(model_name)
        print("\nRunning BASELINE...")
        baseline = evaluate(model, tokenizer, truthfulqa, halueval)
        all_results[model_key]["baseline"] = baseline
        print(f"Baseline: TruthfulQA={baseline['truthfulqa']:.1%}, HaluEval={baseline['halueval']:.1%}")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Test each topology + params combination
        for topo_name, topology in TOPOLOGIES.items():
            for params in PARAMS:
                config_name = f"{topo_name}_r{params['radius']}_a{params['alpha']}"
                print(f"\nTesting {config_name}...")

                model, tokenizer = load_model(model_name)
                n_wrapped = apply_topology_constraint(model, topology, params["radius"], params["alpha"])
                print(f"Wrapped {n_wrapped} attention layers")

                result = evaluate(model, tokenizer, truthfulqa, halueval)
                all_results[model_key][config_name] = result

                # Compare to baseline
                tqa_diff = result["truthfulqa"] - baseline["truthfulqa"]
                halu_diff = result["halueval"] - baseline["halueval"]
                print(f"Result: TruthfulQA={result['truthfulqa']:.1%} ({tqa_diff:+.1%}), "
                      f"HaluEval={result['halueval']:.1%} ({halu_diff:+.1%})")

                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"results/hyperparameter_search_{timestamp}.json"
    Path("results").mkdir(exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("="*70)

    for model_key, results in all_results.items():
        print(f"\n{model_key}:")
        baseline = results["baseline"]
        print(f"  {'Config':<35} {'TruthfulQA':>12} {'HaluEval':>12}")
        print(f"  {'-'*60}")
        print(f"  {'baseline':<35} {baseline['truthfulqa']:>11.1%} {baseline['halueval']:>11.1%}")

        for config, res in results.items():
            if config == "baseline":
                continue
            tqa_diff = res["truthfulqa"] - baseline["truthfulqa"]
            halu_diff = res["halueval"] - baseline["halueval"]
            marker = "**" if halu_diff < -0.05 else ""  # Highlight improvements
            print(f"  {config:<35} {res['truthfulqa']:>11.1%} {res['halueval']:>11.1%} {marker}")

    print(f"\nResults saved to: {results_file}")
    return all_results


if __name__ == "__main__":
    run_hyperparameter_search()
