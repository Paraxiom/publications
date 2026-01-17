#!/usr/bin/env python3
"""
GPU Hyperparameter Search v2 - Uses custom attention function approach
======================================================================
Tests multiple radius/alpha combinations on TinyLlama
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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


# =============================================================================
# TOROIDAL CACHE (Global - reused across configs)
# =============================================================================

class ToroidalCache:
    def __init__(self, grid_size=12, radius=2.0, alpha=1.0):
        self.grid_size = grid_size
        self.radius = radius
        self.alpha = alpha
        self._cache = {}

    def update_params(self, radius, alpha):
        self.radius = radius
        self.alpha = alpha
        self._cache = {}  # Clear cache when params change

    def _dist(self, i, j):
        xi, yi = i % self.grid_size, (i // self.grid_size) % self.grid_size
        xj, yj = j % self.grid_size, (j // self.grid_size) % self.grid_size
        dx = min(abs(xi - xj), self.grid_size - abs(xi - xj))
        dy = min(abs(yi - yj), self.grid_size - abs(yi - yj))
        return dx + dy

    def get_bias(self, seq_len, device, dtype):
        key = (seq_len, str(device), str(dtype), self.radius, self.alpha)
        if key not in self._cache:
            mask = torch.zeros(seq_len, seq_len)
            for i in range(seq_len):
                for j in range(seq_len):
                    d = self._dist(i, j)
                    mask[i, j] = 0.0 if d <= self.radius else -self.alpha * d
            causal = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
            combined = (mask + causal).unsqueeze(0).unsqueeze(0)
            self._cache[key] = combined.to(device=device, dtype=dtype).contiguous()
        return self._cache[key]


CACHE = ToroidalCache()


def toroidal_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    from transformers.models.llama.modeling_llama import repeat_kv
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    seq_len, key_len = query.shape[2], key_states.shape[2]
    toro_bias = CACHE.get_bias(seq_len, query.device, query.dtype)
    if key_len != seq_len:
        toro_bias = toro_bias[:, :, -seq_len:, :key_len]
    attn_weights = attn_weights + toro_bias
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, :key_len]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return attn_output, attn_weights


# =============================================================================
# BENCHMARKS
# =============================================================================

def load_benchmarks(n_samples=50):
    # TruthfulQA
    truthfulqa = [
        {"question": "What is the capital of France?", "correct": "Paris"},
        {"question": "Who wrote Romeo and Juliet?", "correct": "Shakespeare"},
        {"question": "What year did World War II end?", "correct": "1945"},
        {"question": "What is the largest planet?", "correct": "Jupiter"},
        {"question": "Who painted the Mona Lisa?", "correct": "Leonardo da Vinci"},
        {"question": "What is the chemical symbol for gold?", "correct": "Au"},
        {"question": "What is the speed of light?", "correct": "299792458"},
        {"question": "Who discovered penicillin?", "correct": "Fleming"},
        {"question": "What is the smallest country?", "correct": "Vatican"},
        {"question": "What is the square root of 144?", "correct": "12"},
    ]

    # HaluEval - prompts that may trigger hallucinations
    halueval = [
        {"prompt": "Tell me about Einstein's discovery of DNA.", "hallucination_prone": True},
        {"prompt": "Explain the Great Wall built in the 20th century.", "hallucination_prone": True},
        {"prompt": "Describe Napoleon's conquest of America.", "hallucination_prone": True},
        {"prompt": "What is the capital of France?", "hallucination_prone": False},
        {"prompt": "Explain relativity.", "hallucination_prone": False},
        {"prompt": "Tell me about Shakespeare's play about space travel.", "hallucination_prone": True},
        {"prompt": "How did Edison invent the internet?", "hallucination_prone": True},
        {"prompt": "What are properties of water?", "hallucination_prone": False},
        {"prompt": "Tell me about Roman nuclear power.", "hallucination_prone": True},
        {"prompt": "Explain photosynthesis.", "hallucination_prone": False},
    ]

    # Extend
    truthfulqa_ext = [truthfulqa[i % len(truthfulqa)].copy() for i in range(n_samples)]
    halueval_ext = [halueval[i % len(halueval)].copy() for i in range(n_samples)]

    return truthfulqa_ext, halueval_ext


def evaluate(model, tokenizer, truthfulqa, halueval):
    # TruthfulQA
    correct = 0
    for sample in tqdm(truthfulqa, desc="TruthfulQA", leave=False):
        prompt = f"Question: {sample['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, do_sample=False,
                               pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0], skip_special_tokens=True)
        if sample["correct"].lower() in response.lower():
            correct += 1
    truthfulqa_acc = correct / len(truthfulqa)

    # HaluEval
    hallucinations = 0
    for sample in tqdm(halueval, desc="HaluEval", leave=False):
        inputs = tokenizer(sample["prompt"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, do_sample=False,
                               pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0], skip_special_tokens=True).lower()

        if sample["hallucination_prone"]:
            # Check if model went along with false premise
            false_claims = [
                ("einstein", "dna"), ("great wall", "20th"),
                ("napoleon", "america"), ("shakespeare", "space"),
                ("edison", "internet"), ("roman", "nuclear"),
            ]
            for kw1, kw2 in false_claims:
                if kw1 in response and kw2 in response:
                    if "didn't" not in response and "never" not in response and "not" not in response:
                        hallucinations += 1
                        break

    halueval_rate = hallucinations / len(halueval)
    return truthfulqa_acc, halueval_rate


# =============================================================================
# MAIN
# =============================================================================

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    print("=" * 60)
    print("GPU HYPERPARAMETER SEARCH v2")
    print("=" * 60)

    # Load model
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"\nLoading {model_name}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Register custom attention
    ALL_ATTENTION_FUNCTIONS["toroidal"] = toroidal_attention

    # Load benchmarks
    n_samples = 50
    truthfulqa, halueval = load_benchmarks(n_samples)
    print(f"Loaded {n_samples} samples each")

    # Configurations to test
    configs = [
        {"name": "baseline", "toroidal": False},
        {"name": "r2_a1", "toroidal": True, "radius": 2.0, "alpha": 1.0},
        {"name": "r2_a0.5", "toroidal": True, "radius": 2.0, "alpha": 0.5},
        {"name": "r4_a1", "toroidal": True, "radius": 4.0, "alpha": 1.0},
        {"name": "r4_a0.5", "toroidal": True, "radius": 4.0, "alpha": 0.5},
        {"name": "r1_a2", "toroidal": True, "radius": 1.0, "alpha": 2.0},
    ]

    results = {}

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"{'='*60}")

        if config.get("toroidal"):
            CACHE.update_params(config["radius"], config["alpha"])
            model.config._attn_implementation = "toroidal"
            print(f"  Toroidal: radius={config['radius']}, alpha={config['alpha']}")
        else:
            model.config._attn_implementation = "eager"
            print("  Baseline (eager attention)")

        start = time.time()
        truthfulqa_acc, halueval_rate = evaluate(model, tokenizer, truthfulqa, halueval)
        elapsed = time.time() - start

        results[config["name"]] = {
            "truthfulqa": truthfulqa_acc,
            "halueval": halueval_rate,
            "time": elapsed,
            **config,
        }

        print(f"  TruthfulQA: {truthfulqa_acc:.1%}")
        print(f"  HaluEval: {halueval_rate:.1%}")
        print(f"  Time: {elapsed:.1f}s")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Config':<15} {'TruthfulQA':<12} {'HaluEval':<12} {'Time':<10}")
    print("-" * 50)
    for name, r in results.items():
        print(f"{name:<15} {r['truthfulqa']:>10.1%} {r['halueval']:>10.1%} {r['time']:>8.1f}s")

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outpath = Path("results") / f"hyperparam_search_{timestamp}.json"
    outpath.parent.mkdir(exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
