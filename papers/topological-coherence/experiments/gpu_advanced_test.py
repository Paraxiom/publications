#!/usr/bin/env python3
"""
Advanced GPU Tests: Different Models & Topology Variants
========================================================

Tests:
1. Mistral-7B - Different architecture
2. Layer-selective topology - Only apply to certain layers
3. Inverted topology - Boost distant attention instead of suppress
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
import argparse

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# =============================================================================
# TOPOLOGY VARIANTS
# =============================================================================

class TopologyCache:
    """Flexible topology cache supporting multiple variants."""

    def __init__(self, grid_size=12, radius=2.0, alpha=1.0, variant="standard"):
        self.grid_size = grid_size
        self.radius = radius
        self.alpha = alpha
        self.variant = variant  # "standard", "inverted", "layer_early", "layer_late"
        self._cache = {}
        self.current_layer = 0
        self.total_layers = 32  # Will be set based on model

    def set_layer_info(self, current_layer, total_layers):
        self.current_layer = current_layer
        self.total_layers = total_layers

    def _dist(self, i, j):
        xi, yi = i % self.grid_size, (i // self.grid_size) % self.grid_size
        xj, yj = j % self.grid_size, (j // self.grid_size) % self.grid_size
        dx = min(abs(xi - xj), self.grid_size - abs(xi - xj))
        dy = min(abs(yi - yj), self.grid_size - abs(yi - yj))
        return dx + dy

    def should_apply(self):
        """Check if topology should be applied based on variant and layer."""
        if self.variant == "standard":
            return True
        elif self.variant == "inverted":
            return True
        elif self.variant == "layer_early":
            # Only first 1/3 of layers
            return self.current_layer < self.total_layers // 3
        elif self.variant == "layer_late":
            # Only last 1/3 of layers
            return self.current_layer >= 2 * self.total_layers // 3
        elif self.variant == "layer_middle":
            # Only middle 1/3 of layers
            third = self.total_layers // 3
            return third <= self.current_layer < 2 * third
        return True

    def get_bias(self, seq_len, device, dtype):
        # Check if we should apply based on layer
        if not self.should_apply():
            # Return zeros (no modification)
            key = (seq_len, "none", str(device), str(dtype))
            if key not in self._cache:
                causal = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
                self._cache[key] = causal.unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype).contiguous()
            return self._cache[key]

        key = (seq_len, self.variant, self.radius, self.alpha, str(device), str(dtype))
        if key not in self._cache:
            mask = torch.zeros(seq_len, seq_len)
            for i in range(seq_len):
                for j in range(seq_len):
                    d = self._dist(i, j)

                    if self.variant == "inverted":
                        # INVERTED: Boost distant, suppress nearby
                        if d <= self.radius:
                            mask[i, j] = -self.alpha * (self.radius - d + 1)  # Suppress nearby
                        else:
                            mask[i, j] = 0.0  # Allow distant
                    else:
                        # STANDARD: Suppress distant, allow nearby
                        if d <= self.radius:
                            mask[i, j] = 0.0
                        else:
                            mask[i, j] = -self.alpha * d

            causal = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
            combined = (mask + causal).unsqueeze(0).unsqueeze(0)
            self._cache[key] = combined.to(device=device, dtype=dtype).contiguous()
        return self._cache[key]


CACHE = TopologyCache()
LAYER_COUNTER = [0]  # Mutable to track layers


def make_toroidal_attention(model_type="llama"):
    """Create attention function for specific model type."""

    def toroidal_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        # Get repeat_kv for the model type
        if model_type == "llama" or model_type == "mistral":
            from transformers.models.llama.modeling_llama import repeat_kv
        else:
            from transformers.models.llama.modeling_llama import repeat_kv

        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

        seq_len, key_len = query.shape[2], key_states.shape[2]

        # Update layer counter
        CACHE.set_layer_info(LAYER_COUNTER[0], CACHE.total_layers)
        LAYER_COUNTER[0] = (LAYER_COUNTER[0] + 1) % CACHE.total_layers

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

    return toroidal_attention


# =============================================================================
# BENCHMARKS (Same as before)
# =============================================================================

def load_benchmarks(n_samples=30):
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

    return (
        [truthfulqa[i % len(truthfulqa)].copy() for i in range(n_samples)],
        [halueval[i % len(halueval)].copy() for i in range(n_samples)]
    )


def evaluate(model, tokenizer, truthfulqa, halueval):
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

    hallucinations = 0
    for sample in tqdm(halueval, desc="HaluEval", leave=False):
        inputs = tokenizer(sample["prompt"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, do_sample=False,
                               pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0], skip_special_tokens=True).lower()

        if sample["hallucination_prone"]:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tinyllama", choices=["tinyllama", "mistral", "gemma", "qwen"])
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--test", default="all", choices=["all", "inverted", "layer", "baseline"])
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    print("=" * 60)
    print("ADVANCED GPU TESTS")
    print("=" * 60)

    # Model selection
    models = {
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "gemma": "google/gemma-2b-it",
        "qwen": "Qwen/Qwen2-1.5B-Instruct",
    }

    model_name = models[args.model]
    print(f"\nLoading {model_name}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Count layers
    n_layers = len([n for n, _ in model.named_modules() if 'self_attn' in n and 'proj' not in n])
    CACHE.total_layers = max(n_layers, 1)
    print(f"Model has {CACHE.total_layers} attention layers")

    # Register custom attention
    toroidal_fn = make_toroidal_attention(args.model)
    ALL_ATTENTION_FUNCTIONS["toroidal"] = toroidal_fn

    # Load benchmarks
    truthfulqa, halueval = load_benchmarks(args.n_samples)
    print(f"Loaded {args.n_samples} samples each")

    # Test configurations
    if args.test == "all":
        configs = [
            {"name": "baseline", "apply": False},
            {"name": "standard_r2a1", "apply": True, "variant": "standard", "radius": 2.0, "alpha": 1.0},
            {"name": "inverted_r2a1", "apply": True, "variant": "inverted", "radius": 2.0, "alpha": 1.0},
            {"name": "layer_early", "apply": True, "variant": "layer_early", "radius": 2.0, "alpha": 1.0},
            {"name": "layer_late", "apply": True, "variant": "layer_late", "radius": 2.0, "alpha": 1.0},
        ]
    elif args.test == "inverted":
        configs = [
            {"name": "baseline", "apply": False},
            {"name": "inverted_r2a1", "apply": True, "variant": "inverted", "radius": 2.0, "alpha": 1.0},
            {"name": "inverted_r2a0.5", "apply": True, "variant": "inverted", "radius": 2.0, "alpha": 0.5},
        ]
    elif args.test == "layer":
        configs = [
            {"name": "baseline", "apply": False},
            {"name": "layer_early", "apply": True, "variant": "layer_early", "radius": 2.0, "alpha": 1.0},
            {"name": "layer_middle", "apply": True, "variant": "layer_middle", "radius": 2.0, "alpha": 1.0},
            {"name": "layer_late", "apply": True, "variant": "layer_late", "radius": 2.0, "alpha": 1.0},
        ]
    else:
        configs = [{"name": "baseline", "apply": False}]

    results = {}

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"{'='*60}")

        LAYER_COUNTER[0] = 0  # Reset layer counter

        if config.get("apply"):
            CACHE.variant = config["variant"]
            CACHE.radius = config["radius"]
            CACHE.alpha = config["alpha"]
            CACHE._cache = {}  # Clear cache
            model.config._attn_implementation = "toroidal"
            print(f"  Variant: {config['variant']}, r={config['radius']}, α={config['alpha']}")
        else:
            model.config._attn_implementation = "eager"
            print("  Baseline (no topology)")

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
    print(f"RESULTS SUMMARY - {args.model.upper()}")
    print("=" * 60)
    print(f"\n{'Config':<20} {'TruthfulQA':<12} {'HaluEval':<12} {'Time':<10}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<20} {r['truthfulqa']:>10.1%} {r['halueval']:>10.1%} {r['time']:>8.1f}s")

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outpath = Path("results") / f"advanced_{args.model}_{timestamp}.json"
    outpath.parent.mkdir(exist_ok=True)
    with open(outpath, "w") as f:
        json.dump({"model": model_name, "results": results}, f, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
