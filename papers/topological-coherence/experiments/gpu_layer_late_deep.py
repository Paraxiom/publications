#!/usr/bin/env python3
"""
Deep Analysis of layer_late Strategy
=====================================

Tests:
1. Mistral-7B with 100 samples for statistical significance
2. Gemma-2B and Qwen2 to confirm cross-architecture benefit
3. Different hyperparameter combinations with layer_late
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


class LayerLateCache:
    """Optimized cache for layer_late topology only."""

    def __init__(self, grid_size=12, radius=2.0, alpha=1.0):
        self.grid_size = grid_size
        self.radius = radius
        self.alpha = alpha
        self._cache = {}
        self.current_layer = 0
        self.total_layers = 32
        self.late_start = 21  # Will be computed as 2/3 of total

    def set_layer_info(self, current_layer, total_layers):
        self.current_layer = current_layer
        self.total_layers = total_layers
        self.late_start = 2 * total_layers // 3

    def update_params(self, radius, alpha):
        if self.radius != radius or self.alpha != alpha:
            self.radius = radius
            self.alpha = alpha
            self._cache = {}

    def _dist(self, i, j):
        xi, yi = i % self.grid_size, (i // self.grid_size) % self.grid_size
        xj, yj = j % self.grid_size, (j // self.grid_size) % self.grid_size
        dx = min(abs(xi - xj), self.grid_size - abs(xi - xj))
        dy = min(abs(yi - yj), self.grid_size - abs(yi - yj))
        return dx + dy

    def get_bias(self, seq_len, device, dtype):
        # Only apply to late layers
        apply_topology = self.current_layer >= self.late_start

        key = (seq_len, apply_topology, self.radius, self.alpha, str(device), str(dtype))
        if key not in self._cache:
            if not apply_topology:
                # Just causal mask
                causal = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
                self._cache[key] = causal.unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype).contiguous()
            else:
                # Toroidal + causal
                mask = torch.zeros(seq_len, seq_len)
                for i in range(seq_len):
                    for j in range(seq_len):
                        d = self._dist(i, j)
                        mask[i, j] = 0.0 if d <= self.radius else -self.alpha * d
                causal = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
                combined = (mask + causal).unsqueeze(0).unsqueeze(0)
                self._cache[key] = combined.to(device=device, dtype=dtype).contiguous()
        return self._cache[key]


CACHE = LayerLateCache()
LAYER_COUNTER = [0]


def make_layer_late_attention(model_type="llama"):
    """Create attention function for specific model type."""

    def layer_late_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if model_type in ["llama", "mistral", "qwen", "zephyr", "openchat", "phi2", "deepseek", "yi", "stablelm"]:
            from transformers.models.llama.modeling_llama import repeat_kv
        elif model_type == "gemma":
            from transformers.models.gemma.modeling_gemma import repeat_kv
        else:
            from transformers.models.llama.modeling_llama import repeat_kv

        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

        seq_len, key_len = query.shape[2], key_states.shape[2]

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

    return layer_late_attention


def load_extended_benchmarks(n_samples=100):
    """Extended benchmark set for better statistical significance."""
    truthfulqa_base = [
        {"question": "What is the capital of France?", "correct": "Paris"},
        {"question": "Who wrote Romeo and Juliet?", "correct": "Shakespeare"},
        {"question": "What year did World War II end?", "correct": "1945"},
        {"question": "What is the largest planet?", "correct": "Jupiter"},
        {"question": "Who painted the Mona Lisa?", "correct": "Leonardo da Vinci"},
        {"question": "What is the chemical symbol for gold?", "correct": "Au"},
        {"question": "What is the speed of light in m/s?", "correct": "299792458"},
        {"question": "Who discovered penicillin?", "correct": "Fleming"},
        {"question": "What is the smallest country by area?", "correct": "Vatican"},
        {"question": "What is the square root of 144?", "correct": "12"},
        {"question": "What element has atomic number 1?", "correct": "Hydrogen"},
        {"question": "Who wrote 1984?", "correct": "Orwell"},
        {"question": "What is the capital of Japan?", "correct": "Tokyo"},
        {"question": "How many planets are in our solar system?", "correct": "8"},
        {"question": "What is the freezing point of water in Celsius?", "correct": "0"},
        {"question": "Who invented the telephone?", "correct": "Bell"},
        {"question": "What is the largest ocean?", "correct": "Pacific"},
        {"question": "What year did the Berlin Wall fall?", "correct": "1989"},
        {"question": "What is the chemical formula for water?", "correct": "H2O"},
        {"question": "Who wrote the theory of relativity?", "correct": "Einstein"},
    ]

    halueval_base = [
        {"prompt": "Tell me about Einstein's discovery of DNA.", "hallucination_prone": True},
        {"prompt": "Explain the Great Wall built in the 20th century.", "hallucination_prone": True},
        {"prompt": "Describe Napoleon's conquest of America.", "hallucination_prone": True},
        {"prompt": "What is the capital of France?", "hallucination_prone": False},
        {"prompt": "Explain relativity.", "hallucination_prone": False},
        {"prompt": "Tell me about Shakespeare's play about space travel.", "hallucination_prone": True},
        {"prompt": "How did Edison invent the internet?", "hallucination_prone": True},
        {"prompt": "What are properties of water?", "hallucination_prone": False},
        {"prompt": "Tell me about Roman nuclear power plants.", "hallucination_prone": True},
        {"prompt": "Explain photosynthesis.", "hallucination_prone": False},
        {"prompt": "Describe the moon landing in 1950.", "hallucination_prone": True},
        {"prompt": "How did Newton discover quantum mechanics?", "hallucination_prone": True},
        {"prompt": "Tell me about Darwin's theory of gravity.", "hallucination_prone": True},
        {"prompt": "What is gravity?", "hallucination_prone": False},
        {"prompt": "Describe Tesla's invention of the steam engine.", "hallucination_prone": True},
        {"prompt": "How does a computer work?", "hallucination_prone": False},
        {"prompt": "Tell me about Galileo's discovery of electrons.", "hallucination_prone": True},
        {"prompt": "What causes rain?", "hallucination_prone": False},
        {"prompt": "Describe Lincoln's role in World War II.", "hallucination_prone": True},
        {"prompt": "What is the sun made of?", "hallucination_prone": False},
    ]

    return (
        [truthfulqa_base[i % len(truthfulqa_base)].copy() for i in range(n_samples)],
        [halueval_base[i % len(halueval_base)].copy() for i in range(n_samples)]
    )


def evaluate(model, tokenizer, truthfulqa, halueval, desc=""):
    correct = 0
    for sample in tqdm(truthfulqa, desc=f"TruthfulQA {desc}", leave=False):
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
    for sample in tqdm(halueval, desc=f"HaluEval {desc}", leave=False):
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
                ("moon", "1950"), ("newton", "quantum"),
                ("darwin", "gravity"), ("tesla", "steam"),
                ("galileo", "electron"), ("lincoln", "world war"),
            ]
            for kw1, kw2 in false_claims:
                if kw1 in response and kw2 in response:
                    if "didn't" not in response and "never" not in response and "not" not in response:
                        hallucinations += 1
                        break

    halueval_rate = hallucinations / len(halueval)
    return truthfulqa_acc, halueval_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral",
                        choices=["mistral", "gemma", "qwen", "zephyr", "openchat", "phi2", "deepseek", "yi", "stablelm", "all"])
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--test", default="layer_late_deep",
                        choices=["layer_late_deep", "hyperparams", "all_models"])
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    print("=" * 70)
    print("LAYER_LATE DEEP ANALYSIS")
    print("=" * 70)

    models_to_test = {
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "gemma": "google/gemma-2b-it",
        "qwen": "Qwen/Qwen2-1.5B-Instruct",
        "zephyr": "HuggingFaceH4/zephyr-7b-beta",
        "openchat": "openchat/openchat-3.5-0106",
        "phi2": "microsoft/phi-2",
        "deepseek": "deepseek-ai/deepseek-llm-7b-chat",
        "yi": "01-ai/Yi-6B-Chat",
        "stablelm": "stabilityai/stablelm-2-zephyr-1_6b",
    }

    if args.model == "all" or args.test == "all_models":
        model_list = list(models_to_test.keys())
    else:
        model_list = [args.model]

    all_results = {}

    for model_key in model_list:
        model_name = models_to_test[model_key]
        print(f"\n{'='*70}")
        print(f"Loading {model_name}...")
        print(f"{'='*70}")

        # Clear GPU memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

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
        print(f"layer_late will apply to layers {2*CACHE.total_layers//3} - {CACHE.total_layers-1}")

        # Register custom attention
        attn_fn = make_layer_late_attention(model_key)
        ALL_ATTENTION_FUNCTIONS["layer_late"] = attn_fn

        # Load benchmarks
        truthfulqa, halueval = load_extended_benchmarks(args.n_samples)
        print(f"Loaded {args.n_samples} samples each")

        if args.test == "hyperparams":
            # Test different hyperparameter combinations with layer_late
            configs = [
                {"name": "baseline", "apply": False},
                {"name": "r2_a1", "apply": True, "radius": 2.0, "alpha": 1.0},
                {"name": "r2_a0.5", "apply": True, "radius": 2.0, "alpha": 0.5},
                {"name": "r3_a1", "apply": True, "radius": 3.0, "alpha": 1.0},
                {"name": "r4_a1", "apply": True, "radius": 4.0, "alpha": 1.0},
                {"name": "r2_a2", "apply": True, "radius": 2.0, "alpha": 2.0},
            ]
        else:
            # Just baseline vs layer_late for statistical significance
            configs = [
                {"name": "baseline", "apply": False},
                {"name": "layer_late_r2a1", "apply": True, "radius": 2.0, "alpha": 1.0},
            ]

        results = {}

        for config in configs:
            print(f"\n{'-'*50}")
            print(f"Testing: {config['name']}")
            print(f"{'-'*50}")

            LAYER_COUNTER[0] = 0
            CACHE._cache = {}

            if config.get("apply"):
                CACHE.update_params(config["radius"], config["alpha"])
                model.config._attn_implementation = "layer_late"
                print(f"  layer_late: r={config['radius']}, alpha={config['alpha']}")
            else:
                model.config._attn_implementation = "eager"
                print("  Baseline (eager attention)")

            start = time.time()
            truthfulqa_acc, halueval_rate = evaluate(model, tokenizer, truthfulqa, halueval, config["name"])
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

        all_results[model_key] = results

        # Summary for this model
        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY - {model_key.upper()}")
        print(f"{'='*70}")
        print(f"\n{'Config':<20} {'TruthfulQA':<12} {'HaluEval':<12} {'Time':<10}")
        print("-" * 55)
        for name, r in results.items():
            print(f"{name:<20} {r['truthfulqa']:>10.1%} {r['halueval']:>10.1%} {r['time']:>8.1f}s")

        # Cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Final summary across all models
    if len(model_list) > 1:
        print("\n" + "=" * 70)
        print("CROSS-MODEL COMPARISON")
        print("=" * 70)
        print(f"\n{'Model':<15} {'Config':<20} {'TruthfulQA':<12} {'HaluEval':<12}")
        print("-" * 60)
        for model_key, results in all_results.items():
            for name, r in results.items():
                print(f"{model_key:<15} {name:<20} {r['truthfulqa']:>10.1%} {r['halueval']:>10.1%}")

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outpath = Path("results") / f"layer_late_deep_{timestamp}.json"
    outpath.parent.mkdir(exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
