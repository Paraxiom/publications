"""
Cross-Architecture Validation: Toroidal Topology Reduces Hallucination
=======================================================================

Validates that the toroidal constraint effect generalizes across architectures:
- Phi-2 (Microsoft, 2.78B) - Primary validation
- TinyLlama-1.1B (Llama architecture) - Cross-architecture validation

Both use the same Tonnetz attention mask injection.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import time
from tqdm import tqdm
import argparse

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# =============================================================================
# TONNETZ TOPOLOGY
# =============================================================================

class TonnetzTopology:
    """Implements the Tonnetz as a toroidal lattice."""

    def __init__(self, grid_size: int = 12):
        self.grid_size = grid_size
        self._cache = {}

    def toroidal_distance(self, i: int, j: int) -> int:
        """Manhattan distance on torus with wraparound."""
        xi, yi = i % self.grid_size, (i // self.grid_size) % self.grid_size
        xj, yj = j % self.grid_size, (j // self.grid_size) % self.grid_size
        dx = min(abs(xi - xj), self.grid_size - abs(xi - xj))
        dy = min(abs(yi - yj), self.grid_size - abs(yi - yj))
        return dx + dy

    def create_attention_mask(
        self,
        seq_len: int,
        radius: float = 2.0,
        alpha: float = 1.0,
        device: torch.device = DEVICE
    ) -> torch.Tensor:
        """Tonnetz attention mask with exponential decay."""
        cache_key = (seq_len, radius, alpha)
        if cache_key in self._cache:
            return self._cache[cache_key].to(device)

        mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(seq_len):
                d = self.toroidal_distance(i, j)
                if d <= radius:
                    mask[i, j] = 1.0
                else:
                    mask[i, j] = np.exp(-alpha * d)

        self._cache[cache_key] = mask
        return mask.to(device)

    @property
    def spectral_gap(self) -> float:
        """Theoretical spectral gap λ₁ = 2 - 2cos(2π/N)."""
        return 2 - 2 * np.cos(2 * np.pi / self.grid_size)


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

MODEL_CONFIGS = {
    "phi-2": {
        "name": "microsoft/phi-2",
        "display_name": "Phi-2 (2.78B)",
        "architecture": "Phi",
        "has_rlhf": False,
    },
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "display_name": "TinyLlama (1.1B)",
        "architecture": "Llama",
        "has_rlhf": True,  # Chat version has some alignment
    },
    "tinyllama-base": {
        "name": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "display_name": "TinyLlama Base (1.1B)",
        "architecture": "Llama",
        "has_rlhf": False,  # Base model, no RLHF
    },
}


# =============================================================================
# BENCHMARKS
# =============================================================================

def load_truthfulqa(max_samples: int = None) -> List[Dict]:
    """Load TruthfulQA benchmark."""
    try:
        from datasets import load_dataset
        print("Loading TruthfulQA...")
        dataset = load_dataset("truthful_qa", "generation", split="validation")
        prompts = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            prompts.append({
                "prompt": item["question"],
                "category": item.get("category", "unknown"),
                "best_answer": item.get("best_answer", ""),
                "correct_answers": item.get("correct_answers", []),
                "incorrect_answers": item.get("incorrect_answers", []),
                "source": "truthfulqa"
            })
        print(f"Loaded {len(prompts)} TruthfulQA samples")
        return prompts
    except Exception as e:
        print(f"Could not load TruthfulQA: {e}")
        return []


def load_halueval(max_samples: int = None) -> List[Dict]:
    """Load HaluEval benchmark."""
    try:
        from datasets import load_dataset
        print("Loading HaluEval...")
        dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
        prompts = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            prompts.append({
                "prompt": item.get("question", ""),
                "knowledge": item.get("knowledge", ""),
                "right_answer": item.get("right_answer", ""),
                "hallucinated_answer": item.get("hallucinated_answer", ""),
                "source": "halueval"
            })
        print(f"Loaded {len(prompts)} HaluEval samples")
        return prompts
    except Exception as e:
        print(f"Could not load HaluEval: {e}")
        return []


# =============================================================================
# EXPERIMENT CLASS
# =============================================================================

class CrossArchitectureExperiment:
    """Run toroidal constraint experiments across different model architectures."""

    def __init__(
        self,
        model_key: str,
        use_toroidal: bool = False,
        toroidal_radius: float = 2.0,
        toroidal_alpha: float = 1.0,
        max_new_tokens: int = 100,
    ):
        self.model_key = model_key
        self.config = MODEL_CONFIGS[model_key]
        self.use_toroidal = use_toroidal
        self.toroidal_radius = toroidal_radius
        self.toroidal_alpha = toroidal_alpha
        self.max_new_tokens = max_new_tokens
        self.topology = TonnetzTopology(grid_size=12)

        self._load_model()

    def _load_model(self):
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"\n{'='*50}")
        print(f"Loading {self.config['display_name']}")
        print(f"{'='*50}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["name"],
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["name"],
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="auto" if DEVICE.type == "cuda" else None,
        )

        if DEVICE.type != "cuda":
            self.model = self.model.to(DEVICE)

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"Parameters: {n_params:.2f}B")
        print(f"Architecture: {self.config['architecture']}")
        print(f"Has RLHF: {'YES' if self.config['has_rlhf'] else 'NO'}")

        if self.use_toroidal:
            self._apply_toroidal_constraints()

        print(f"\nToroidal constraints: {'ENABLED' if self.use_toroidal else 'DISABLED'}")
        if self.use_toroidal:
            print(f"  Radius: {self.toroidal_radius}")
            print(f"  Alpha: {self.toroidal_alpha}")
            print(f"  Theoretical spectral gap: {self.topology.spectral_gap:.4f}")

    def _apply_toroidal_constraints(self):
        """Apply Tonnetz attention mask to all attention layers."""
        print("Applying toroidal attention constraints...")

        n_layers = 0
        for name, module in self.model.named_modules():
            if 'self_attn' in name and hasattr(module, 'q_proj'):
                original_forward = module.forward

                def make_wrapper(orig_fwd, topology, radius, alpha):
                    def wrapper(hidden_states, position_embeddings, attention_mask=None,
                                past_key_values=None, cache_position=None, **kwargs):
                        """Wrapper with Tonnetz mask injection."""
                        seq_len = hidden_states.shape[1]

                        # Create topological mask
                        topo_mask = topology.create_attention_mask(
                            seq_len, radius, alpha,
                            device=hidden_states.device
                        )
                        topo_bias = torch.log(topo_mask + 1e-10)

                        # Add causal mask
                        causal_bias = torch.triu(
                            torch.ones(seq_len, seq_len, device=hidden_states.device) * float('-inf'),
                            diagonal=1
                        )
                        combined_bias = topo_bias + causal_bias

                        # Create 4D attention mask
                        new_mask = combined_bias.unsqueeze(0).unsqueeze(0)

                        # Merge with existing mask
                        if attention_mask is not None:
                            attention_mask = attention_mask + new_mask.to(attention_mask.dtype)
                        else:
                            attention_mask = new_mask

                        return orig_fwd(hidden_states, position_embeddings, attention_mask,
                                       past_key_values, cache_position, **kwargs)

                    return wrapper

                module.forward = make_wrapper(
                    original_forward,
                    self.topology,
                    self.toroidal_radius,
                    self.toroidal_alpha
                )
                n_layers += 1

        print(f"  Wrapped {n_layers} attention modules with toroidal mask")

    def generate(self, prompt: str) -> str:
        """Generate response for a prompt."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def evaluate_truthfulqa(self, prompts: List[Dict]) -> float:
        """Evaluate on TruthfulQA."""
        correct = 0
        total = len(prompts)

        for item in tqdm(prompts, desc="TruthfulQA"):
            response = self.generate(item["prompt"])
            response_lower = response.lower()

            # Check against correct and incorrect answers
            has_correct = any(
                ans.lower() in response_lower
                for ans in item.get("correct_answers", [])
            )
            has_incorrect = any(
                ans.lower() in response_lower
                for ans in item.get("incorrect_answers", [])
            )

            if has_correct and not has_incorrect:
                correct += 1

        return correct / total if total > 0 else 0.0

    def evaluate_halueval(self, prompts: List[Dict]) -> float:
        """Evaluate on HaluEval (hallucination detection)."""
        correct = 0
        total = len(prompts)

        for item in tqdm(prompts, desc="HaluEval"):
            # Generate with knowledge context
            full_prompt = f"Knowledge: {item['knowledge']}\nQuestion: {item['prompt']}\nAnswer:"
            response = self.generate(full_prompt)

            response_lower = response.lower()
            right_answer = item.get("right_answer", "").lower()
            hallucinated = item.get("hallucinated_answer", "").lower()

            # Correct if response contains right answer OR avoids hallucinated answer
            if right_answer and right_answer in response_lower:
                correct += 1
            elif hallucinated and hallucinated not in response_lower:
                correct += 1

        return correct / total if total > 0 else 0.0


# =============================================================================
# MAIN
# =============================================================================

def run_experiment(model_key: str, n_samples: int = 10) -> Dict:
    """Run baseline and toroidal experiments for a model."""
    results = {}

    # Load benchmarks
    truthfulqa = load_truthfulqa(max_samples=n_samples)
    halueval = load_halueval(max_samples=n_samples)

    for condition in ["baseline", "toroidal"]:
        print(f"\n{'#'*60}")
        print(f"# RUNNING: {MODEL_CONFIGS[model_key]['display_name']} ({condition})")
        print(f"{'#'*60}")

        use_toroidal = (condition == "toroidal")
        exp = CrossArchitectureExperiment(
            model_key=model_key,
            use_toroidal=use_toroidal,
        )

        start_time = time.time()

        truthfulqa_acc = exp.evaluate_truthfulqa(truthfulqa) if truthfulqa else 0.0
        halueval_score = exp.evaluate_halueval(halueval) if halueval else 0.0

        elapsed = time.time() - start_time

        results[condition] = {
            "model": model_key,
            "display_name": MODEL_CONFIGS[model_key]["display_name"],
            "architecture": MODEL_CONFIGS[model_key]["architecture"],
            "has_rlhf": MODEL_CONFIGS[model_key]["has_rlhf"],
            "condition": condition,
            "truthfulqa_accuracy": truthfulqa_acc,
            "halueval_score": halueval_score,
            "total_time": elapsed,
        }

        # Clear model from memory
        del exp
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-Architecture Toroidal Validation")
    parser.add_argument("--model", type=str, default="tinyllama-base",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model to test")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples per benchmark")
    parser.add_argument("--all", action="store_true",
                        help="Run all models")
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    all_results = {}

    if args.all:
        models_to_test = ["phi-2", "tinyllama-base"]
    else:
        models_to_test = [args.model]

    for model_key in models_to_test:
        print(f"\n{'='*70}")
        print(f"TESTING: {MODEL_CONFIGS[model_key]['display_name']}")
        print(f"{'='*70}")

        results = run_experiment(model_key, n_samples=args.samples)
        all_results[model_key] = results

    # Print summary
    print("\n" + "="*70)
    print("CROSS-ARCHITECTURE VALIDATION RESULTS")
    print("="*70)
    print(f"\n{'Model':<25} {'Condition':<12} {'TruthfulQA':<12} {'HaluEval':<12}")
    print("-" * 65)

    for model_key, model_results in all_results.items():
        for condition, data in model_results.items():
            print(f"{data['display_name']:<25} {condition:<12} "
                  f"{data['truthfulqa_accuracy']*100:>8.1f}%    "
                  f"{data['halueval_score']*100:>8.1f}%")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"cross_architecture_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
