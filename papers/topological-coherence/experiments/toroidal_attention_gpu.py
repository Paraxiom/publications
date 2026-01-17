#!/usr/bin/env python3
"""
Creative Solution: Register Custom Toroidal Attention Function
==============================================================

Instead of trying to pass masks through the parameter chain (which breaks on GPU),
we REPLACE the attention computation function itself with one that includes
the toroidal topology built-in.

This approach:
1. Works with any attention backend (eager, SDPA, Flash)
2. Bypasses all mask format/dtype issues
3. Computes the toroidal bias directly in the attention computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from functools import lru_cache


# =============================================================================
# TOROIDAL MASK CACHE (Precomputed for efficiency)
# =============================================================================

class ToroidalMaskCache:
    """
    Precomputes and caches toroidal masks for different sequence lengths.
    The mask is computed once per (seq_len, radius, alpha) combination.
    """

    def __init__(self, grid_size: int = 12, radius: float = 2.0, alpha: float = 1.0):
        self.grid_size = grid_size
        self.radius = radius
        self.alpha = alpha
        self._cache = {}

    def _toroidal_distance(self, i: int, j: int) -> int:
        """Manhattan distance on torus with wraparound."""
        xi, yi = i % self.grid_size, (i // self.grid_size) % self.grid_size
        xj, yj = j % self.grid_size, (j // self.grid_size) % self.grid_size
        dx = min(abs(xi - xj), self.grid_size - abs(xi - xj))
        dy = min(abs(yi - yj), self.grid_size - abs(yi - yj))
        return dx + dy

    def get_bias(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Get toroidal attention bias for given sequence length.
        Returns shape [1, 1, seq_len, seq_len] ready for broadcasting.
        """
        cache_key = (seq_len, str(device), str(dtype))

        if cache_key not in self._cache:
            # Compute base mask on CPU (faster for the loop)
            mask = torch.zeros(seq_len, seq_len)
            for i in range(seq_len):
                for j in range(seq_len):
                    d = self._toroidal_distance(i, j)
                    if d <= self.radius:
                        mask[i, j] = 0.0  # No penalty for nearby positions
                    else:
                        # Log-space: negative values = attenuation
                        mask[i, j] = -self.alpha * d

            # Add causal mask (upper triangle = -inf)
            causal = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
            combined = mask + causal

            # Expand to [1, 1, seq_len, seq_len] for broadcasting over batch and heads
            combined = combined.unsqueeze(0).unsqueeze(0)

            # Move to device and dtype
            self._cache[cache_key] = combined.to(device=device, dtype=dtype).contiguous()

        return self._cache[cache_key]


# Global cache instance
TOROIDAL_CACHE = None


def get_toroidal_cache(grid_size: int = 12, radius: float = 2.0, alpha: float = 1.0):
    """Get or create the global toroidal cache."""
    global TOROIDAL_CACHE
    if TOROIDAL_CACHE is None or TOROIDAL_CACHE.grid_size != grid_size:
        TOROIDAL_CACHE = ToroidalMaskCache(grid_size, radius, alpha)
    return TOROIDAL_CACHE


# =============================================================================
# CUSTOM ATTENTION FUNCTIONS
# =============================================================================

def toroidal_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """
    Custom eager attention with toroidal topology built-in.

    This replaces the standard eager_attention_forward but adds
    the Tonnetz mask directly to attention scores.
    """
    from transformers.models.phi.modeling_phi import repeat_kv

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # Standard attention score computation
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    # Get sequence length from query
    seq_len = query.shape[2]
    key_len = key_states.shape[2]

    # Get toroidal bias (this is the magic!)
    cache = get_toroidal_cache()
    toroidal_bias = cache.get_bias(seq_len, query.device, query.dtype)

    # Handle the case where key_len != seq_len (during generation with cache)
    if key_len != seq_len:
        # Only use the relevant portion of toroidal bias
        # For autoregressive generation, we want the last row
        toroidal_bias = toroidal_bias[:, :, -seq_len:, :key_len]

    # Add toroidal bias to attention weights
    attn_weights = attn_weights + toroidal_bias

    # Also add original attention mask if present (handles padding, etc.)
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :key_len]
        attn_weights = attn_weights + causal_mask

    # Softmax and dropout
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

    # Compute output
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def toroidal_sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """
    Custom SDPA attention with toroidal topology.

    Uses torch.nn.functional.scaled_dot_product_attention but with
    our toroidal bias incorporated.
    """
    from transformers.models.phi.modeling_phi import repeat_kv

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    seq_len = query.shape[2]
    key_len = key_states.shape[2]

    # Get toroidal bias
    cache = get_toroidal_cache()
    toroidal_bias = cache.get_bias(seq_len, query.device, query.dtype)

    # Combine with original attention mask
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :key_len]
        combined_mask = toroidal_bias[:, :, -seq_len:, :key_len] + causal_mask
    else:
        combined_mask = toroidal_bias[:, :, -seq_len:, :key_len]

    # Use scaled_dot_product_attention with our combined mask
    # Note: SDPA expects the mask to be broadcastable
    attn_output = F.scaled_dot_product_attention(
        query,
        key_states,
        value_states,
        attn_mask=combined_mask,
        dropout_p=dropout if module.training else 0.0,
        scale=scaling,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None  # SDPA doesn't return weights


# =============================================================================
# MODEL PATCHING
# =============================================================================

def apply_toroidal_topology(
    model,
    grid_size: int = 12,
    radius: float = 2.0,
    alpha: float = 1.0,
    use_sdpa: bool = False
):
    """
    Apply toroidal topology to a model by replacing its attention function.

    This is the creative solution: instead of trying to inject masks through
    parameters, we replace the attention computation function itself.

    Args:
        model: The HuggingFace model
        grid_size: Tonnetz grid size (default 12 for music theory)
        radius: Neighborhood radius for full attention
        alpha: Decay rate for distant positions
        use_sdpa: Use SDPA backend (faster but may have issues)

    Returns:
        The modified model (modified in-place)
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    # Initialize the global cache with our parameters
    global TOROIDAL_CACHE
    TOROIDAL_CACHE = ToroidalMaskCache(grid_size, radius, alpha)

    # Choose attention function based on backend
    if use_sdpa:
        attn_fn = toroidal_sdpa_attention_forward
        print(f"Applying toroidal topology with SDPA backend")
    else:
        attn_fn = toroidal_eager_attention_forward
        print(f"Applying toroidal topology with eager backend")

    # Register our custom attention function
    ALL_ATTENTION_FUNCTIONS["toroidal"] = attn_fn

    # Set the model to use our custom attention
    model.config._attn_implementation = "toroidal"

    print(f"  Grid size: {grid_size}")
    print(f"  Radius: {radius}")
    print(f"  Alpha: {alpha}")
    print(f"  Theoretical spectral gap: {2 - 2 * np.cos(2 * np.pi / grid_size):.4f}")

    return model


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import time
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("TOROIDAL ATTENTION GPU TEST")
    print("=" * 60)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Load model
    print("\nLoading Phi-2...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    if device.type != "cuda":
        model = model.to(device)

    # Test prompts
    prompts = [
        "The capital of France is",
        "Einstein's theory of relativity states that",
        "The largest planet in our solar system is",
    ]

    # Test WITHOUT toroidal
    print("\n" + "=" * 60)
    print("BASELINE (No toroidal)")
    print("=" * 60)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            start = time.time()
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            elapsed = time.time() - start
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Output: {response}")
        print(f"Time: {elapsed:.3f}s")

    # Apply toroidal topology
    print("\n" + "=" * 60)
    print("APPLYING TOROIDAL TOPOLOGY")
    print("=" * 60)

    apply_toroidal_topology(model, grid_size=12, radius=2.0, alpha=1.0, use_sdpa=False)

    # Test WITH toroidal
    print("\n" + "=" * 60)
    print("TOROIDAL (With topology)")
    print("=" * 60)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            start = time.time()
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            elapsed = time.time() - start
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Output: {response}")
        print(f"Time: {elapsed:.3f}s")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nIf outputs differ between baseline and toroidal, the mask is working!")
