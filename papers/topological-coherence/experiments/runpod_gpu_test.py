#!/usr/bin/env python3
"""
Minimal GPU Test for Toroidal Attention
========================================

Run this on RunPod to verify the creative solution works on GPU.

Usage:
    pip install torch transformers
    python runpod_gpu_test.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


def main():
    print("=" * 60)
    print("RUNPOD GPU TEST - Toroidal Attention")
    print("=" * 60)

    # 1. Check CUDA
    print(f"\n1. CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return

    print(f"   GPU: {torch.cuda.get_device_name()}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 2. Load model
    print("\n2. Loading Phi-2...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("   Model loaded successfully!")

    # 3. Test baseline
    print("\n3. Testing BASELINE...")
    prompt = "Einstein's theory of relativity states that"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        start = time.time()
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        baseline_time = time.time() - start

    baseline_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Prompt: {prompt}")
    print(f"   Output: {baseline_response}")
    print(f"   Time: {baseline_time:.3f}s")

    # 4. Set up toroidal attention
    print("\n4. Setting up TOROIDAL attention...")

    # Create toroidal mask cache
    class ToroidalCache:
        def __init__(self, grid_size=12, radius=2.0, alpha=1.0):
            self.grid_size = grid_size
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
            key = (seq_len, str(device), str(dtype))
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

    CACHE = ToroidalCache(grid_size=12, radius=2.0, alpha=1.0)

    # Custom attention function
    def toroidal_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        from transformers.models.phi.modeling_phi import repeat_kv
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

    # Register and apply
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["toroidal"] = toroidal_attention
    model.config._attn_implementation = "toroidal"
    print("   Toroidal attention registered!")

    # 5. Test toroidal
    print("\n5. Testing TOROIDAL...")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        start = time.time()
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        toroidal_time = time.time() - start

    toroidal_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Prompt: {prompt}")
    print(f"   Output: {toroidal_response}")
    print(f"   Time: {toroidal_time:.3f}s")

    # 6. Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nBaseline: {baseline_response}")
    print(f"Toroidal: {toroidal_response}")
    print(f"\nOutputs differ: {baseline_response != toroidal_response}")
    print(f"Baseline time: {baseline_time:.3f}s")
    print(f"Toroidal time: {toroidal_time:.3f}s")

    if baseline_response != toroidal_response:
        print("\n✓ SUCCESS: Toroidal mask is affecting attention on GPU!")
    else:
        print("\n? Same output - may need different prompt or parameters")

    print("\n" + "=" * 60)
    print("GPU TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
