#!/usr/bin/env python3
"""Quick GPU diagnostic for Mistral."""

import torch
import time

print("=" * 60)
print("GPU DIAGNOSTIC")
print("=" * 60)

print(f"\n1. CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n2. Loading Mistral-7B...")
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    'mistralai/Mistral-7B-Instruct-v0.2',
    torch_dtype=torch.float16,
    device_map='auto',
    attn_implementation='eager',
)
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
tokenizer.pad_token = tokenizer.eos_token

# Check where model actually is
param_device = next(model.parameters()).device
print(f"   Model device: {param_device}")

if 'cuda' not in str(param_device):
    print("   ⚠️  WARNING: Model is NOT on GPU!")
else:
    print("   ✓ Model is on GPU")

print("\n3. Testing generation speed...")
prompt = 'The capital of France is'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

# Warmup
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)

# Timed run
start = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
elapsed = time.time() - start

print(f"   Time: {elapsed:.2f}s")
print(f"   Output: {tokenizer.decode(out[0], skip_special_tokens=True)}")

if elapsed > 10:
    print("\n   ⚠️  TOO SLOW - likely running on CPU")
else:
    print("\n   ✓ GPU speed looks good")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
