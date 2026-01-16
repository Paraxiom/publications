#!/usr/bin/env python3
"""Debug why toroidal wrapper fails on GPU."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.insert(0, '.')
from phi2_definitive_proof import TonnetzTopology, wrap_attention_with_topology

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    'microsoft/phi-2',
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("Testing WITHOUT toroidal wrapper...")
inputs = tokenizer('The capital of France is', return_tensors='pt').to('cuda')
out1 = model.generate(**inputs, max_new_tokens=10, do_sample=False)
print(f"Normal output: {tokenizer.decode(out1[0])}")

print("\nApplying toroidal wrapper...")
wrap_attention_with_topology(model, TonnetzTopology(12), 2.0, 1.0)

print("Testing WITH toroidal wrapper...")
try:
    out2 = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    print(f"Toroidal output: {tokenizer.decode(out2[0])}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\nIf toroidal output is empty or instant, the wrapper is crashing silently.")
