#!/usr/bin/env python3
"""Test if attention mask actually affects GPU output."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    'microsoft/phi-2',
    torch_dtype=torch.float16,
    device_map='auto',
    attn_implementation='eager'
)
tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2')

prompt = 'The capital of France is'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

print(f"\nPrompt: {prompt}")

# Normal generation
out1 = model.generate(**inputs, max_new_tokens=10, do_sample=False)
print(f"Normal output: {tokenizer.decode(out1[0], skip_special_tokens=True)}")

print("\nIf mask injection works, wrapping attention should change output.")
print("The hyperparameter search showed NO change = mask not working on GPU.")
