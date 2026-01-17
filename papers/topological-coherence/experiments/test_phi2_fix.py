#!/usr/bin/env python3
"""Test Phi-2 with different settings to fix !!! issue."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading Phi-2...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-2",
    trust_remote_code=True,
    use_fast=False,  # Try slow tokenizer
)

print(f"Vocab size: {tokenizer.vocab_size}")
print(f"EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
print(f"Pad token: {tokenizer.pad_token}")

# Set pad token
tokenizer.pad_token = tokenizer.eos_token

prompt = "The capital of France is"
print(f"\nPrompt: {prompt}")

# Encode
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
print(f"Input IDs: {input_ids}")

# Generate with explicit settings
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=20,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

print(f"Output IDs: {output}")
print(f"Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")

# Try with temperature
print("\nWith temperature=0.7:")
with torch.no_grad():
    output2 = model.generate(
        input_ids,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
print(f"Output: {tokenizer.decode(output2[0], skip_special_tokens=True)}")
