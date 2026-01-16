#!/usr/bin/env python3
"""Debug why toroidal wrapper fails on GPU."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("ERROR: CUDA not available. Reinstall torch with CUDA support.")
    exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nLoading model...")
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

# Now apply toroidal wrapper manually
print("\nApplying toroidal wrapper...")

class TonnetzTopology:
    def __init__(self, grid_size=12):
        self.grid_size = grid_size

    def create_attention_mask(self, seq_len, radius, alpha, device):
        mask = torch.ones(seq_len, seq_len, device=device)
        for i in range(seq_len):
            for j in range(seq_len):
                xi, yi = i % self.grid_size, (i // self.grid_size) % self.grid_size
                xj, yj = j % self.grid_size, (j // self.grid_size) % self.grid_size
                dx = min(abs(xi - xj), self.grid_size - abs(xi - xj))
                dy = min(abs(yi - yj), self.grid_size - abs(yi - yj))
                dist = dx + dy
                if dist <= radius:
                    mask[i, j] = 1.0
                else:
                    mask[i, j] = torch.exp(torch.tensor(-alpha * dist))
        return mask

topology = TonnetzTopology(12)
radius, alpha = 2.0, 1.0
wrapped_count = 0

for name, module in model.named_modules():
    if 'self_attn' in name and hasattr(module, 'forward'):
        original_forward = module.forward

        def make_wrapper(orig_fwd):
            # Flexible wrapper that handles both positional and keyword args
            def wrapper(*args, **kwargs):
                # Extract hidden_states (always first arg or kwarg)
                if args:
                    hidden_states = args[0]
                else:
                    hidden_states = kwargs.get('hidden_states')

                seq_len = hidden_states.shape[1]
                device = hidden_states.device
                dtype = hidden_states.dtype

                # Create toroidal mask
                topo_mask = topology.create_attention_mask(seq_len, radius, alpha, device)
                topo_bias = torch.log(topo_mask + 1e-10).to(dtype)
                causal = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype), diagonal=1)
                combined = (topo_bias + causal).unsqueeze(0).unsqueeze(0)

                # Modify attention_mask in kwargs
                if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                    kwargs['attention_mask'] = kwargs['attention_mask'] + combined
                else:
                    kwargs['attention_mask'] = combined

                return orig_fwd(*args, **kwargs)
            return wrapper

        module.forward = make_wrapper(original_forward)
        wrapped_count += 1

print(f"Wrapped {wrapped_count} attention modules")

print("\nTesting WITH toroidal wrapper...")
try:
    out2 = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    print(f"Toroidal output: {tokenizer.decode(out2[0])}")
except Exception as e:
    import traceback
    print(f"ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\nDone.")
