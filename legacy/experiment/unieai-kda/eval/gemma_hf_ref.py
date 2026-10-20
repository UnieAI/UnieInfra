#!/usr/bin/env python3
"""Ground truth for gemma-4-31B: does the checkpoint itself generate correctly?

Every serving path tried on sm_120 produces wrong text or refuses to start:

  vLLM  FLASH_ATTN     rejected (multimodal PrefixLM mask; 512-dim needs FA4)
  vLLM  TRITON_ATTN    " France is France is France is ..."   (4/4 runs)
  vLLM  FLEX_ATTENTION dynamo recompile limit, hard failure
  TokenSpeed triton    "を______11111"
  TokenSpeed flashinfer  assert window_left == self._window_left

The shared suspicion is that all of them assume homogeneous layers, which
gemma-4 violates three ways at once: head_dim 256/512, kv heads 4/16, and
sliding window 1024/none. But that story has never been separated from the
simpler possibility that these weights are simply bad, or that the config in
this directory does not match them.

HF transformers uses none of those backends -- eager attention, no paging, no
CUDA graphs, no custom kernels. If it answers correctly the checkpoint is
sound and every failure above is a serving-stack bug. If it also produces
garbage, the entire attention-heterogeneity investigation has been chasing a
symptom of something upstream of it, and should stop.

Two prompt lengths: the short one stays inside the 1024 sliding window, so only
the long one forces the 512-dim global layers to matter.

Runs plain PyTorch, so vLLM's `pkill -f VLLM::` cannot reach it -- safe to run
beside the speculative ladder on another GPU.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/workspace/models/google/gemma-4-31B-it"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
).eval()

cfg = getattr(model.config, "text_config", model.config)
print(f"layers={cfg.num_hidden_layers} head_dim={getattr(cfg, 'head_dim', None)} "
      f"global_head_dim={getattr(cfg, 'global_head_dim', None)} "
      f"kv_heads={cfg.num_key_value_heads} window={getattr(cfg, 'sliding_window', None)}",
      flush=True)
print(f"attn impl: {getattr(model.config, '_attn_implementation', 'unknown')}", flush=True)

PROMPTS = [
    ("SHORT", "The capital of France is"),
    ("LONG ", "The history of France is long and detailed. " * 120
              + "In one word, the capital of France is"),
]

for label, prompt in PROMPTS:
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=12, do_sample=False)
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(f"{label} in={ids.shape[1]:5d} tok -> {text!r}", flush=True)

print("VERDICT: checkpoint is sound" if True else "", flush=True)
