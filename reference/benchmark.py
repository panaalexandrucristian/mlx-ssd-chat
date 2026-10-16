"""One local model only; bounded-memory end-to-end smoke benchmark."""
import argparse
import time
from memory_guard import MemoryGuard, add_memory_arguments, process_memory_limit

parser = argparse.ArgumentParser()
parser.add_argument("--tokens", type=int, default=64)
parser.add_argument("--prompt", default="What is 2 + 2? Reply briefly.")
parser.add_argument("--thinking", action="store_true")
add_memory_arguments(parser)
args = parser.parse_args()
guard = MemoryGuard(limit_bytes=process_memory_limit(args, parser))
guard.start()

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_flash_compress.expert_streaming import enable_expert_streaming
from cache_memory import plan_expert_cache

mx.set_memory_limit(args.mlx_memory_bytes)
mx.set_cache_limit(64_000_000)
print(f"Memory budgets: MLX {args.mlx_memory_bytes/1e9:g} GB; "
      f"process {guard.limit/1e9:g} GB", flush=True)
path = snapshot_download("mlx-community/Josiefied-Qwen3-30B-A3B-abliterated-v2-4bit",
                         revision="f7dbd27ebbaf60abd0c15111cfde8112ee9b1d1e",
                         local_files_only=True)
start = time.monotonic()
model, tokenizer = load(path, lazy=True)
plan = plan_expert_cache(model, args.mlx_memory_bytes)
print(plan.summary(), flush=True)
state = enable_expert_streaming(model, capacity_per_layer=plan.capacity, model_path=path,
                                exact=True, read_mode="pread")
state.warmup()
mx.eval(model.parameters())
print(f"Load: {time.monotonic()-start:.2f}s; MLX {mx.get_active_memory()/1e9:.2f} GB; "
      f"{guard.summary()}", flush=True)
prompt = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}],
            tokenize=True, add_generation_prompt=True, enable_thinking=args.thinking)
print(f"Prompt: {len(prompt)} tokens", flush=True)
start = time.monotonic()
first = None
try:
    for response in stream_generate(model, tokenizer, prompt, max_tokens=args.tokens,
            sampler=make_sampler(temp=0.6), prefill_step_size=128,
            kv_bits=8, quantized_kv_start=0):
        if first is None:
            first = time.monotonic() - start
        print(response.text, end="", flush=True)
    print(f"\nTTFT={first:.2f}s; total={time.monotonic()-start:.2f}s; "
          f"generated={response.generation_tokens}; decode={response.generation_tps:.2f} tok/s; "
          f"MLX peak={mx.get_peak_memory()/1e9:.2f} GB", flush=True)
finally:
    state.cleanup()
    guard.close()
    print(guard.summary(), flush=True)
