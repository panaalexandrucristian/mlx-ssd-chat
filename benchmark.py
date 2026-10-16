"""Seeded, bounded-memory benchmark; optional fixed cache isolates allocator effects."""
import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, cast

from ai import MODEL, REVISION
from memory_guard import add_memory_arguments, process_memory_limit
from runtime import guarded_process


def benchmark(args, guard):
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_flash_compress.expert_streaming import enable_expert_streaming
    from cache_memory import plan_expert_cache

    mx.set_memory_limit(args.mlx_memory_bytes)
    mx.set_cache_limit(64_000_000)
    mx.random.seed(args.seed)
    path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    start = time.monotonic()
    model, tokenizer = cast(tuple[Any, Any], load(
        path, lazy=True, tokenizer_config={"fix_mistral_regex": True}))
    plan = plan_expert_cache(model, args.mlx_memory_bytes)
    capacity = args.capacity if args.capacity is not None else plan.capacity
    minimum = plan.top_k
    if not minimum <= capacity <= plan.capacity:
        raise ValueError(f"--capacity must be between top_k={minimum} and budget capacity={plan.capacity}")
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}],
                tokenize=True, add_generation_prompt=True, enable_thinking=args.thinking)
    if len(prompt) + args.tokens > 4096:
        raise ValueError("Prompt plus response reserve exceeds 4096 tokens")
    state = enable_expert_streaming(model, capacity_per_layer=capacity, model_path=path,
                                   exact=True, read_mode="pread")
    try:
        state.warmup()
        mx.eval(model.parameters())
        mx.synchronize()
        load_seconds = time.monotonic() - start
        loaded_bytes = mx.get_active_memory()
        print(f"MLX {args.mlx_memory_bytes/1e9:g} GB; process {guard.limit/1e9:g} GB; "
              f"capacity {capacity}/{plan.num_experts}; loaded {loaded_bytes/1e9:.2f} GB", flush=True)
        start = time.monotonic()
        first = None
        response = None
        text = []
        tokens = []
        for response in stream_generate(model, tokenizer, prompt, max_tokens=args.tokens,
                sampler=make_sampler(temp=args.temperature, top_p=0.95), prefill_step_size=128,
                kv_bits=8, quantized_kv_start=0):
            if first is None:
                first = time.monotonic() - start
            text.append(response.text)
            tokens.append(response.token)
            print(response.text, end="", flush=True)
        if response is None:
            raise RuntimeError("Generation returned no response")
        mx.synchronize()
        guard.sample()
        result = {
            "model": MODEL, "revision": REVISION, "prompt": args.prompt,
            "seed": args.seed, "thinking": args.thinking, "temperature": args.temperature,
            "max_tokens": args.tokens, "prompt_tokens": len(prompt),
            "generated_tokens": response.generation_tokens, "token_ids": tokens,
            "finish_reason": response.finish_reason, "text": "".join(text),
            "mlx_limit_bytes": args.mlx_memory_bytes, "process_limit_bytes": guard.limit,
            "allocator_cache_limit_bytes": 64_000_000, "capacity": capacity,
            "cache_policy": "fixed" if args.capacity is not None else "auto",
            "load_seconds": load_seconds, "ttft_seconds": first,
            "generation_seconds": time.monotonic() - start,
            "decode_tps": response.generation_tps, "prompt_tps": response.prompt_tps,
            "mlx_loaded_bytes": loaded_bytes, "mlx_active_bytes": mx.get_active_memory(),
            "mlx_cache_bytes": mx.get_cache_memory(), "mlx_peak_bytes": mx.get_peak_memory(),
            "rss_peak_bytes": guard.peak_rss, "footprint_peak_bytes": guard.peak_footprint,
        }
        print("\n" + json.dumps(result, ensure_ascii=False), flush=True)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result
    finally:
        state.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--prompt", default="What is 2 + 2? Reply briefly.")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, choices=(0.0, 0.6), default=0.0)
    parser.add_argument("--capacity", type=int, help="Fixed expert cache for controlled experiments only")
    parser.add_argument("--json", type=Path, help="Write machine-readable measurements")
    add_memory_arguments(parser)
    args = parser.parse_args()
    if not 1 <= args.tokens <= 2048:
        parser.error("--tokens must be between 1 and 2048")
    if args.capacity is not None and args.capacity < 1:
        parser.error("--capacity must be positive")
    limit = process_memory_limit(args, parser)
    try:
        with guarded_process(limit) as guard:
            benchmark(args, guard)
        return 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"benchmark: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
