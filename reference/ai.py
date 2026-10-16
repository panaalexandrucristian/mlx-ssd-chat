"""On-demand terminal chat: Qwen3-MoE, exact SSD streaming, configurable memory."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

from memory_guard import MemoryGuard, add_memory_arguments, process_memory_limit

MODEL = "mlx-community/Josiefied-Qwen3-30B-A3B-abliterated-v2-4bit"
REVISION = "f7dbd27ebbaf60abd0c15111cfde8112ee9b1d1e"


def common_prefix(a, b):
    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", help="One message, then exit")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--no-thinking", action="store_true",
                        help="Optional direct-answer mode (thinking is on by default)")
    add_memory_arguments(parser)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 2048:
        parser.error("--max-tokens must be between 1 and 2048")
    process_limit = process_memory_limit(args, parser)

    lock_path = Path.home() / ".local/share/mlx-ssd-chat/ai.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Un chat ai rulează deja. Închide-l înainte de a porni altul.", file=sys.stderr)
            return 1
        guard = MemoryGuard(limit_bytes=process_limit)
        guard.start()
        try:
            return chat(args, guard)
        except KeyboardInterrupt:
            print("\nChat închis.", file=sys.stderr)
            return 130
        except (ValueError, RuntimeError, OSError) as exc:
            print(f"\nai: {exc}", file=sys.stderr)
            return 1
        finally:
            guard.close()
            print(guard.summary(), file=sys.stderr)


def chat(args, guard):
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    from mlx_flash_compress.expert_streaming import enable_expert_streaming
    from cache_memory import plan_expert_cache

    mx.set_memory_limit(args.mlx_memory_bytes)
    mx.set_cache_limit(64_000_000)
    path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    print(f"Încarc Qwen3 30B · SSD offload exact · MLX {args.mlx_memory_bytes/1e9:g} GB · "
          f"buget proces {guard.limit/1e9:g} GB…", file=sys.stderr)
    start = time.monotonic()
    model, tokenizer = load(path, lazy=True, tokenizer_config={"fix_mistral_regex": True})
    plan = plan_expert_cache(model, args.mlx_memory_bytes)
    print(plan.summary(), file=sys.stderr)
    state = enable_expert_streaming(model, capacity_per_layer=plan.capacity, model_path=path,
                                     exact=True, read_mode="pread")
    try:
        state.warmup()
        mx.eval(model.parameters())
        print(f"MLX active after loading: {mx.get_active_memory()/1e9:.2f} GB", file=sys.stderr)
        print(f"Gata în {time.monotonic()-start:.1f}s. Thinking: "
              f"{'oprit' if args.no_thinking else 'pornit'}. "
              "q / Ctrl+D: ieșire; /reset: conversație nouă; Ctrl+C: închide.",
              file=sys.stderr)
        messages = []
        cache = make_prompt_cache(model)
        cached_tokens = []
        context_limit = 4096
        while True:
            if args.prompt is not None:
                query = args.prompt
            else:
                try:
                    query = input("\nTu> ")
                except EOFError:
                    break
            if not query.strip():
                if args.prompt is not None:
                    break
                continue
            if query.strip() in ("q", "/quit"):
                break
            if query.strip() in ("r", "/reset"):
                messages.clear()
                cached_tokens.clear()
                cache = make_prompt_cache(model)
                mx.clear_cache()
                print("Conversație resetată.", file=sys.stderr)
                if args.prompt is not None:
                    break
                continue

            proposed = messages + [{"role": "user", "content": query}]
            prompt = tokenizer.apply_chat_template(proposed, tokenize=True,
                add_generation_prompt=True, enable_thinking=not args.no_thinking)
            if len(prompt) + args.max_tokens > context_limit:
                print(f"Bugetul de context pentru acest profil este {context_limit} tokeni, "
                      "inclusiv răspunsul. Folosește /reset sau un mesaj mai scurt.", file=sys.stderr)
                if args.prompt is not None:
                    return 2
                continue

            # Chat templates may remove old thinking. Reuse ONLY an exact prefix.
            offset = cache[0].offset
            if offset != len(cached_tokens):
                cache = make_prompt_cache(model)
                cached_tokens = []
                offset = 0
            prefix = min(common_prefix(cached_tokens, prompt), len(prompt) - 1)
            if trim_prompt_cache(cache, offset - prefix) != offset - prefix:
                cache = make_prompt_cache(model)
                prefix = 0
            messages = proposed
            generated = []
            text = []
            start = time.monotonic()
            first = None
            print("\nAI> ", end="", flush=True)
            for response in stream_generate(
                model, tokenizer, prompt[prefix:], max_tokens=args.max_tokens,
                sampler=make_sampler(temp=0.6, top_p=0.95), prompt_cache=cache,
                prefill_step_size=128, kv_bits=8, quantized_kv_start=0,
            ):
                if first is None:
                    first = time.monotonic() - start
                print(response.text, end="", flush=True)
                generated.append(response.token)
                text.append(response.text)
            print(flush=True)
            cached_tokens = prompt + generated
            messages.append({"role": "assistant", "content": "".join(text)})
            guard.sample()
            print(f"[{response.generation_tokens} tokeni; primul token {first:.1f}s; "
                  f"decode {response.generation_tps:.2f} tok/s; "
                  f"prefix reutilizat {prefix} tokeni; {guard.summary()}]", file=sys.stderr)
            if response.finish_reason == "length":
                print("[Limită de răspuns atinsă; poate include thinking neterminat.]", file=sys.stderr)
            if args.prompt is not None:
                break
        return 0
    finally:
        state.cleanup()


if __name__ == "__main__":
    sys.exit(main())
