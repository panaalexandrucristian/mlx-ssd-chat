"""Alternate repeated 2.8/8 GB trials, with fixed and automatically sized caches."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Explain how a hash table handles collisions. Give a detailed explanation with examples."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=64)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("At least two alternating repetitions are required")
    if not args.label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in args.label):
        parser.error("Use a lowercase, filesystem-safe label")
    destination = ROOT / "artifacts/comparisons" / args.label
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for mode in ("fixed", "auto"):
        for repetition in range(-1, args.repeats):
            budgets = (2.8, 8) if repetition % 2 == 0 else (8, 2.8)
            for budget in budgets:
                name = f"{mode}-{budget}-{'warmup' if repetition == -1 else repetition}"
                path = destination / f"{name}.json"
                command = [sys.executable, str(ROOT / "benchmark.py"), "--mlx-memory-gb", str(budget),
                           "--process-memory-gb", "9.2", "--tokens", str(args.tokens), "--seed", "42",
                           "--temperature", "0", "--prompt", PROMPT, "--json", str(path)]
                if mode == "fixed":
                    command.extend(["--capacity", "12"])
                print(name, flush=True)
                with (destination / f"{name}.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
                result = json.loads(path.read_text())
                if result["generated_tokens"] != args.tokens:
                    raise RuntimeError("Trial stopped early; change the common prompt before comparing speed")
                if repetition >= 0:
                    results.append({"trial": name, **result})
    summary = {}
    for mode in ("fixed", "auto"):
        for budget in (2.8, 8):
            group = [r for r in results if r["cache_policy"] == mode and r["mlx_limit_bytes"] == int(budget * 1e9)]
            summary[f"{mode}-{budget}"] = {
                "capacity": group[0]["capacity"],
                "median_ttft_seconds": statistics.median(r["ttft_seconds"] for r in group),
                "median_decode_tps": statistics.median(r["decode_tps"] for r in group),
                "max_footprint_bytes": max(r["footprint_peak_bytes"] for r in group),
            }
    report = {"label": args.label, "python": sys.executable, "trials": results, "summary": summary,
              "all_token_sequences_identical": all(r["token_ids"] == results[0]["token_ids"] for r in results),
              "warmup": "One excluded process per configuration. OS file cache is shared and not flushed."}
    (destination / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
