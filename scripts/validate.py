"""Run offline regressions and, optionally, the real-model checks sequentially."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-model", action="store_true")
    parser.add_argument("--label", default="validation")
    args = parser.parse_args()
    if not args.label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in args.label):
        parser.error("Use a lowercase, filesystem-safe label")
    logs = ROOT / "artifacts/validation" / args.label
    logs.mkdir(parents=True, exist_ok=True)
    commands = [("regression", ["-m", "unittest", "-v", "test_cli", "test_reader", "test_streaming"], 120)]
    if args.real_model:
        commands.extend([
            ("smoke", ["ai.py", "--no-thinking", "--max-tokens", "16", "--seed", "42",
                       "--prompt", "What is 2 + 2? Reply only with the number."], 300),
            ("chat-direct", ["test_chat.py"], 1200),
            ("chat-thinking", ["test_chat.py", "--thinking"], 2400),
            ("context", ["test_context.py"], 1200),
            ("exit", ["test_exit.py"], 300),
        ])
    results = []
    for name, arguments, timeout in commands:
        print(name, flush=True)
        start = time.monotonic()
        with (logs / f"{name}.log").open("w") as log:
            result = subprocess.run([sys.executable, *arguments], cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=timeout)
        results.append({"name": name, "returncode": result.returncode, "seconds": time.monotonic()-start})
        (logs / "summary.json").write_text(json.dumps({"python": sys.executable, "results": results}, indent=2) + "\n")
        if result.returncode:
            sys.exit(f"{name} FAILED; see {logs / (name + '.log')}")
    print(f"All {len(results)} checks passed. Logs: {logs}")


if __name__ == "__main__":
    main()
