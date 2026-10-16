"""Real-model boundary test: reject overflow, fill ~4K context, recall, reset."""
from pathlib import Path
import re
import subprocess
import sys

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from ai import MODEL, REVISION


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_download(MODEL, revision=REVISION, local_files_only=True),
        fix_mistral_regex=True)
    base = "The remembered code is ORBIT-73. "
    unit = "The quick brown fox jumps over the lazy dog. "
    suffix = "\nWhat is the remembered code at the beginning? Reply only with that code."

    def tokens(text):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True, return_dict=False,
            add_generation_prompt=True, enable_thinking=False)

    lo, hi = 0, 1000
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if len(tokens(base + unit * mid + suffix)) + 16 <= 4096:
            lo = mid
        else:
            hi = mid
    valid = base + unit * lo + suffix
    invalid = base + unit * hi + suffix
    count = len(tokens(valid))
    assert count > 4000 and len(tokens(invalid)) + 16 > 4096
    print(f"Accepted prompt: {count} tokens + 16 response reserve", flush=True)
    # Interactive input is one line per message.
    lines = [invalid.replace("\n", " "), valid.replace("\n", " "),
             "/reset", "What is 2 + 2? Reply only with the number.", "q"]
    # A newline and a space may tokenize differently: verify the actual inputs.
    assert len(tokens(lines[0])) + 16 > 4096
    assert len(tokens(lines[1])) + 16 <= 4096
    print(f"Actual one-line prompt: {len(tokens(lines[1]))} tokens", flush=True)
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("ai.py")), "--seed", "42",
         "--no-thinking", "--max-tokens", "16"],
        input="\n".join(lines) + "\n", text=True, capture_output=True, timeout=900)
    output = result.stdout + "\n" + result.stderr
    logs = Path(__file__).resolve().parent / "test-logs"
    logs.mkdir(exist_ok=True)
    (logs / "context.log").write_text(output)
    print(output, flush=True)
    assert result.returncode == 0
    assert "Bugetul de context" in result.stderr
    assert "AI> ORBIT-73" in result.stdout
    assert "Conversație resetată" in result.stderr
    assert "AI> 4" in result.stdout
    assert "Limită de răspuns" not in result.stderr
    assert max(map(float, re.findall(r"physical footprint=([\d.]+) GB", output))) < 4
    print("Context boundary, recall and reset: PASS")


if __name__ == "__main__":
    main()
