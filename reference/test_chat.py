"""Real-model PTY smoke test: conversation memory, KV reuse, reset and exit."""
import argparse
import os
from pathlib import Path
import pty
import re
import select
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    command = [str(Path.home() / ".local/bin/ai"), "--max-tokens", "512" if args.thinking else "64"]
    if not args.thinking:
        command.append("--no-thinking")
    master, slave = pty.openpty()
    process = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave)
    os.close(slave)
    transcript = bytearray()

    def until(marker, timeout=360):
        output = bytearray()
        deadline = time.monotonic() + timeout
        while marker not in output:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([master], [], [], remaining)[0]:
                raise TimeoutError(f"Waiting for {marker!r}")
            data = os.read(master, 65536)
            if not data:
                raise RuntimeError("Chat exited before the expected prompt")
            transcript.extend(data)
            output.extend(data)
        return output.decode()

    def send(query):
        os.write(master, (query + "\n").encode())
        return until(b"Tu> ")

    try:
        until(b"Tu> ", timeout=60)
        first = send("Remember this code: LUNA-47. Reply only OK. Think briefly.")
        assert "Limită de răspuns" not in first, first
        second = send("What code did I ask you to remember? Reply only with the code. Think briefly.")
        answer = second.split("AI> ", 1)[1].split("[", 1)[0]
        assert "LUNA-47" in answer, second
        assert "Limită de răspuns" not in second, second
        prefix = int(re.search(r"prefix reutilizat (\d+) tokeni", second)[1])
        # This pinned template removes the first answer's thinking, so only
        # the 26-token first user turn + assistant header can remain cached.
        assert prefix == 26, second
        if args.thinking:
            assert "</think>" in first and "</think>" in second
        reset = send("/reset")
        assert "Conversație resetată" in reset, reset
        third = send("What is 2 + 2? Reply briefly. Think briefly.")
        assert "prefix reutilizat 0 tokeni" in third, third
        assert "Limită de răspuns" not in third, third
        assert "4" in third.split("AI> ", 1)[1], third
        os.write(master, b"q\n")
        assert process.wait(timeout=30) == 0
        print(transcript.decode(), flush=True)
        print(f"PTY conversation, recall, KV prefix={prefix}, reset, q: PASS")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        os.close(master)
        logs = Path(__file__).resolve().parent / "test-logs"
        logs.mkdir(exist_ok=True)
        (logs / ("chat-thinking.log" if args.thinking else "chat-direct.log")).write_bytes(transcript)


if __name__ == "__main__":
    main()
