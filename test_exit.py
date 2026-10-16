"""Verify terminal EOF and SIGINT with a real controlling pseudo-terminal."""
import errno
import os
from pathlib import Path
import pty
import select
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent


def check_exit(key, expected, label):
    pid, master = pty.fork()
    if pid == 0:
        os.execv(sys.executable, [sys.executable, str(ROOT / "ai.py"), "--no-thinking", "--max-tokens", "16"])
    transcript = bytearray()
    reaped = False
    try:
        deadline = time.monotonic() + 120
        while b"Tu> " not in transcript:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([master], [], [], remaining)[0]:
                raise TimeoutError("Chat did not become ready")
            transcript.extend(os.read(master, 65536))
        os.write(master, key)
        deadline = time.monotonic() + 30
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Chat did not close")
            if select.select([master], [], [], 0.1)[0]:
                try:
                    data = os.read(master, 65536)
                except OSError as exc:
                    if exc.errno != errno.EIO:
                        raise
                    data = b""
                transcript.extend(data)
            child, status = os.waitpid(pid, os.WNOHANG)
            if child:
                reaped = True
                actual = os.waitstatus_to_exitcode(status)
                assert actual == expected, (label, actual, transcript.decode())
                break
        print(f"{label}: exit {actual}, PASS")
    finally:
        if not reaped:
            os.kill(pid, signal.SIGTERM)
            os.waitpid(pid, 0)
        os.close(master)
        logs = ROOT / "test-logs"
        logs.mkdir(exist_ok=True)
        (logs / f"exit-{label}.log").write_bytes(transcript)


if __name__ == "__main__":
    check_exit(b"\x04", 0, "eof")
    check_exit(b"\x03", 130, "interrupt")
