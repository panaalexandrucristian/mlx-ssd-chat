"""Shared process lock and memory monitor for chat and benchmark runs."""
from contextlib import contextmanager
import fcntl
from pathlib import Path

from memory_guard import MemoryGuard


@contextmanager
def guarded_process(limit):
    path = Path.home() / ".local/share/mlx-ssd-chat/ai.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Un chat ai rulează deja. Închide-l înainte de a porni altul.") from None
        guard = MemoryGuard(limit_bytes=limit)
        guard.start()
        try:
            yield guard
        finally:
            guard.close()
