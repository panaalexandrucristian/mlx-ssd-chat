"""Pre-load validation, exclusivity and process-watchdog integration checks."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


class CLIRegression(unittest.TestCase):
    def run_cli(self, script, *arguments, env=None):
        return subprocess.run([sys.executable, str(ROOT / script), *arguments],
                              capture_output=True, text=True, timeout=20, env=env)

    def test_memory_and_token_arguments_rejected_before_loading(self):
        cases = [["--mlx-memory-gb", x] for x in ("0", "-1", "nan", "inf", "text", "1e100", "1e-20", "18446744073")]
        cases += [["--process-memory-gb", x] for x in ("0", "-1", "nan", "inf", "2.8", "1")]
        for script, token_option in (("ai.py", "--max-tokens"), ("benchmark.py", "--tokens")):
            for arguments in cases + [[token_option, "0"], [token_option, "2049"]]:
                with self.subTest(script=script, arguments=arguments):
                    result = self.run_cli(script, *arguments)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertNotIn("Încarc", result.stderr)
            self.assertEqual(self.run_cli(script, "--help").returncode, 0)

    def test_chat_and_benchmark_share_process_lock(self):
        with tempfile.TemporaryDirectory() as home:
            path = Path(home) / ".local/share/mlx-ssd-chat/ai.lock"
            path.parent.mkdir(parents=True)
            with path.open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for script in ("ai.py", "benchmark.py"):
                    result = self.run_cli(script, env={**os.environ, "HOME": home})
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertIn("Un chat ai rulează deja", result.stderr)

    def test_watchdog_only_terminates_child(self):
        result = subprocess.run([sys.executable, "-c",
            "from memory_guard import MemoryGuard; import time; "
            "g=MemoryGuard(limit_bytes=1); g.start(); time.sleep(5)"],
            cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 78, result.stderr)
        self.assertIn("Memory budget exceeded", result.stderr)


if __name__ == "__main__":
    unittest.main()
