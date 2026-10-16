"""Build inputs must be reproducible from a clean commit or verified export."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.source_inventory import snapshot_flash


class SourceRegression(unittest.TestCase):
    def git(self, root, *arguments):
        return subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)

    def test_dirty_package_is_rejected_and_generated_files_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "vendor/mlx-flash"
            source.mkdir(parents=True)
            module = source / "module.py"
            module.write_text("VALUE = 1\n")
            (root / ".gitignore").write_text("build/\n")
            self.git(root, "init")
            self.git(root, "add", ".")
            self.git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                     "commit", "-m", "Source fixture")
            (source / "build").mkdir()
            (source / "build/stale.py").write_text("stale\n")
            output = root / "first-build"
            hashes = snapshot_flash(root, output)
            self.assertEqual((output / "module.py").read_bytes(), module.read_bytes())
            self.assertFalse((output / "build").exists())
            self.assertEqual(len(hashes), 1)
            module.write_text("VALUE = 2\n")
            with self.assertRaisesRegex(RuntimeError, "Commit MLX-Flash"):
                snapshot_flash(root, root / "second-build")
            self.assertFalse((root / "second-build").exists())
            module.write_text("VALUE = 1\n")
            (source / "uncommitted.py").write_text("NEW = 1\n")
            with self.assertRaisesRegex(RuntimeError, "Commit MLX-Flash"):
                snapshot_flash(root, root / "third-build")


if __name__ == "__main__":
    unittest.main()
