"""Read source identities from this checkout or a verified standalone export."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def is_repository(root):
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root, text=True, capture_output=True)
    return result.returncode == 0 and Path(result.stdout.strip()).resolve() == root.resolve()


def revision(root):
    if is_repository(root):
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return (root / "EXPORT-REVISION.txt").read_text().strip()


def inventory(root):
    if is_repository(root):
        entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root).decode().split("\0")
        files = {}
        for entry in filter(None, entries):
            metadata, name = entry.split("\t", 1)
            mode = metadata.split()[0]
            if mode not in ("100644", "100755"):
                raise ValueError(f"Unsupported source mode {mode}: {name}")
            files[name] = {"mode": 0o755 if mode == "100755" else 0o644}
        return files
    return json.loads((root / "EXPORT-MANIFEST.json").read_bytes())


def snapshot_flash(root, destination):
    prefix = "vendor/mlx-flash/"
    source = root / "vendor/mlx-flash"
    if is_repository(root):
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--", prefix], cwd=root, text=True)
        if dirty:
            raise RuntimeError("Commit MLX-Flash source changes before building: " + dirty.strip())
    records = inventory(root)
    files = {name: record for name, record in records.items() if name.startswith(prefix)}
    if not files:
        raise RuntimeError("No versioned MLX-Flash source files")
    # Include only recorded inputs; stale build/lib files cannot enter the wheel.
    hashes = {}
    for name, record in files.items():
        path = root / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if "sha256" in record and record["sha256"] != digest:
            raise RuntimeError(f"Exported source changed: {name}")
        hashes[name] = digest
    for name, record in files.items():
        target = destination / (root / name).relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
        target.chmod(record["mode"])
    return hashes
