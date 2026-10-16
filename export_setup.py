"""Export tracked sources and build evidence as one hash-verified Markdown file."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/mlx-ssd-setup.md")
    args = parser.parse_args()
    entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=ROOT).decode().split("\0")
    files = {}
    for entry in filter(None, entries):
        metadata, name = entry.split("\t", 1)
        mode = metadata.split()[0]
        files[name] = ((ROOT / name).read_bytes(), 0o755 if mode == "100755" else 0o644)
    files["EXPORT-REVISION.txt"] = (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT), 0o644)
    for folder in (ROOT / "artifacts", ROOT / "test-logs"):
        if folder.exists():
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix in (".json", ".log"):
                    files[str(path.relative_to(ROOT))] = (path.read_bytes(), 0o644)
    manifest = {}
    sections = ["# MLX SSD chat — reproducible source export\n\n",
                "Generated from the working copies of Git-tracked files; build evidence is included when available.\n\n",
                (ROOT / "README.md").read_text(),
                "\n## Restore this document\n\nSave the `scripts/restore_export.py` block below as `restore_export.py`, then run:\n\n",
                "```sh\npython3 restore_export.py mlx-ssd-setup.md ./mlx-ssd-chat\n```\n\n",
                "All files are verified before writing. Build wheels and model weights are recreated using the pinned commands.\n"]
    for name, (data, mode) in files.items():
        try:
            content = data.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            content = base64.b64encode(data).decode() + "\n"
            encoding = "base64"
        # The envelope always appends one newline; the reader removes that one.
        fence = "`" * max(4, max((len(m[0]) + 1 for m in re.finditer(r"`+", content)), default=4))
        sections.append(f"\n### File: `{name}`\n\n{fence}{encoding}\n{content}\n{fence}\n")
        manifest[name] = {"sha256": hashlib.sha256(data).hexdigest(), "mode": mode, "encoding": encoding}
    sections.append("\n### File: `EXPORT-MANIFEST.json`\n\n````utf-8\n" + json.dumps(manifest, indent=2) + "\n\n````\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(sections))
    print(f"Exported {len(files)} files to {args.output}")


if __name__ == "__main__":
    main()
