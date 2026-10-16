"""Verify and extract the self-contained setup document without executing it."""
import argparse
import hashlib
from pathlib import Path
import re


def extract(document, destination):
    blocks = re.findall(r"^### Fișier: `([^`]+)`\n\n```[^\n]*\n(.*?)^```$",
                        document.read_text(encoding="utf-8"), re.MULTILINE | re.DOTALL)
    files = dict(blocks)
    if not blocks or len(files) != len(blocks):
        raise ValueError("Missing appendix or duplicate file names")
    for name in files:
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Invalid file name: {name}")
    checksums = {}
    for line in files["SHA256SUMS"].splitlines():
        expected, name = line.split("  ", 1)
        if name in checksums:
            raise ValueError(f"Duplicate checksum: {name}")
        checksums[name] = expected
    if set(checksums) != set(files) - {"SHA256SUMS"}:
        raise ValueError("Incomplete checksum inventory")
    for name, expected in checksums.items():
        if hashlib.sha256(files[name].encode("utf-8")).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        target = destination / name
        data = content.encode("utf-8")
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError(f"Refusing to overwrite different file: {target}")
    for name, content in files.items():
        (destination / name).write_bytes(content.encode("utf-8"))
    print(f"Verified and extracted {len(files)} files to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    extract(args.document, args.destination)
