"""Restore a source export after verifying names, inventory and every checksum."""
import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


def restore(document, destination):
    destination = destination.resolve()
    pattern = r"^### File: `([^`]+)`\n\n(?P<fence>`{4,})(utf-8|base64)\n(.*?)\n(?P=fence)$"
    blocks = list(re.finditer(pattern, document.read_bytes().decode("utf-8"), re.MULTILINE | re.DOTALL))
    files = {}
    encodings = {}
    for block in blocks:
        name, _, encoding, content = block.groups()
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or not path.parts or str(path) != name:
            raise ValueError(f"Invalid path: {name}")
        if name in files:
            raise ValueError(f"Duplicate file: {name}")
        files[name] = base64.b64decode(content.strip(), validate=True) if encoding == "base64" else content.encode()
        encodings[name] = encoding
    manifest_bytes = files.pop("EXPORT-MANIFEST.json")
    manifest = json.loads(manifest_bytes)
    if set(files) != set(manifest):
        raise ValueError("Incomplete export inventory")
    for name, data in files.items():
        if encodings[name] != manifest[name]["encoding"] or hashlib.sha256(data).hexdigest() != manifest[name]["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
        if manifest[name]["mode"] not in (0o644, 0o755):
            raise ValueError(f"Invalid file mode: {name}")
    # Persist the source inventory so standalone restores can export again.
    files["EXPORT-MANIFEST.json"] = manifest_bytes
    for name, data in files.items():
        target = destination / name
        if any(p.is_symlink() for p in (target, *target.parents)):
            raise ValueError(f"Symlink in destination: {name}")
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError(f"Refusing to overwrite different file: {target}")
    for name, data in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(manifest[name]["mode"] if name in manifest else 0o644)
    print(f"Verified and restored {len(files)} files to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    restore(args.document, args.destination)
