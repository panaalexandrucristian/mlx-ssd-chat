"""Explicitly download the pinned model; inference itself is offline-only."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

pin = json.loads((Path(__file__).resolve().parents[1] / "upstream.lock.json").read_text())["model"]
print(snapshot_download(pin["id"], revision=pin["revision"], max_workers=2))
