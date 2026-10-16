"""Restore real text/binary envelopes without corrupting terminal transcripts."""
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.restore_export import restore


class ExportRegression(unittest.TestCase):
    def document(self, files):
        sections = []
        manifest = {}
        for name, data in files.items():
            try:
                content = data.decode("utf-8")
                encoding = "utf-8"
            except UnicodeDecodeError:
                content = base64.b64encode(data).decode() + "\n"
                encoding = "base64"
            sections.append(f"### File: `{name}`\n\n````{encoding}\n{content}\n````\n")
            manifest[name] = {"sha256": hashlib.sha256(data).hexdigest(), "mode": 0o644, "encoding": encoding}
        sections.append("### File: `EXPORT-MANIFEST.json`\n\n````utf-8\n" + json.dumps(manifest) + "\n````\n")
        return "\n".join(sections).encode()

    def test_terminal_line_endings_and_binary_round_trip(self):
        files = {"logs/pty.log": b"Tu> hello\r\nAI> hi\r\n\x08\n", "code.py": b"print('ok')",
                 "lib/binary": b"\xff\x00\xfe\r\n"}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "export.md"
            source.write_bytes(self.document(files))
            destination = Path(directory) / "restored"
            restore(source, destination)
            for name, expected in files.items():
                self.assertEqual((destination / name).read_bytes(), expected)

    def test_corruption_does_not_write_partial_export(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "export.md"
            source.write_bytes(self.document({"first.txt": b"valid", "second.txt": b"original"})
                               .replace(b"original", b"tampered"))
            destination = Path(directory) / "restored"
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                restore(source, destination)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
