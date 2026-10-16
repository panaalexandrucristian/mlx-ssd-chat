"""Embed complete local sources, configuration and test evidence in the setup doc."""
import difflib
import hashlib
from pathlib import Path
import subprocess
import sys
import sysconfig


ROOT = Path(__file__).resolve().parent
DOCUMENT = Path.home() / "mlx-opencode-setup.md"
MARKER = "\n## 8. Anexă reproductibilă — fișiere complete\n"


def main():
    installed = Path(sysconfig.get_paths()["purelib"]) / "mlx_flash_compress/expert_streaming.py"
    original = ROOT / "expert_streaming.upstream-0.9.2.py"
    patch = "".join(difflib.unified_diff(
        original.read_text().splitlines(keepends=True),
        installed.read_text().splitlines(keepends=True),
        fromfile="a/expert_streaming.py", tofile="b/expert_streaming.py"))
    files = {
        "expert_streaming.py": installed.read_text(),
        original.name: original.read_text(),
        "expert_streaming.before-resume.py": (ROOT / "expert_streaming.before-resume.py").read_text(),
        "expert_streaming.patch": patch,
        "requirements.txt": subprocess.check_output(
            ["uv", "pip", "freeze", "--python", sys.executable], text=True),
        "upstream.sha256": hashlib.sha256(original.read_bytes()).hexdigest()
                           + "  expert_streaming.py\n",
        "ai": (Path.home() / ".local/bin/ai").read_text(),
        "ai.zsh": '# Local MLX chat: Qwen3 30B, exact SSD offload, 4 GB process budget\n'
                  'alias ai=\'"$HOME/.local/bin/ai"\'\n',
    }
    for name in ("ai.py", "memory_guard.py", "cache_memory.py", "benchmark.py", "test_streaming.py",
                 "test_chat.py", "test_context.py", "export_setup.py"):
        files[name] = (ROOT / name).read_text()
    for path in sorted((ROOT / "test-logs").glob("*")):
        if path.is_file() and path.suffix in (".log", ".txt"):
            # PTY logs use CRLF and may contain terminal backspaces. Keep them
            # readable in Markdown; hashes describe the embedded text form.
            content = path.read_text().replace("\b", "\\b")
            files[path.name] = content if content.endswith("\n") else content + "\n"
    bundle = ROOT / "setup-bundle"
    bundle.mkdir(exist_ok=True)
    for name, content in files.items():
        (bundle / name).write_text(content)
    files["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(content.encode()).hexdigest()}  {name}\n"
        for name, content in files.items())
    (bundle / "SHA256SUMS").write_text(files["SHA256SUMS"])
    sections = [MARKER, "\nToate sursele locale, configurația shell, patch-ul, versiunile "
                "fixate și dovezile text ale testelor sunt incluse integral mai jos. "
                "Numele identifică blocurile de extras; conținutul se află în acest document. "
                "Modulul instalat, originalul 0.9.2 și copia istorică sunt complete, "
                "iar generatorul acestei anexe este inclus și el.\n\n"
                "Logurile sunt reprezentate textual: CRLF/CR devin LF, backspace devine "
                "`\\b`, iar blocurile primesc un newline final dacă lipsea. Hash-urile "
                "din `SHA256SUMS` verifică exact forma inclusă în document. "
                "Sursele Python și patch-ul păstrează conținutul instalării verificate.\n\n"
                "Inventar de blocuri complete:\n"]
    sections.extend(f"- `{name}`\n" for name in files)
    for name, content in files.items():
        language = ("diff" if name.endswith(".patch") else "python" if name.endswith(".py")
                    else "zsh" if name == "ai" or name.endswith(".zsh") else "text")
        sections.append(f"\n### Fișier: `{name}`\n\n```{language}\n{content}```\n")
    DOCUMENT.write_text(DOCUMENT.read_text().split(MARKER)[0] + "".join(sections))
    print(f"Exported {len(files)} files to {bundle} and {DOCUMENT}")


if __name__ == "__main__":
    main()
