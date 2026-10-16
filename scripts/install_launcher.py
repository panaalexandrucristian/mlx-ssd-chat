"""Install ai against an explicitly selected, verified runtime."""
import argparse
from datetime import datetime
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("native", "flash", "reference"), default="native")
    parser.add_argument("--shell", action="store_true", help="Add the ai alias to ~/.zshrc")
    args = parser.parse_args()
    environment = ".venv" if args.profile == "native" else f".venv-{args.profile}"
    python = ROOT / environment / "bin/python"
    if not python.exists():
        parser.error(f"Missing runtime: {python}; run scripts/build.py {args.profile}")
    if args.profile != "reference":
        subprocess.run([str(python), str(ROOT / "scripts/verify_runtime.py"), "--profile", args.profile], check=True)
    subprocess.run([str(python), str(ROOT / "ai.py"), "--help"], check=True)
    launcher = Path.home() / ".local/bin/ai"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    content = f"#!/bin/zsh\nexec {shlex.quote(str(python))} {shlex.quote(str(ROOT / 'ai.py'))} \"$@\"\n"
    if launcher.exists() and launcher.read_text() != content:
        shutil.copy2(launcher, launcher.with_name(f"ai.before-{stamp}"))
    temporary = launcher.with_name("ai.installing")
    temporary.write_text(content)
    temporary.chmod(0o755)
    temporary.replace(launcher)
    if args.shell:
        rc = Path.home() / ".zshrc"
        current = rc.read_text() if rc.exists() else ""
        alias = 'alias ai=\'"$HOME/.local/bin/ai"\''
        if alias not in current.splitlines():
            if rc.exists():
                shutil.copy2(rc, rc.with_name(f".zshrc.before-mlx-{stamp}"))
            rc.write_text(current + ("\n" if current and not current.endswith("\n") else "")
                          + "\n# Local MLX SSD chat\n" + alias + "\n")
        subprocess.run(["zsh", "-n", str(rc)], check=True)
    subprocess.run(["zsh", "-n", str(launcher)], check=True)
    print(f"Installed {launcher} -> {environment} ({args.profile})")


if __name__ == "__main__":
    main()
