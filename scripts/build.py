"""Build pinned sources, install isolated runtimes and record artifact provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from source_inventory import revision, snapshot_flash

ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / "upstream.lock.json").read_text())
UV = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
ARTIFACTS = ROOT / "artifacts"
BUILD_PYTHON = ROOT / ".build-venv/bin/python"
BUILD_LOG = None


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, *, cwd=ROOT, env=None):
    assert BUILD_LOG is not None
    command = [str(item) for item in command]
    heading = "+ " + subprocess.list2cmdline(command)
    print(heading, flush=True)
    with BUILD_LOG.open("a") as log:
        log.write(heading + "\n")
        with subprocess.Popen(command, cwd=cwd, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)


def output(command):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"command": command, "exit_code": result.returncode, "output": result.stdout.strip()}


def environment(path, requirements):
    python = path / "bin/python"
    if not python.exists():
        run([UV, "venv", "--python", LOCK["python"], path])
    check_python(python)
    run([UV, "pip", "sync", "--link-mode", "copy", "--python", python, requirements])
    return python


def check_python(python):
    version = subprocess.check_output([str(python), "-c", "import platform; print(platform.python_version())"], text=True).strip()
    if version != LOCK["python"]:
        raise RuntimeError(f"Expected Python {LOCK['python']}, found {version} in {python.parent.parent}")


def clone_mlx():
    source = ROOT / ".sources/mlx"
    pin = LOCK["mlx"]
    if not source.exists():
        run(["git", "clone", "--branch", pin["tag"], "--depth", "1", pin["url"], source])
        run(["git", "-C", source, "switch", "-c", "local/exact-ssd"])
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    if commit != pin["commit"] or dirty:
        raise RuntimeError("MLX checkout differs from the pinned, clean upstream source")
    return source


def build_environment():
    environment(ROOT / ".build-venv", ROOT / "requirements/build.txt")
    env = os.environ.copy()
    env["PATH"] = str(BUILD_PYTHON.parent) + os.pathsep + env["PATH"]
    return env


def build_wheel(source, destination, env):
    destination.mkdir(parents=True, exist_ok=True)
    run([UV, "build", "--wheel", "--no-build-isolation", "--python", BUILD_PYTHON,
         "--out-dir", destination, source], env=env)
    wheels = list(destination.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one wheel in {destination}; found {len(wheels)}")
    return wheels[0]


def build_flash(env, directory):
    source = directory / "sources/mlx-flash"
    hashes = snapshot_flash(ROOT, source)
    expected = ROOT / "reference/expert_streaming.py"
    if sha256(source / "mlx_flash_compress/expert_streaming.py") != sha256(expected):
        raise RuntimeError("Vendored streaming implementation differs from the verified patch")
    return build_wheel(source, directory / "flash", env), hashes


def write_manifest(name, wheels, details, directory):
    assert BUILD_LOG is not None
    data = {
        "profile": name,
        "build_log": str(BUILD_LOG.relative_to(ROOT)),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_commit": revision(ROOT),
        "upstream": LOCK,
        "runtime_requirements_sha256": sha256(ROOT / "requirements/runtime.txt"),
        "build_requirements_sha256": sha256(ROOT / "requirements/build.txt"),
        "streaming_sha256": sha256(ROOT / "vendor/mlx-flash/mlx_flash_compress/expert_streaming.py"),
        "wheels": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in wheels],
        **details,
    }
    content = json.dumps(data, indent=2) + "\n"
    (ARTIFACTS / f"{name}.json").write_text(content)
    (directory / "manifest.json").write_text(content)


def install(name, wheels):
    path = ROOT / (".venv" if name == "native" else ".venv-flash")
    python = path / "bin/python"
    if not python.exists():
        run([UV, "venv", "--python", LOCK["python"], path])
    check_python(python)
    # Replacing lock entries with wheel paths prevents resolving local packages
    # back to the index while preserving all other exact runtime pins.
    packages = {wheel.name.split("-", 1)[0].replace("_", "-") for wheel in wheels}
    lines = [line for line in (ROOT / "requirements/runtime.txt").read_text().splitlines()
             if line.split("==")[0] not in packages]
    requirements = ARTIFACTS / f"{name}-install.txt"
    requirements.write_text("\n".join(lines + [str(w) for w in wheels]) + "\n")
    run([UV, "pip", "sync", "--link-mode", "copy", "--python", python, requirements])
    # Reinstall local wheels even when their upstream version number is unchanged.
    run([UV, "pip", "install", "--no-deps", "--reinstall", "--link-mode", "copy",
         "--python", python, *wheels])
    run([UV, "pip", "check", "--python", python])
    run([python, ROOT / "scripts/verify_runtime.py", "--profile", name])


def reference():
    python = environment(ROOT / ".venv-reference", ROOT / "requirements/runtime.txt")
    site = subprocess.check_output([str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], text=True).strip()
    module = Path(site) / "mlx_flash_compress/expert_streaming.py"
    original = ROOT / "reference/expert_streaming.upstream-0.9.2.py"
    patched = ROOT / "reference/expert_streaming.py"
    if module.read_bytes() not in (original.read_bytes(), patched.read_bytes()):
        raise RuntimeError("Reference module differs from both verified originals; refusing to overwrite")
    shutil.copy2(patched, module)
    run([UV, "pip", "check", "--python", python])
    run([python, ROOT / "test_streaming.py"])


def main():
    global BUILD_LOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("reference", "flash", "native"))
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("This profile requires Apple Silicon macOS")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "logs").mkdir(exist_ok=True)
    BUILD_LOG = ARTIFACTS / "logs" / f"{args.profile}-{time.time_ns()}.log"
    BUILD_LOG.touch()
    if args.profile == "reference":
        reference()
        return
    if args.profile == "native":
        metal = output(["xcrun", "--sdk", "macosx", "--find", "metal"])
        if metal["exit_code"]:
            (ARTIFACTS / "native-blocker.json").write_text(json.dumps(metal, indent=2) + "\n")
            raise RuntimeError("Metal compiler unavailable. Select a full Xcode installation using DEVELOPER_DIR or xcode-select.")
    env = build_environment()
    directory = ARTIFACTS / "builds" / f"{args.profile}-{time.time_ns()}"
    wheel, hashes = build_flash(env, directory)
    wheels = [wheel]
    details: dict[str, object] = {"flash_source_sha256": hashes}
    if args.profile == "native":
        source = clone_mlx()
        deployment = ".".join(platform.mac_ver()[0].split(".")[:2])
        options = {
            "PYPI_RELEASE": "1",
            "DEBUG": "0",
            "MACOSX_DEPLOYMENT_TARGET": deployment,
            "CMAKE_BUILD_PARALLEL_LEVEL": str(args.jobs),
            "CMAKE_ARGS": "-DMLX_BUILD_METAL=ON -DMLX_BUILD_CPU=ON -DMLX_METAL_JIT=OFF "
                          f"-DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET={deployment}",
        }
        env.update(options)
        toolchain = [output(cmd) for cmd in [
            ["sw_vers"], ["xcode-select", "-p"], ["xcodebuild", "-version"],
            ["xcrun", "--sdk", "macosx", "--show-sdk-version"],
            ["xcrun", "--sdk", "macosx", "metal", "--version"],
            ["clang", "--version"], [str(BUILD_PYTHON.parent / "cmake"), "--version"],
        ]]
        details.update({"build_options": options, "toolchain": toolchain, "native_source_modified": False})
        (ARTIFACTS / "native-toolchain.json").write_text(json.dumps(details, indent=2) + "\n")
        for package in ("frontend", "backend"):
            package_env = {**env, "MLX_BUILD_FRONTEND_PACKAGE": str(int(package == "frontend")),
                           "MLX_BUILD_BACKEND_PACKAGE": str(int(package == "backend"))}
            wheels.append(build_wheel(source, directory / package, package_env))
    write_manifest(args.profile, wheels, details, directory)
    install(args.profile, wheels)
    print(f"Built and verified {args.profile}; artifacts in {ARTIFACTS}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))
