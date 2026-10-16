"""Verify local wheel contents, GPU execution and loaded native libraries."""
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import sysconfig
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(profile):
    import mlx.core as mx
    import mlx_flash_compress.expert_streaming as streaming

    manifest = json.loads((ROOT / f"artifacts/{profile}.json").read_text())
    site = Path(sysconfig.get_paths()["purelib"])
    if digest(Path(streaming.__file__).read_bytes()) != manifest["streaming_sha256"]:
        raise RuntimeError("Installed streaming module does not match the build")
    verified = []
    for entry in manifest["wheels"]:
        wheel = ROOT / entry["path"]
        if digest(wheel.read_bytes()) != entry["sha256"]:
            raise RuntimeError(f"Wheel changed: {wheel}")
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if name.endswith("/") or ".dist-info/" in name or ".data/" in name:
                    continue
                installed = site / name
                if not installed.is_file() or digest(installed.read_bytes()) != digest(archive.read(name)):
                    raise RuntimeError(f"Installed file differs from local wheel: {name}")
        verified.append(wheel.name)
    if not mx.metal.is_available():
        raise RuntimeError("Metal GPU is unavailable")
    with mx.stream(mx.gpu):
        value = mx.sum(mx.arange(16, dtype=mx.float32)).item()
    if value != 120:
        raise RuntimeError("GPU smoke test failed")
    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    libraries = [dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count())]
    native = [str(Path(p).resolve()) for p in libraries if "libmlx" in Path(p).name]
    if not native or any(not Path(p).is_relative_to(site.resolve()) for p in native):
        raise RuntimeError(f"MLX library loaded outside this environment: {native}")
    report = {
        "profile": profile, "python": sys.version, "interpreter": sys.executable,
        "mlx": importlib.metadata.version("mlx"), "mlx-metal": importlib.metadata.version("mlx-metal"),
        "mlx-flash": importlib.metadata.version("mlx-flash"), "core": mx.__file__,
        "streaming": streaming.__file__, "libraries": native, "gpu_sum": value,
        "verified_wheels": verified,
        "metallibs": [{"path": str(p), "sha256": digest(p.read_bytes())} for p in site.glob("mlx/**/*.metallib")],
    }
    (ROOT / f"artifacts/{profile}-runtime.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("flash", "native"), required=True)
    verify(parser.parse_args().profile)
