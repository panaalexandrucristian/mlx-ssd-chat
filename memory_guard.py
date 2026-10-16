"""macOS process memory sampling, including Metal's physical footprint."""
import argparse
import ctypes
import math
import os
import threading


def memory_bytes(value):
    """Parse decimal GB into a positive byte count accepted by MLX's size_t API."""
    try:
        size = float(value) * 1_000_000_000
    except ValueError:
        raise argparse.ArgumentTypeError("memory must be a number in decimal GB") from None
    if not math.isfinite(size) or not 1 <= size < 2**64:
        raise argparse.ArgumentTypeError("memory must be finite, positive and fit in 64-bit bytes")
    return int(size)


def add_memory_arguments(parser):
    parser.add_argument("--mlx-memory-gb", type=memory_bytes, default="2.8",
                        dest="mlx_memory_bytes", metavar="GB",
                         help="MLX memory guideline and automatic expert cache budget in decimal GB (default: 2.8)")
    parser.add_argument("--process-memory-gb", type=memory_bytes,
                        dest="process_memory_bytes", metavar="GB",
                        help="Process watchdog budget in decimal GB (default: MLX + 1.2)")


def process_memory_limit(args, parser):
    limit = args.process_memory_bytes
    if limit is None:
        limit = args.mlx_memory_bytes + 1_200_000_000
    if limit <= args.mlx_memory_bytes:
        parser.error("--process-memory-gb must be greater than --mlx-memory-gb")
    return limit


class MemoryGuard:
    def __init__(self, limit_bytes=4_000_000_000, interval=0.05):
        self.limit = limit_bytes
        self.interval = interval
        self.peak_rss = 0
        self.peak_footprint = 0
        self._stop = threading.Event()
        self._lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self._lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self._lib.proc_pid_rusage.restype = ctypes.c_int

    def sample(self):
        # rusage_info_v0: uuid[16], then ten uint64_t fields.
        info = ctypes.create_string_buffer(1024)
        if self._lib.proc_pid_rusage(os.getpid(), 0, ctypes.byref(info)) != 0:
            raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
        fields = (ctypes.c_uint64 * 10).from_buffer(info, 16)
        rss, footprint = fields[6], fields[7]
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_footprint = max(self.peak_footprint, footprint)
        return rss, footprint

    def start(self):
        self.sample()  # Fail before loading weights if monitoring is unavailable.

        def monitor():
            while not self._stop.wait(self.interval):
                try:
                    rss, footprint = self.sample()
                except OSError:
                    os.write(2, b"\nMemory monitor failed; closing ai.\n")
                    os._exit(78)
                if max(rss, footprint) > self.limit:
                    message = (f"\nMemory budget exceeded ({max(rss, footprint)/1e9:.2f} "
                               f"> {self.limit/1e9:.2f} GB); closing ai.\n")
                    os.write(2, message.encode())
                    os._exit(78)

        self._thread = threading.Thread(target=monitor, daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        self._thread.join()
        self.sample()

    def summary(self):
        return (f"process peak RSS={self.peak_rss/1e9:.2f} GB; "
                f"peak physical footprint={self.peak_footprint/1e9:.2f} GB")
