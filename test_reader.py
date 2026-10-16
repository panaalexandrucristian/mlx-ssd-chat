"""Reader fidelity and failure checks for real BF16 safetensors data."""
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
from mlx_flash_compress.expert_streaming import SafetensorsMap


class ReaderRegression(unittest.TestCase):
    def test_bf16_stacked_and_per_expert_reads(self):
        values = mx.array([1.0, -3.5, 0.125, 1000.0, 5.0, 6.0, 7.0, 8.0], dtype=mx.bfloat16).reshape(2, 2, 2)
        key = "model.layers.0.mlp.switch_mlp.gate_proj.scales"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            for stacked in (True, False):
                tensors = {key: values} if stacked else {
                    key.replace("switch_mlp", f"experts.{i}"): values[i] for i in range(2)}
                mx.save_safetensors(str(path), tensors)
                for mode in ("pread", "mmap"):
                    with self.subTest(stacked=stacked, mode=mode):
                        reader = SafetensorsMap([str(path)], read_mode=mode)
                        try:
                            actual = reader.get_expert_slice(key, [1, 0])
                            self.assertEqual(actual.dtype, mx.bfloat16)
                            self.assertTrue(mx.array_equal(actual, values[mx.array([1, 0])]).item())
                            with self.assertRaises(IndexError if stacked else KeyError):
                                reader.get_expert_slice(key, [2])
                        finally:
                            reader.close()
                        self.assertFalse(reader._fds)
                        self.assertFalse(reader._mmaps)

    def test_short_pread_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            mx.save_safetensors(str(path), {"weights": mx.ones((2, 2, 2))})
            reader = SafetensorsMap([str(path)], read_mode="pread")
            try:
                with path.open("r+b") as file:
                    file.truncate(path.stat().st_size - 1)
                with self.assertRaisesRegex(OSError, "Short safetensors read"):
                    reader.get_expert_slice("weights", [1])
            finally:
                reader.close()


if __name__ == "__main__":
    unittest.main()
