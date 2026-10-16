"""Differential regression: real Qwen3-MoE calls, tiny deterministic weights."""
import copy
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_moe import Model, ModelArgs
from mlx_flash_compress.expert_streaming import enable_expert_streaming
from cache_memory import plan_expert_cache


class StreamingRegression(unittest.TestCase):
    def test_full_model_prefill_decode_and_eviction(self):
        mx.random.seed(42)
        args = ModelArgs(
            model_type="qwen3_moe", hidden_size=64, num_hidden_layers=2,
            intermediate_size=128, num_attention_heads=2, num_experts=16,
            num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[],
            moe_intermediate_size=64, rms_norm_eps=1e-6, vocab_size=128,
            num_key_value_heads=1, head_dim=32, rope_theta=1000000,
            tie_word_embeddings=False, max_position_embeddings=40960,
            norm_topk_prob=True,
        )
        reference = Model(args)
        reference.set_dtype(mx.bfloat16)
        nn.quantize(reference, group_size=32, bits=4)
        mx.eval(reference.parameters())
        before = mx.get_active_memory()
        plans = [plan_expert_cache(reference, budget)
                 for budget in (400_130_000, 400_230_000, 1_000_000_000)]
        self.assertEqual(mx.get_active_memory(), before)
        self.assertLess(plans[0].capacity, plans[1].capacity)
        self.assertLess(plans[1].capacity, plans[2].capacity)
        self.assertEqual(plans[-1].capacity, args.num_experts)
        with self.assertRaisesRegex(ValueError, "budget too small"):
            plan_expert_cache(reference, 400_000_000)
        with tempfile.TemporaryDirectory() as directory:
            mx.save_safetensors(str(Path(directory) / "model.safetensors"),
                                dict(tree_flatten(reference.parameters())))
            for plan in [None, *plans]:
                capacity = plan.capacity if plan else 4
                with self.subTest(capacity=capacity):
                    streamed = copy.deepcopy(reference)
                    state = enable_expert_streaming(
                        streamed, capacity_per_layer=capacity, model_path=directory,
                        exact=True, read_mode="pread")
                    try:
                        state.warmup()
                        if plan:
                            resident = sum(a.nbytes for _, a in tree_flatten(streamed.parameters()))
                            for cache in state.caches:
                                for arrays in (cache.weights, cache.scales, cache.biases):
                                    resident += sum(a.nbytes for a in arrays.values())
                            self.assertEqual(resident, plan.resident_bytes)
                        from mlx_lm.models.cache import make_prompt_cache
                        ref_cache = make_prompt_cache(reference)
                        stream_cache = make_prompt_cache(streamed)
                        # Prefill overflow for small caches, then decode with evictions.
                        for tokens in [list(range(12)), [97], [30], [115], [3]]:
                            x = mx.array([tokens])
                            expected = reference(x, cache=ref_cache)
                            actual = streamed(x, cache=stream_cache)
                            mx.eval(expected, actual)
                            diff = mx.max(mx.abs(expected - actual)).item()
                            print(f"capacity={capacity} tokens={len(tokens)} max_logit_diff={diff}", flush=True)
                            self.assertLessEqual(diff, 0.03125)
                    finally:
                        state.cleanup()


if __name__ == "__main__":
    unittest.main()
