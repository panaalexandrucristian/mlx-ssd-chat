"""Size the resident Qwen3 expert cache from the MLX budget, before evaluation."""
from dataclasses import dataclass

from mlx.utils import tree_flatten


@dataclass(frozen=True)
class ExpertCachePlan:
    capacity: int
    num_experts: int
    fixed_bytes: int
    bytes_per_slot: int
    reserve_bytes: int

    @property
    def resident_bytes(self):
        return self.fixed_bytes + self.capacity * self.bytes_per_slot

    def summary(self):
        return (f"Expert cache: {self.capacity}/{self.num_experts} per layer; "
                f"resident weights ~{self.resident_bytes/1e9:.2f} GB; "
                f"MLX working reserve {self.reserve_bytes/1e9:.2f} GB")


def plan_expert_cache(model, memory_limit):
    """Inspect lazy parameter sizes without loading the full expert weights.

    Keep 10% (at least 400 MB) for KV, prefill and temporary arrays. This is a
    sizing estimate, not a hard limit; MemoryGuard monitors the whole process.
    """
    if getattr(model, "model_type", None) != "qwen3_moe":
        raise ValueError("Automatic expert cache sizing supports qwen3_moe only")

    total_bytes = sum(array.nbytes for _, array in tree_flatten(model.parameters()))
    expert_bytes = 0
    bytes_per_slot = 0
    num_experts = None
    top_k = 0
    for layer in model.model.layers:
        mlp = layer.mlp
        switch = getattr(mlp, "switch_mlp", None)
        if switch is None:
            continue
        count = switch.gate_proj.weight.shape[0]
        if num_experts is not None and count != num_experts:
            raise ValueError("Automatic cache sizing requires uniform expert counts")
        num_experts = count
        top_k = max(top_k, mlp.top_k)
        for _, array in tree_flatten(switch.parameters()):
            if array.shape[0] != count or array.nbytes % count:
                raise ValueError("Unsupported expert parameter layout for cache sizing")
            expert_bytes += array.nbytes
            bytes_per_slot += array.nbytes // count

    if not bytes_per_slot:
        raise ValueError("Cannot find expert weights for automatic cache sizing")
    fixed_bytes = total_bytes - expert_bytes
    reserve_bytes = max(400_000_000, memory_limit // 10)
    capacity = min(num_experts, (memory_limit - reserve_bytes - fixed_bytes) // bytes_per_slot)
    if capacity < top_k:
        required = fixed_bytes + top_k * bytes_per_slot + reserve_bytes
        raise ValueError(
            f"MLX budget too small for exact streaming: need at least {top_k} "
            f"experts per layer and working memory (~{required/1e9:.2f} GB). "
            "Increase --mlx-memory-gb.")
    return ExpertCachePlan(capacity, num_experts, fixed_bytes, bytes_per_slot, reserve_bytes)
