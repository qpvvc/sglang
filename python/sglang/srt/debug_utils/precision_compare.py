"""
Hook-based precision comparison tool for cross-platform debugging (e.g., MUSA vs A100).

Design principle: "逐算子输入对齐、逐算子输出比较"
  - Hook every leaf nn.Module (Linear, RMSNorm, SiluAndMul, etc.) inside decoder layers
  - In compare+align mode: cascade-replace each operator's input with A100 reference,
    so each operator's output diff reflects ONLY that operator's platform difference
  - Non-Module callables (LayerCommunicator.prepare_attn/prepare_mlp) are wrapped separately

Two operation modes:
  - capture: Save input/output tensors of each leaf module during inference.
  - compare: Load reference tensors, replace inputs, and compute diff metrics per operator.

Environment Variables:
  SGLANG_PRECISION_MODE       : "capture" or "compare" (unset = disabled)
  SGLANG_PRECISION_DIR        : Directory for saving captured tensors (default: /tmp/sglang_precision)
  SGLANG_PRECISION_REF_DIR    : Reference directory for compare mode (required in compare mode)
  SGLANG_PRECISION_LAYERS     : Comma-separated layer IDs to trace, e.g. "0,1,2" (default: all)
  SGLANG_PRECISION_MIN_TOKENS : Min token count to trigger capture (default: 100)
  SGLANG_PRECISION_ALIGN_INPUT: "1" to replace each operator's input with reference (default: "0")
  SGLANG_PRECISION_LEAF_ONLY  : "1" to only hook leaf modules, "0" to also hook containers (default: "1")
  SGLANG_PRECISION_METHODS    : Comma-separated method paths to wrap on decoder layers, e.g.
                                "layer_communicator.prepare_attn,layer_communicator.prepare_mlp"

Usage (zero model-code changes):
  Automatically called from model_runner.py after weight loading when env vars are set.
  Or manually:
      from sglang.srt.debug_utils.precision_compare import PrecisionDebugger
      debugger = PrecisionDebugger.from_env()
      if debugger:
          debugger.attach(model)
"""

import functools
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_METHOD_WRAPS = [
    "layer_communicator.prepare_attn",
    "layer_communicator.prepare_mlp",
]


def _to_cpu(t):
    """Recursively move tensors to CPU. NamedTuples are converted to plain tuples
    to avoid pickling custom classes that cause weights_only=True load failures."""
    if isinstance(t, torch.Tensor):
        return t.clone().detach().cpu()
    if isinstance(t, (list, tuple)):
        converted = [_to_cpu(x) for x in t]
        # Convert NamedTuple → plain tuple so torch.save/load doesn't depend on the class
        return tuple(converted) if type(t) is not list else converted
    if isinstance(t, dict):
        return {k: _to_cpu(v) for k, v in t.items()}
    return None


def _first_tensor(args) -> Optional[torch.Tensor]:
    """Find the first torch.Tensor in args (positional)."""
    if isinstance(args, torch.Tensor):
        return args
    if isinstance(args, (tuple, list)):
        for a in args:
            if isinstance(a, torch.Tensor) and a.is_floating_point():
                return a
    return None


def _clone_float_tensors_to_cpu(args) -> list:
    """Clone all floating-point tensors in args to CPU (before forward can modify in-place)."""
    clones = []
    if isinstance(args, torch.Tensor):
        if args.is_floating_point():
            clones.append(args.clone().detach().cpu())
    elif isinstance(args, (tuple, list)):
        for a in args:
            if isinstance(a, torch.Tensor) and a.is_floating_point():
                clones.append(a.clone().detach().cpu())
    return clones


def _replace_all_float_tensors(args, ref_tensors: list, device, dtype):
    """Replace ALL floating-point tensors in args with corresponding ref tensors.
    Returns modified args tuple, or original args if shapes don't match."""
    if not ref_tensors:
        return args
    new_args = list(args)
    ref_idx = 0
    for i, a in enumerate(new_args):
        if isinstance(a, torch.Tensor) and a.is_floating_point() and ref_idx < len(ref_tensors):
            ref_t = ref_tensors[ref_idx]
            if isinstance(ref_t, torch.Tensor) and ref_t.shape == a.shape:
                new_args[i] = ref_t.to(device).to(dtype)
            ref_idx += 1
    return tuple(new_args)


def _safe_name(module_path: str) -> str:
    """Convert dotted module path to a filesystem-safe name."""
    return module_path.replace(".", "_")


def _compute_metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """Compute comparison metrics between two tensors."""
    if a.shape != b.shape:
        return {"error": "shape_mismatch", "a_shape": str(a.shape), "b_shape": str(b.shape)}
    a_f = a.float()
    b_f = b.float()
    diff = (a_f - b_f).abs()
    rel_diff = diff / (b_f.abs().clamp(min=1e-7))
    cos = torch.nn.functional.cosine_similarity(
        a_f.flatten().unsqueeze(0), b_f.flatten().unsqueeze(0)
    ).item()
    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_rel_diff": rel_diff.mean().item(),
        "cosine_similarity": cos,
    }


def _sort_topk_by_ids(
    topk_weights: torch.Tensor, topk_ids: torch.Tensor
) -> tuple:
    """Sort topk results per token by expert id for order-insensitive comparison."""
    sorted_indices = topk_ids.argsort(dim=-1)
    sorted_ids = topk_ids.gather(1, sorted_indices)
    sorted_weights = topk_weights.gather(1, sorted_indices.long())
    return sorted_weights, sorted_ids


def _compute_topk_ids_metrics(
    actual_ids: torch.Tensor, ref_ids: torch.Tensor
) -> Dict[str, float]:
    """Compute set-match and sorted-exact-match metrics for topk_ids."""
    num_tokens = actual_ids.shape[0]
    set_match = sum(
        set(actual_ids[i].tolist()) == set(ref_ids[i].tolist())
        for i in range(num_tokens)
    )
    exact_match = (actual_ids == ref_ids).all(dim=-1).sum().item()
    return {
        "set_match_rate": set_match / num_tokens,
        "exact_match_rate": exact_match / num_tokens,
        "num_tokens": num_tokens,
    }


def _is_topk_key(key: str) -> bool:
    """Detect if the hook key corresponds to a TopK operator output."""
    return key.endswith("_topk") or "_topk_" in key


def _is_leaf_module(module: nn.Module) -> bool:
    """Check if a module is a leaf (has no nn.Module children)."""
    return len(list(module.children())) == 0


def _is_in_decoder_layers(name: str) -> bool:
    """Check if module path is under model.layers.N (a decoder layer)."""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return True
    return False


def _extract_layer_id_from_path(name: str) -> Optional[int]:
    """Extract layer ID from a module path like 'model.layers.3.self_attn.q_b_proj'."""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def _is_decoder_layer(name: str) -> bool:
    """Check if this is a top-level decoder layer (e.g., 'model.layers.3')."""
    parts = name.split(".")
    return (
        len(parts) >= 3
        and parts[-2] == "layers"
        and parts[-1].isdigit()
    )


class PrecisionDebugger:
    """Hook-based precision comparison debugger — hooks every leaf operator."""

    def __init__(
        self,
        mode: str,
        save_dir: str,
        ref_dir: Optional[str] = None,
        layer_ids: Optional[Set[int]] = None,
        method_wraps: Optional[List[str]] = None,
        min_tokens: int = 100,
        align_input: bool = False,
        leaf_only: bool = True,
    ):
        assert mode in ("capture", "compare"), f"Invalid mode: {mode}"
        self.mode = mode
        self.save_dir = Path(save_dir)
        self.ref_dir = Path(ref_dir) if ref_dir else None
        self.layer_ids = layer_ids
        self.method_wraps = method_wraps or DEFAULT_METHOD_WRAPS
        self.min_tokens = min_tokens
        self.align_input = align_input
        self.leaf_only = leaf_only
        self._saved: Set[str] = set()
        self._pre_clone: Dict[str, Any] = {}  # key → list of CPU tensor clones (pre-forward)
        self._rank: Optional[int] = None

        self.save_dir.mkdir(parents=True, exist_ok=True)

        if mode == "compare":
            assert self.ref_dir and self.ref_dir.exists(), (
                f"Reference directory '{self.ref_dir}' not found for compare mode"
            )

    @classmethod
    def from_env(cls) -> Optional["PrecisionDebugger"]:
        """Create a debugger from environment variables. Returns None if disabled."""
        mode = os.environ.get("SGLANG_PRECISION_MODE", "").strip().lower()
        if mode not in ("capture", "compare"):
            return None

        save_dir = os.environ.get("SGLANG_PRECISION_DIR", "/tmp/sglang_precision")
        ref_dir = os.environ.get("SGLANG_PRECISION_REF_DIR")
        min_tokens = int(os.environ.get("SGLANG_PRECISION_MIN_TOKENS", "100"))
        align_input = os.environ.get("SGLANG_PRECISION_ALIGN_INPUT", "0") == "1"
        leaf_only = os.environ.get("SGLANG_PRECISION_LEAF_ONLY", "1") == "1"

        layer_ids = None
        layers_str = os.environ.get("SGLANG_PRECISION_LAYERS", "").strip()
        if layers_str:
            layer_ids = set(int(x.strip()) for x in layers_str.split(",") if x.strip())

        method_wraps = None
        methods_str = os.environ.get("SGLANG_PRECISION_METHODS", "").strip()
        if methods_str:
            method_wraps = [s.strip() for s in methods_str.split(",") if s.strip()]

        return cls(
            mode=mode,
            save_dir=save_dir,
            ref_dir=ref_dir,
            layer_ids=layer_ids,
            method_wraps=method_wraps,
            min_tokens=min_tokens,
            align_input=align_input,
            leaf_only=leaf_only,
        )

    def _get_rank(self) -> int:
        if self._rank is None:
            try:
                from sglang.srt.distributed import get_tensor_model_parallel_rank
                self._rank = get_tensor_model_parallel_rank()
            except Exception:
                self._rank = 0
        return self._rank

    def _is_rank0(self) -> bool:
        return self._get_rank() == 0

    def _should_hook(self, name: str, module: nn.Module) -> bool:
        """Decide whether to hook this module.

        Strategy: hook all leaf nn.Modules inside decoder layers.
        Also hooks embedding/norm/lm_head at model level.
        Skips modules not in a decoder layer unless they are top-level model modules.
        """
        # Always skip the top-level model container itself
        if name == "" or name == "model":
            return False

        # Top-level modules outside decoder layers: embed_tokens, norm, lm_head
        if not _is_in_decoder_layers(name):
            # Hook these specific top-level modules
            top_level_names = {"embed_tokens", "norm", "lm_head"}
            leaf_name = name.split(".")[-1]
            if leaf_name in top_level_names:
                return True
            return False

        # Layer filter
        if self.layer_ids is not None:
            layer_id = _extract_layer_id_from_path(name)
            if layer_id is not None and layer_id not in self.layer_ids:
                return False

        # For decoder layer modules: hook leaves (or all if leaf_only=False)
        if self.leaf_only:
            if _is_leaf_module(module):
                # Skip quant_method — it's an nn.Module (via MultiPlatformOp) but its
                # computation is invoked via .apply(), not __call__, so hooks never fire.
                leaf_name = name.split(".")[-1]
                if leaf_name == "quant_method":
                    return False
                return True
            # FusedMoE (model.layers.N.mlp.experts) is not a leaf because it owns
            # quant_method as a child, but it has a real forward() that drives the
            # entire expert GEMM.  Hook it as a special non-leaf.
            leaf_name = name.split(".")[-1]
            if leaf_name == "experts":
                return True
            return False
        return True

    def _trace_key(self, name: str) -> str:
        return _safe_name(name)

    def _should_trigger(self, first_tensor: Optional[torch.Tensor]) -> bool:
        if first_tensor is None:
            return False
        return first_tensor.shape[0] >= self.min_tokens

    def _save_tensors(self, key: str, input_data, output_data):
        """Save input and output tensors to disk (once per key).

        input_data: list of CPU tensor clones (from pre_hook), or legacy single tensor.
        output_data: raw output from forward (will be _to_cpu'd).
        """
        if key in self._saved:
            return
        self._saved.add(key)

        # input_data is already CPU clones from pre_hook; save as list
        if isinstance(input_data, list):
            input_cpu = input_data  # already cloned to CPU in pre_hook
        else:
            input_cpu = _to_cpu(input_data)
        output_cpu = _to_cpu(output_data)

        if input_cpu is not None:
            torch.save(input_cpu, str(self.save_dir / f"{key}_input.pt"))
        if output_cpu is not None:
            torch.save(output_cpu, str(self.save_dir / f"{key}_output.pt"))

        logger.info(f"[PrecisionDebugger] Captured: {key}")

    def _load_ref(self, key: str, suffix: str):
        if self.ref_dir is None:
            return None
        ref_path = self.ref_dir / f"{key}_{suffix}.pt"
        if not ref_path.exists():
            return None
        try:
            # weights_only=False is required when the file contains custom classes
            # (e.g. StandardTopKOutput NamedTuple saved by older captures).
            # This is safe because the files are our own debug captures.
            return torch.load(str(ref_path), map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning(f"[PrecisionDebugger] Failed to load ref {ref_path}: {e}")
            return None

    def _compare_topk_output(self, key: str, output_tuple, ref_tuple):
        """Compare TopK output (topk_weights, topk_ids, ...) with reference.

        Performs both raw (order-sensitive) and id-sorted (order-insensitive)
        weight comparison, plus expert-id set-match metrics.
        """
        actual_weights = output_tuple[0] if isinstance(output_tuple[0], torch.Tensor) else None
        actual_ids = (
            output_tuple[1]
            if len(output_tuple) > 1 and isinstance(output_tuple[1], torch.Tensor)
            else None
        )
        ref_weights = ref_tuple[0] if isinstance(ref_tuple[0], torch.Tensor) else None
        ref_ids = (
            ref_tuple[1]
            if len(ref_tuple) > 1 and isinstance(ref_tuple[1], torch.Tensor)
            else None
        )

        if actual_weights is None or ref_weights is None:
            return

        all_metrics: Dict[str, float] = {}

        # Raw (order-sensitive) weight comparison
        raw_metrics = _compute_metrics(actual_weights, ref_weights)
        all_metrics.update({f"weights_raw_{k}": v for k, v in raw_metrics.items()})

        # Sorted (order-insensitive) weight comparison + id metrics
        if actual_ids is not None and ref_ids is not None:
            sorted_actual_w, sorted_actual_ids = _sort_topk_by_ids(
                actual_weights.float(), actual_ids
            )
            sorted_ref_w, sorted_ref_ids = _sort_topk_by_ids(
                ref_weights.float(), ref_ids
            )
            sorted_metrics = _compute_metrics(sorted_actual_w, sorted_ref_w)
            all_metrics.update({f"weights_sorted_{k}": v for k, v in sorted_metrics.items()})

            ids_metrics = _compute_topk_ids_metrics(sorted_actual_ids, sorted_ref_ids)
            all_metrics.update({f"ids_{k}": v for k, v in ids_metrics.items()})

        logger.info(f"[PrecisionDebugger] Compare {key} (topk): {all_metrics}")

        metrics_path = self.save_dir / f"{key}_metrics.txt"
        with open(str(metrics_path), "w") as f:
            for k, v in all_metrics.items():
                f.write(f"{k}: {v}\n")

    def _compare_and_log(self, key: str, output_tensor):
        """Compare output with reference and log metrics."""
        if key in self._saved:
            return
        self._saved.add(key)

        ref_output = self._load_ref(key, "output")
        if ref_output is None:
            logger.warning(f"[PrecisionDebugger] No reference output for: {key}")
            return

        output_cpu = _to_cpu(output_tensor)
        if output_cpu is None:
            return

        # TopK output: sort by expert ids before comparing weights, also compare ids.
        if (
            _is_topk_key(key)
            and isinstance(output_cpu, (tuple, list))
            and len(output_cpu) >= 2
            and isinstance(ref_output, (tuple, list))
            and len(ref_output) >= 2
        ):
            torch.save(output_cpu, str(self.save_dir / f"{key}_output.pt"))
            self._compare_topk_output(key, output_cpu, ref_output)
            return

        # Handle tuple/list outputs (default: take first element)
        if isinstance(output_cpu, (tuple, list)):
            output_cpu = output_cpu[0] if output_cpu else None
        if isinstance(ref_output, (tuple, list)):
            ref_output = ref_output[0] if ref_output else None

        if not isinstance(output_cpu, torch.Tensor) or not isinstance(ref_output, torch.Tensor):
            return

        metrics = _compute_metrics(output_cpu, ref_output)
        logger.info(f"[PrecisionDebugger] Compare {key}: {metrics}")

        # Save output and metrics for offline analysis
        torch.save(output_cpu, str(self.save_dir / f"{key}_output.pt"))
        metrics_path = self.save_dir / f"{key}_metrics.txt"
        with open(str(metrics_path), "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")

    def _replace_first_tensor_in_args(self, args, replacement: torch.Tensor):
        """Replace the first floating-point tensor in args tuple with replacement."""
        new_args = list(args)
        for i, a in enumerate(new_args):
            if isinstance(a, torch.Tensor) and a.is_floating_point():
                new_args[i] = replacement
                return tuple(new_args)
        return args

    def _make_pre_hook(self, key: str):
        """Pre-hook: clone inputs (in-place protection) and optionally align with reference."""
        debugger = self

        def pre_hook(module, args):
            if not debugger._is_rank0():
                return None
            try:
                first_t = _first_tensor(args)
                if not debugger._should_trigger(first_t):
                    return None

                # Clone all float tensors BEFORE forward (protects against in-place ops
                # like rotary_emb which modifies query/key in-place)
                if debugger.mode == "capture":
                    debugger._pre_clone[key] = _clone_float_tensors_to_cpu(args)

                if debugger.mode == "compare" and debugger.align_input:
                    ref_input = debugger._load_ref(key, "input")
                    if ref_input is not None and first_t is not None:
                        # Normalize old format (single tensor) to list
                        if isinstance(ref_input, torch.Tensor):
                            ref_input = [ref_input]
                        elif isinstance(ref_input, (tuple, list)):
                            ref_input = [t for t in ref_input if isinstance(t, torch.Tensor)]
                        return _replace_all_float_tensors(
                            args, ref_input, first_t.device, first_t.dtype
                        )
            except Exception as e:
                logger.warning(f"[PrecisionDebugger] pre_hook error for {key}: {e}")
            return None

        return pre_hook

    def _make_post_hook(self, key: str):
        """Post-hook: capture or compare this operator's output."""
        debugger = self

        def post_hook(module, input, output):
            if not debugger._is_rank0():
                return
            try:
                first_t = _first_tensor(input)
                if not debugger._should_trigger(first_t):
                    return

                if debugger.mode == "capture":
                    # Use pre-cloned input (safe from in-place modification)
                    input_clones = debugger._pre_clone.pop(key, None)
                    debugger._save_tensors(key, input_clones, output)
                elif debugger.mode == "compare":
                    debugger._compare_and_log(key, output)
            except Exception as e:
                logger.warning(f"[PrecisionDebugger] post_hook error for {key}: {e}")

        return post_hook

    def _wrap_method(self, owner, method_attr: str, trace_key: str):
        """Wrap a non-Module callable to trace its inputs and outputs."""
        original = getattr(owner, method_attr)
        debugger = self

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            if not debugger._is_rank0():
                return original(*args, **kwargs)

            try:
                first_t = _first_tensor(args)
                if not debugger._should_trigger(first_t):
                    return original(*args, **kwargs)

                # Clone inputs before forward (in-place protection)
                input_clones = None
                if debugger.mode == "capture":
                    input_clones = _clone_float_tensors_to_cpu(args)

                if debugger.mode == "compare" and debugger.align_input:
                    ref_input = debugger._load_ref(trace_key, "input")
                    if ref_input is not None and first_t is not None:
                        # Normalize old format (single tensor) to list
                        if isinstance(ref_input, torch.Tensor):
                            ref_input = [ref_input]
                        elif isinstance(ref_input, (tuple, list)):
                            ref_input = [t for t in ref_input if isinstance(t, torch.Tensor)]
                        args = _replace_all_float_tensors(
                            args, ref_input, first_t.device, first_t.dtype
                        )

                result = original(*args, **kwargs)

                if debugger.mode == "capture":
                    debugger._save_tensors(trace_key, input_clones, result)
                elif debugger.mode == "compare":
                    debugger._compare_and_log(trace_key, result)

                return result
            except Exception as e:
                logger.warning(f"[PrecisionDebugger] wrapper error for {trace_key}: {e}")
                return original(*args, **kwargs)

        setattr(owner, method_attr, wrapper)
        logger.info(f"[PrecisionDebugger] Wrapped method: {trace_key}")

    def attach(self, model: nn.Module):
        """Attach hooks to all leaf operators in decoder layers. Call once after model init."""
        if not self._is_rank0():
            return

        hooked = 0
        for name, module in model.named_modules():
            if self._should_hook(name, module):
                key = self._trace_key(name)
                module.register_forward_pre_hook(self._make_pre_hook(key))
                module.register_forward_hook(self._make_post_hook(key))
                hooked += 1
                logger.info(f"[PrecisionDebugger] Hooked: {name} (type={type(module).__name__})")

        # Wrap non-Module methods on decoder layers (e.g., LayerCommunicator.prepare_attn)
        wrapped = 0
        if self.method_wraps:
            for name, module in model.named_modules():
                if not _is_decoder_layer(name):
                    continue
                layer_id = _extract_layer_id_from_path(name)
                if self.layer_ids is not None and layer_id not in self.layer_ids:
                    continue

                for method_path in self.method_wraps:
                    parts = method_path.split(".")
                    owner = module
                    valid = True
                    for part in parts[:-1]:
                        owner = getattr(owner, part, None)
                        if owner is None:
                            valid = False
                            break
                    if not valid:
                        continue

                    method_name = parts[-1]
                    if not hasattr(owner, method_name):
                        continue

                    trace_key = self._trace_key(f"{name}.{method_path}")
                    self._wrap_method(owner, method_name, trace_key)
                    wrapped += 1

        logger.info(
            f"[PrecisionDebugger] mode={self.mode}, "
            f"hooked={hooked} leaf modules, wrapped={wrapped} methods, "
            f"save_dir={self.save_dir}"
        )

    def summary(self) -> str:
        """Return a summary of all comparison results (for compare mode)."""
        if self.mode != "compare":
            return "Not in compare mode."

        lines = ["=== Precision Comparison Summary ==="]
        for p in sorted(self.save_dir.glob("*_metrics.txt")):
            key = p.stem.replace("_metrics", "")
            lines.append(f"\n--- {key} ---")
            lines.append(p.read_text().strip())

        return "\n".join(lines)
