#!/usr/bin/env python3
"""
Offline precision comparison tool.

Compares captured tensors from two platforms (e.g., MUSA vs A100).
Each platform should have run with SGLANG_PRECISION_MODE=capture to produce .pt files.

Usage:
    python compare_precision.py --ref /path/to/a100_capture --target /path/to/musa_capture [--output report.csv]
"""

import argparse
import csv
import sys
from pathlib import Path

import torch


def compute_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    if a.shape != b.shape:
        return {"error": f"shape_mismatch: {a.shape} vs {b.shape}"}
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
        "ref_norm": a_f.norm().item(),
        "target_norm": b_f.norm().item(),
    }


def find_tensor_pairs(ref_dir: Path, target_dir: Path):
    """Find matching _input.pt and _output.pt pairs between two directories."""
    pairs = []
    for ref_file in sorted(ref_dir.glob("*_output.pt")):
        key = ref_file.stem.replace("_output", "")
        target_file = target_dir / ref_file.name
        if target_file.exists():
            pairs.append((key, "output", ref_file, target_file))

    for ref_file in sorted(ref_dir.glob("*_input.pt")):
        key = ref_file.stem.replace("_input", "")
        target_file = target_dir / ref_file.name
        if target_file.exists():
            pairs.append((key, "input", ref_file, target_file))

    return pairs


def load_tensor(path: Path) -> torch.Tensor:
    data = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(data, (list, tuple)):
        data = data[0]
    return data


def main():
    parser = argparse.ArgumentParser(description="Precision comparison tool")
    parser.add_argument("--ref", required=True, help="Reference capture directory (e.g., A100)")
    parser.add_argument("--target", required=True, help="Target capture directory (e.g., MUSA)")
    parser.add_argument("--output", default=None, help="CSV output file (default: stdout)")
    args = parser.parse_args()

    ref_dir = Path(args.ref)
    target_dir = Path(args.target)

    if not ref_dir.exists():
        print(f"Error: Reference directory not found: {ref_dir}", file=sys.stderr)
        sys.exit(1)
    if not target_dir.exists():
        print(f"Error: Target directory not found: {target_dir}", file=sys.stderr)
        sys.exit(1)

    pairs = find_tensor_pairs(ref_dir, target_dir)
    if not pairs:
        print("No matching tensor pairs found.", file=sys.stderr)
        sys.exit(1)

    fieldnames = [
        "module", "type", "max_abs_diff", "mean_abs_diff",
        "max_rel_diff", "mean_rel_diff", "cosine_similarity",
        "ref_norm", "target_norm", "error"
    ]

    out_file = open(args.output, "w", newline="") if args.output else sys.stdout
    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    writer.writeheader()

    for key, tensor_type, ref_path, target_path in pairs:
        try:
            ref_t = load_tensor(ref_path)
            target_t = load_tensor(target_path)
            metrics = compute_metrics(ref_t, target_t)
        except Exception as e:
            metrics = {"error": str(e)}

        row = {"module": key, "type": tensor_type}
        row.update(metrics)
        writer.writerow(row)

        # Print summary to stderr
        cos = metrics.get("cosine_similarity", "N/A")
        max_diff = metrics.get("max_abs_diff", "N/A")
        err = metrics.get("error", "")
        status = f"cos={cos:.6f} max_diff={max_diff:.6e}" if not err else f"ERROR: {err}"
        print(f"  {key} [{tensor_type}]: {status}", file=sys.stderr)

    if args.output:
        out_file.close()
        print(f"\nReport saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
