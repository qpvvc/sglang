"""MMLU input logprob regression test.

For each MMLU question and each answer label (A/B/C/D), this test:
  1. Uses the pre-built full prompt stored in the MMLU reference JSON
     (5-shot context + question + "Answer: X")
  2. Calls SGLang with max_new_tokens=0 to obtain all input token logprobs
     (pure prefill scoring, no generation)
  3. Computes BPB over the answer label token:
       BPB = -logprob(last_token) / ln(2)
  4. Compares BPB with the reference value in the JSON  (tolerance 1e-4)
  5. Saves / compares per-token logprobs against a baseline file (tolerance 1e-4)

Workflow:
  # Step 1 – generate baseline (run once against the reference model/config)
  pytest test_accuracy_mmlu_logprobs.py -m integration --update-baseline

  # Step 2 – run regression
  pytest test_accuracy_mmlu_logprobs.py -m integration
"""

from __future__ import annotations

import json
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

LABELS = ["A", "B", "C", "D"]
_LN2 = math.log(2)   # ln(2), used for nats → bits conversion

# Hard-coded tolerances (per user requirement: 1e-4)
_BPB_TOL = 1e-4       # max |BPB_current - BPB_reference|
_LP_TOL = 1e-4        # max |logprob_current - logprob_baseline| per token


# --------------------------------------------------------------------------- #
# SGLang HTTP scoring (prefill-only)
# --------------------------------------------------------------------------- #

def _check_health(endpoint: str) -> None:
    with urllib.request.urlopen(f"{endpoint}/health", timeout=5):
        pass


def _score_prompt(endpoint: str, text: str) -> list[float]:
    """Return all input token logprobs (nats) via SGLang /generate.

    max_new_tokens=0 triggers prefill-only mode: the model scores every token
    in the prompt but generates nothing.  The first token's logprob is always
    None (no predecessor); those None entries are filtered out.
    """
    payload = json.dumps({
        "text": text,
        "sampling_params": {
            "max_new_tokens": 0,
            "return_logprob": True,
            "logprob_start_len": 0,
            "temperature": 0.0,
        },
    }).encode()

    req = urllib.request.Request(
        f"{endpoint}/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())

    raw = result.get("meta_info", {}).get("input_token_logprobs") or []
    return [float(item[0]) for item in raw if item is not None and item[0] is not None]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _bpb(logprobs: list[float]) -> float:
    """BPB for the answer label token (last token in the prompt).

    The prompt ends with the single-byte answer character (A/B/C/D), so:
        BPB = -logprob(last_token) / ln(2) = -log2(p(label_token))
    """
    if not logprobs:
        return float("inf")
    return -logprobs[-1] / _LN2


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def endpoint(request):
    target = (
        f"{request.config.getoption('--sglang-host')}:"
        f"{request.config.getoption('--sglang-port')}"
    )
    try:
        _check_health(target)
    except Exception:
        pytest.skip(f"Server is not reachable: {target}")
    return target


# --------------------------------------------------------------------------- #
# Main test
# --------------------------------------------------------------------------- #

@pytest.mark.integration
def test_mmlu_input_logprobs(endpoint, request):
    """Validate prefill logprob accuracy on MMLU security-studies prompts.

    Pass criteria (both must hold):
      - max |logprob_current[i] - logprob_baseline[i]| ≤ 1e-4  (per token)
      - max |BPB_current - BPB_reference|               ≤ 1e-4  (per label)
    """
    mmlu_path = Path(request.config.getoption("--mmlu-file"))
    baseline_file = Path(request.config.getoption("--mmlu-baseline-file"))
    update_baseline = request.config.getoption("--update-baseline")
    num_threads = request.config.getoption("--num-threads")

    if not mmlu_path.exists():
        pytest.fail(f"MMLU data file not found: {mmlu_path}")

    with mmlu_path.open("r", encoding="utf-8") as f:
        mmlu_data: dict[str, Any] = json.load(f)

    # Build task list – use the pre-built "prompt" field stored in the JSON.
    # Each prompt is the complete 5-shot context ending with "Answer: X".
    tasks: list[tuple[str, str]] = []
    for qid, entry in mmlu_data.items():
        for label in LABELS:
            label_key = f"label: {label}"
            if label_key not in entry:
                continue
            tasks.append((f"{qid}:{label}", entry[label_key]["prompt"]))

    # Score all prompts in parallel
    current: dict[str, list[float]] = {}

    def _eval(key: str, prompt: str) -> tuple[str, list[float]]:
        return key, _score_prompt(endpoint, prompt)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(_eval, key, prompt) for key, prompt in tasks]
        for fut in as_completed(futures):
            key, logprobs = fut.result()
            current[key] = logprobs

    # ---- Update baseline mode --------------------------------------------- #
    if update_baseline:
        baseline_file.parent.mkdir(parents=True, exist_ok=True)
        with baseline_file.open("w", encoding="utf-8") as f:
            json.dump(current, f, ensure_ascii=False, indent=2)
        pytest.skip(f"Baseline updated: {baseline_file}")

    if not baseline_file.exists():
        pytest.fail(
            f"Baseline not found: {baseline_file}. "
            "Run once with --update-baseline to create it."
        )

    with baseline_file.open("r", encoding="utf-8") as f:
        baseline: dict[str, list[float]] = json.load(f)

    # ---- Check 1: per-token logprob vs saved baseline --------------------- #
    max_lp_diff = 0.0
    for key, cur_lps in current.items():
        base_lps = baseline.get(key, [])
        if len(cur_lps) != len(base_lps):
            pytest.fail(
                f"{key}: token count mismatch "
                f"current={len(cur_lps)} baseline={len(base_lps)}"
            )
        if not cur_lps:
            continue
        diff = float(np.max(np.abs(np.array(cur_lps) - np.array(base_lps))))
        max_lp_diff = max(max_lp_diff, diff)

    # ---- Check 2: BPB vs MMLU reference JSON ------------------------------ #
    max_bpb_diff = 0.0
    for qid, entry in mmlu_data.items():
        for label in LABELS:
            label_key = f"label: {label}"
            if label_key not in entry:
                continue
            key = f"{qid}:{label}"
            if key not in current:
                continue
            ref_bpb = float(entry[label_key]["BPB"])
            cur_bpb = _bpb(current[key])
            diff = abs(cur_bpb - ref_bpb)
            max_bpb_diff = max(max_bpb_diff, diff)

    print(
        "\nMMLU logprob regression summary:\n"
        f"  endpoint     = {endpoint}\n"
        f"  mmlu_file    = {mmlu_path}\n"
        f"  num_tasks    = {len(tasks)}\n"
        f"  max_lp_diff  = {max_lp_diff:.2e}  (tol={_LP_TOL:.0e})\n"
        f"  max_bpb_diff = {max_bpb_diff:.2e}  (tol={_BPB_TOL:.0e})"
    )

    assert max_lp_diff <= _LP_TOL, (
        f"Per-token logprob diff too large: {max_lp_diff:.2e} > {_LP_TOL:.2e}"
    )
    assert max_bpb_diff <= _BPB_TOL, (
        f"BPB diff too large: {max_bpb_diff:.2e} > {_BPB_TOL:.2e}"
    )
