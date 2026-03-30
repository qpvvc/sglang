"""Regression test for comparing one server against a saved logprob baseline."""

from __future__ import annotations

import urllib.request
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import sglang as sgl
from sglang.lang.api import set_default_backend
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint


@dataclass
class EndpointLogprobs:
    input_logprobs: list[list[float]]
    output_logprobs: list[list[float]]


CASES = [
    "Please provide a very detailed and comprehensive explanation of the history, evolution, and future prospects of artificial intelligence, including its impact on various industries such as healthcare, finance, and transportation. Ensure the response is informative and covers multiple perspectives. Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.",
    "Could you describe the process of cellular respiration in great detail, including the stages of glycolysis, the Krebs cycle, and the electron transport chain? Explain how energy is transferred and stored within the cell, and the role of ATP. Cellular respiration is a set of metabolic reactions and processes that take place in the cells of organisms to convert biochemical energy from nutrients into adenosine triphosphate (ATP), and then release waste products.",
]


def _check_health(endpoint: str):
    with urllib.request.urlopen(f"{endpoint}/health", timeout=3):
        pass


def _extract_logprobs(logprobs) -> list[float]:
    if not logprobs:
        return []

    values = []
    for item in logprobs:
        if item is None:
            continue
        if isinstance(item, (list, tuple)) and item:
            logprob = item[0]
            if logprob is not None:
                values.append(float(logprob))
    return values


def _eval_endpoint(endpoint: str, max_new_tokens: int, num_threads: int) -> EndpointLogprobs:
    set_default_backend(RuntimeEndpoint(endpoint))

    @sgl.function
    def score_fn(s, prompt):
        s += prompt
        s += sgl.gen(
            "answer",
            max_tokens=max_new_tokens,
            # stop=["\n"],
            return_logprob=True,
            logprob_start_len=0,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
        )

    states = score_fn.run_batch(
        [{"prompt": prompt} for prompt in CASES],
        temperature=0.0,
        num_threads=num_threads,
        progress_bar=False,
    )

    input_logprobs = []
    output_logprobs = []
    for state in states:
        meta = state.get_meta_info("answer") or {}
        input_logprobs.append(_extract_logprobs(meta.get("input_token_logprobs")))
        output_logprobs.append(_extract_logprobs(meta.get("output_token_logprobs")))

    return EndpointLogprobs(
        input_logprobs=input_logprobs,
        output_logprobs=output_logprobs,
    )


def _to_dict(logprobs: EndpointLogprobs) -> dict:
    return {
        "input_logprobs": logprobs.input_logprobs,
        "output_logprobs": logprobs.output_logprobs,
    }


def _from_dict(data: dict) -> EndpointLogprobs:
    return EndpointLogprobs(
        input_logprobs=data["input_logprobs"],
        output_logprobs=data["output_logprobs"],
    )


def _max_abs_diff(
    baseline_values: list[list[float]],
    candidate_values: list[list[float]],
    label: str,
) -> float:
    if len(baseline_values) != len(candidate_values):
        pytest.fail(
            f"{label} case count mismatch: {len(baseline_values)} != {len(candidate_values)}"
        )

    max_diff = 0.0
    for case_idx, (baseline_case, candidate_case) in enumerate(
        zip(baseline_values, candidate_values, strict=True)
    ):
        if len(baseline_case) != len(candidate_case):
            pytest.fail(
                f"{label} token count mismatch at case {case_idx}: "
                f"{len(baseline_case)} != {len(candidate_case)}"
            )
        if not baseline_case:
            continue

        case_diff = float(
            np.max(np.abs(np.array(baseline_case) - np.array(candidate_case)))
        )
        max_diff = max(max_diff, case_diff)

    return max_diff


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


@pytest.mark.integration
def test_logprobs_regression(endpoint, request):
    baseline_file = Path(request.config.getoption("--baseline-file"))
    update_baseline = request.config.getoption("--update-baseline")
    max_new_tokens = request.config.getoption("--max-new-tokens")
    num_threads = request.config.getoption("--num-threads")
    max_diff = request.config.getoption("--max-logprob-diff")

    current = _eval_endpoint(
        endpoint,
        max_new_tokens=max_new_tokens,
        num_threads=num_threads,
    )

    if update_baseline:
        baseline_file.parent.mkdir(parents=True, exist_ok=True)
        with baseline_file.open("w", encoding="utf-8") as f:
            json.dump(_to_dict(current), f, ensure_ascii=False, indent=2)
        pytest.skip(f"Baseline updated: {baseline_file}")

    if not baseline_file.exists():
        pytest.fail(
            f"Baseline file not found: {baseline_file}. Run once with --update-baseline."
        )

    with baseline_file.open("r", encoding="utf-8") as f:
        baseline = _from_dict(json.load(f))

    input_diff = _max_abs_diff(
        baseline.input_logprobs,
        current.input_logprobs,
        label="input_logprobs",
    )
    output_diff = _max_abs_diff(
        baseline.output_logprobs,
        current.output_logprobs,
        label="output_logprobs",
    )

    print(
        "\nLogprob regression summary:\n"
        f"  endpoint={endpoint}\n"
        f"  baseline_file={baseline_file}\n"
        f"  max_diff={max_diff}\n"
        f"  max_input_diff={input_diff:.6f}\n"
        f"  max_output_diff={output_diff:.6f}"
    )

    assert input_diff <= max_diff, (
        f"Input logprob diff too large: {input_diff:.6f} > {max_diff:.6f}"
    )
    assert output_diff <= max_diff, (
        f"Output logprob diff too large: {output_diff:.6f} > {max_diff:.6f}"
    )
