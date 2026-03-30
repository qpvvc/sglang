"""
SGLang precision test suite for few-shot GSM-8K evaluation.

Run integration test (requires sglang service on 127.0.0.1:30000):
    pytest test_few_shot_gsm8k.py -v -m integration

Run with custom service endpoint:
    pytest test_few_shot_gsm8k.py -v --sglang-host http://192.168.1.10 --sglang-port 30000

Generate detailed accuracy report:
    pytest test_few_shot_gsm8k.py -v -m integration --tb=short 2>&1 | tee accuracy_report.txt
"""

import argparse
import ast
import json
import re
import time
import pytest
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

from sglang.lang.api import set_default_backend
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.utils import download_and_cache_file, read_jsonl

INVALID = -9999999


# ============================================================================
# Core Logic (extracted from few_shot_gsm8k.py)
# ============================================================================

def get_one_example(lines, i, include_answer):
    """Format single example."""
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    """Construct few-shot prompt."""
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    """Extract numerical answer from string."""
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


@dataclass
class PrecisionTestResult:
    """Structured result container."""
    num_questions: int
    num_correct: int
    num_invalid: int
    accuracy: float
    invalid_ratio: float
    latency: float
    output_throughput: float
    predictions: List[int]
    ground_truth: List[int]
    details: List[Dict[str, Any]]


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_addoption(parser):
    """Register custom CLI options."""
    parser.addoption("--sglang-host", action="store", default="http://127.0.0.1", help="SGLang service host")
    parser.addoption("--sglang-port", action="store", type=int, default=30000, help="SGLang service port")
    parser.addoption("--num-questions", action="store", type=int, default=10, help="Number of test questions")
    parser.addoption("--num-shots", action="store", type=int, default=5, help="Number of few-shot examples")
    parser.addoption("--data-path", action="store", default=None, help="Path to GSM8K test data (JSONL)")

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test (requires sglang service)")

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def sglang_endpoint(request):
    """Get SGLang service endpoint from CLI args."""
    host = request.config.getoption("--sglang-host")
    port = request.config.getoption("--sglang-port")
    endpoint = f"{host}:{port}"
    
    # Check connection
    try:
        RuntimeEndpoint(endpoint)
    except Exception as e:
        pytest.skip(f"SGLang service not available at {endpoint}: {e}")
    
    return endpoint


@pytest.fixture(scope="session")
def gsm8k_data(request):
    """Load GSM8K test data from local file."""
    local_path = request.config.getoption("--data-path", default=None)
    
    if not local_path:
        pytest.skip("--data-path 必须指定")
    
    if not Path(local_path).exists():
        pytest.skip(f"文件不存在: {local_path}")
    
    return list(read_jsonl(local_path))


@pytest.fixture
def test_config(request):
    """Test configuration from CLI arguments."""
    return {
        "num_questions": request.config.getoption("--num-questions"),
        "num_shots": request.config.getoption("--num-shots"),
        "max_new_tokens": 512,
        "temperature": 0.0,
    }


# ============================================================================
# Tests
# ============================================================================

@pytest.mark.integration
def test_gsm8k_few_shot_accuracy(sglang_endpoint, gsm8k_data, test_config):
    """Main precision test: run few-shot GSM8K inference and compare with ground truth."""
    
    set_default_backend(RuntimeEndpoint(sglang_endpoint))
    
    lines = gsm8k_data[:test_config["num_questions"]]
    num_shots = test_config["num_shots"]
    few_shot_examples = get_few_shot_examples(lines, num_shots)
    
    questions, labels = [], []
    for i in range(len(lines)):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    
    assert all(l != INVALID for l in labels), "Test data contains invalid labels"
    
    import sglang as sgl
    
    @sgl.function
    def few_shot_gsm8k(s, question):
        s += few_shot_examples + question
        s += sgl.gen("answer", max_tokens=test_config["max_new_tokens"], stop=["Question", "Assistant:", "<|separator|>"])
    
    tic = time.perf_counter()
    states = few_shot_gsm8k.run_batch([{"question": q} for q in questions], temperature=test_config["temperature"], num_threads=64, progress_bar=True)
    latency = time.perf_counter() - tic
    
    preds = []
    details = []
    for i, state in enumerate(states):
        pred = get_answer_value(state["answer"])
        preds.append(pred)
        details.append({"idx": i, "question": questions[i], "prediction": pred, "ground_truth": labels[i], "generated_text": state["answer"], "correct": pred == labels[i]})
    
    preds_arr = np.array(preds)
    labels_arr = np.array(labels)
    num_correct = np.sum(preds_arr == labels_arr)
    num_invalid = np.sum(preds_arr == INVALID)
    accuracy = num_correct / len(preds)
    invalid_ratio = num_invalid / len(preds)
    num_output_tokens = sum(s.get_meta_info("answer")["completion_tokens"] for s in states)
    output_throughput = num_output_tokens / latency if latency > 0 else 0
    
    result = PrecisionTestResult(
        num_questions=len(lines), num_correct=int(num_correct), num_invalid=int(num_invalid),
        accuracy=float(accuracy), invalid_ratio=float(invalid_ratio), latency=float(latency),
        output_throughput=float(output_throughput), predictions=preds, ground_truth=labels, details=details
    )
    
    print_precision_report(result)
    save_test_results(result)
    
    assert accuracy >= 0.5, f"Accuracy too low: {accuracy:.3f}"
    assert invalid_ratio < 0.3, f"Invalid ratio too high: {invalid_ratio:.3f}"


@pytest.mark.integration
@pytest.mark.parametrize("num_shots", [1, 3, 5])
def test_gsm8k_few_shot_comparison(sglang_endpoint, gsm8k_data, num_shots):
    """Comparative test: accuracy with different few-shot counts."""
    set_default_backend(RuntimeEndpoint(sglang_endpoint))
    
    lines = gsm8k_data[:20]
    few_shot_examples = get_few_shot_examples(lines, num_shots)
    
    questions, labels = [], []
    for i in range(len(lines)):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    
    import sglang as sgl
    
    @sgl.function
    def few_shot_gsm8k(s, question):
        s += few_shot_examples + question
        s += sgl.gen("answer", max_tokens=512, stop=["Question", "Assistant:", "<|separator|>"])
    
    states = few_shot_gsm8k.run_batch([{"question": q} for q in questions], temperature=0.0, num_threads=32)
    preds = [get_answer_value(s["answer"]) for s in states]
    accuracy = np.mean(np.array(preds) == np.array(labels))
    
    print(f"\nFew-shot={num_shots} -> Accuracy: {accuracy:.3f}")
    assert accuracy >= 0.2, f"Accuracy with {num_shots} shots: {accuracy:.3f}"


def test_gsm8k_answer_extraction():
    """Unit test: verify answer extraction logic (no service required)."""
    test_cases = [("The answer is 42", 42), ("First 10, then 20, total 30", 30), ("1,234 items", 1234), ("No number here", INVALID), ("", INVALID)]
    for answer_str, expected in test_cases:
        result = get_answer_value(answer_str)
        assert result == expected, f"Failed for '{answer_str}': got {result}, expected {expected}"


# ============================================================================
# Utilities
# ============================================================================

def print_precision_report(result: PrecisionTestResult):
    """Print formatted precision test report."""
    print("\n" + "=" * 80)
    print("SGLang GSM8K Precision Test Report".center(80))
    print("=" * 80)
    print(f"Test Questions:      {result.num_questions}")
    print(f"Correct Predictions: {result.num_correct}/{result.num_questions}")
    print(f"Invalid Answers:     {result.num_invalid}/{result.num_questions}")
    print("-" * 80)
    print(f"Accuracy:            {result.accuracy:.1%}")
    print(f"Invalid Ratio:       {result.invalid_ratio:.1%}")
    print(f"Latency:             {result.latency:.3f} s")
    print(f"Output Throughput:   {result.output_throughput:.1f} token/s")
    print("=" * 80)
    
    incorrect = [d for d in result.details if not d["correct"]]
    if incorrect:
        print(f"\nIncorrect Predictions ({len(incorrect)}):")
        for d in incorrect[:5]:
            print(f"  Q{d['idx']}: pred={d['prediction']}, truth={d['ground_truth']}")


def save_test_results(result: PrecisionTestResult):
    """Save detailed test results to JSON."""
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"gsm8k_precision_{int(time.time())}.json"
    with open(output_file, "w") as f:
        json.dump({
            "num_questions": result.num_questions, "num_correct": result.num_correct,
            "num_invalid": result.num_invalid, "accuracy": result.accuracy,
            "invalid_ratio": result.invalid_ratio, "latency": result.latency,
            "output_throughput": result.output_throughput, "details": result.details
        }, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])