import pytest


def pytest_addoption(parser):
    """Register custom CLI options."""
    parser.addoption("--sglang-host", action="store", default="http://127.0.0.1", help="SGLang service host")
    parser.addoption("--sglang-port", action="store", type=int, default=30000, help="SGLang service port")
    parser.addoption("--num-questions", action="store", type=int, default=-1, help="Number of eval questions (-1 means full dataset)")
    parser.addoption("--num-shots", action="store", type=int, default=5, help="Number of few-shot examples")
    parser.addoption("--data-path", action="store", default=None, help="Path to GSM8K test data (JSONL)")
    parser.addoption("--data-dir", action="store", default=None, help="Directory containing local parquet evaluation datasets")
    parser.addoption("--max-new-tokens", action="store", type=int, default=256, help="Maximum generated tokens per prompt")
    parser.addoption("--num-threads", action="store", type=int, default=16, help="Number of request threads")
    parser.addoption("--baseline-file", action="store", default="data/logprob_baseline_a100.json", help="Path to saved logprob baseline")
    parser.addoption("--update-baseline", action="store_true", help="Generate or overwrite the logprob baseline file")
    parser.addoption("--max-logprob-diff", action="store", type=float, default=1e-4, help="Maximum allowed absolute diff for input/output logprobs")
    parser.addoption("--mmlu-file", action="store", default="data/lukaemon_mmlu_security_studies_ppl_0.json", help="Path to MMLU reference JSON (with prompt and BPB fields)")
    parser.addoption("--mmlu-baseline-file", action="store", default="data/mmlu_logprob_baseline.json", help="Path to save/load per-token logprob baseline for MMLU test")

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test (requires sglang service)")
