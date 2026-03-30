"""
Test logits retrieval from SGLang server.

Usage:
python3 -m sglang.test.get_logits_ut --num-prompts 10 --host http://127.0.0.1 --port 30000
"""

import argparse
import time
import numpy as np

from sglang.lang.api import set_default_backend
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

def calculate_ppl(logprobs):
    """Calculate PPL from a list of logprobs [[val, id, ...], ...]"""
    if not logprobs:
        return 0.0
    valid_logprobs = [lp[0] for lp in logprobs if lp is not None and lp[0] is not None]
    if not valid_logprobs:
        return 0.0
    return np.exp(-np.mean(valid_logprobs))

def run_logits_test(args):
    """Get logits from SGLang server and return results."""
    
    # Set backend endpoint
    set_default_backend(RuntimeEndpoint(f"{args.host}:{args.port}"))
    
    # Prepare test prompts
    prompts = [
        # "What is 2 + 2?",
        # "Translate 'hello' to French:",
        # "Write a short poem about spring.",
        # "What is the capital of France?",
        # "Explain photosynthesis.",
        # "Who wrote Romeo and Juliet?",
        # "What is the square root of 144?",
        # "List three colors:",
        # "Define quantum computing.",
        # "What year was Python created?",
        "Please provide a very detailed and comprehensive explanation of the history, evolution, and future prospects of artificial intelligence, including its impact on various industries such as healthcare, finance, and transportation. Ensure the response is informative and covers multiple perspectives. Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.",
        # "Could you describe the process of cellular respiration in great detail, including the stages of glycolysis, the Krebs cycle, and the electron transport chain? Explain how energy is transferred and stored within the cell, and the role of ATP. Cellular respiration is a set of metabolic reactions and processes that take place in the cells of organisms to convert biochemical energy from nutrients into adenosine triphosphate (ATP), and then release waste products.",
    ]
    
    prompts = prompts[:args.num_prompts]
    
    import sglang as sgl
    
    @sgl.function
    def get_logits(s, prompt):
        """Generate text and get logits."""
        s += prompt
        s += sgl.gen(
            "output",
            max_tokens=args.max_tokens,
            return_logprob=True,
            logprob_start_len=0,
        )
    
    # Prepare batch arguments
    arguments = [{"prompt": p} for p in prompts]
    
    # Run batch inference
    print(f"Running inference on {len(prompts)} prompts...")
    tic = time.perf_counter()
    states = get_logits.run_batch(
        arguments,
        temperature=args.temperature,
        num_threads=args.parallel,
        progress_bar=True,
    )
    latency = time.perf_counter() - tic
    
    # Extract and display logits
    results = []
    for i, state in enumerate(states):
        output = state["output"]
        meta_info = state.get_meta_info("output")
        
        if meta_info:
            input_logprobs = meta_info.get("input_token_logprobs")
            output_logprobs = meta_info.get("output_token_logprobs")
            input_ppl = calculate_ppl(input_logprobs)
            output_ppl = calculate_ppl(output_logprobs)

        result_item = {
            "prompt": prompts[i],
            "output": output,
            "tokens_count": meta_info.get("completion_tokens") if meta_info else 0,
            "input_ppl": input_logprobs,
            "output_ppl": output_logprobs,
        }
        results.append(result_item)
        
        print(f"\n--- Result {i+1} ---")
        print(f"Prompt: {prompts[i]}")
        print(f"Output: {output}")
        print(f"Tokens: {result_item['tokens_count']}")
        print(f"Input PPL:  {input_ppl:.4f} (tokens: {len(input_logprobs)})")
        print(f"Output PPL: {output_ppl:.4f} (tokens: {len(output_logprobs)})")
                
    
    # Calculate metrics
    total_tokens = sum(r["tokens_count"] for r in results)
    output_throughput = total_tokens / latency if latency > 0 else 0
    
    # Print summary
    print("\n" + "=" * 80)
    print("Logits Retrieval Test Summary")
    print("=" * 80)
    print(f"Number of prompts: {len(prompts)}")
    print(f"Total output tokens: {total_tokens}")
    print(f"Total latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")
    print("=" * 80)
    
    return {
        "num_prompts": len(prompts),
        "total_tokens": total_tokens,
        "latency": latency,
        "output_throughput": output_throughput,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test logits retrieval from SGLang server")
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    
    args = parser.parse_args()
    run_logits_test(args)
