"""Fairer GPT-2 KV-cache benchmark.

TTFT includes prompt prefill and is the time until the first token is
available.  The first model calls are deliberately used only as warm-ups so
that PyTorch/CPU initialization is not attributed to one of the variants.
"""

import gc
import os
import time
import tracemalloc
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "openai-community/gpt2"
INPUT_LENGTHS = [16, 128]#(16, 128, 256, 512)
GENERATION_LENGTHS = [10, 100] #(10, 100, 1000)
RANDOM_SEED = 1234
WARMUP_RUNS = 1
WARMUP_GENERATION_LENGTH = 10


def select_device():
    """Use one GPU when available; otherwise fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device):
    """Make asynchronous GPU work visible to the wall-clock timer."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def cache_bytes(cache):
    """Return cache tensor memory across Transformers cache formats."""
    if cache is None:
        return 0

    def tensor_bytes(value):
        if torch.is_tensor(value):
            return value.nelement() * value.element_size()
        if isinstance(value, (tuple, list)):
            return sum(tensor_bytes(item) for item in value)
        if isinstance(value, dict):
            return sum(tensor_bytes(item) for item in value.values())
        return 0

    # Newer Transformers versions expose separate key/value lists.  This is
    # preferable to iterating the cache because cache iteration semantics have
    # changed between Transformers releases.
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        size = tensor_bytes(cache.key_cache) + tensor_bytes(cache.value_cache)
        if size:
            return size

    # Transformers versions using the newer CacheLayer representation keep
    # each layer in `cache.layers`, commonly under `keys` and `values`.
    if hasattr(cache, "layers"):
        size = 0
        for layer in cache.layers:
            names = ("keys", "values") if hasattr(layer, "keys") or hasattr(layer, "values") else ("key", "value")
            for name in names:
                if hasattr(layer, name):
                    size += tensor_bytes(getattr(layer, name))
        if size:
            return size

    # Legacy GPT-2 output: ((key_layer_0, value_layer_0), ...).  Recursion
    # also handles cache containers whose nesting differs across versions.
    return tensor_bytes(cache)


def make_random_input_ids(model, lengths=INPUT_LENGTHS, seed=RANDOM_SEED):
    """Create deterministic random input_ids tensors with the requested lengths."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return {
        length: torch.randint(
            0, model.config.vocab_size, (1, length),
            dtype=torch.long, generator=generator,
        )
        for length in lengths
    }


def process_rss_bytes():
    """Best-effort current RSS; psutil is optional."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss
    except ImportError:
        return None


@dataclass
class Result:
    ttft: float
    tpot: float
    e2e: float
    generated: int
    cache_bytes: int
    peak_python_bytes: int
    rss_delta_bytes: int | None


def run_once(model, input_ids, use_cache, tokens_to_generate, device):
    """Run one complete greedy-generation experiment."""
    gc.collect()
    tracemalloc.start()
    rss_before = process_rss_bytes()
    step_times = []

    with torch.inference_mode():
        synchronize(device)
        start = time.perf_counter()
        outputs = model(input_ids=input_ids, use_cache=use_cache)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        synchronize(device)
        ttft = time.perf_counter() - start
        kv_cache = outputs.past_key_values if use_cache else None

        context = torch.cat((input_ids, next_token), dim=-1)
        for _ in range(tokens_to_generate - 1):
            synchronize(device)
            step_start = time.perf_counter()
            if use_cache:
                outputs = model(
                    input_ids=next_token,
                    past_key_values=kv_cache,
                    use_cache=True,
                )
                kv_cache = outputs.past_key_values
            else:
                outputs = model(input_ids=context, use_cache=False)

            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            if not use_cache:
                context = torch.cat((context, next_token), dim=-1)
            synchronize(device)
            step_times.append(time.perf_counter() - step_start)

    _, peak_python = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = process_rss_bytes()

    # The first token is included in TTFT; the remaining tokens are measured
    # individually. TPOT/ITL is therefore the average of those decode steps.
    decode_time = sum(step_times)
    return Result(
        ttft=ttft,
        tpot=decode_time / len(step_times),
        e2e=ttft + decode_time,
        generated=tokens_to_generate,
        cache_bytes=cache_bytes(kv_cache),
        peak_python_bytes=peak_python,
        rss_delta_bytes=(rss_after - rss_before) if rss_before is not None else None,
    )


def average(results):
    fields = ("ttft", "tpot", "e2e", "cache_bytes", "peak_python_bytes", "rss_delta_bytes")
    values = {}
    for field in fields:
        samples = [getattr(item, field) for item in results]
        values[field] = None if any(value is None for value in samples) else sum(samples) / len(samples)
    values["generated"] = results[0].generated
    return Result(**values)


def print_result(label, result):
    throughput = result.generated / result.e2e
    rss = "unavailable" if result.rss_delta_bytes is None else f"{result.rss_delta_bytes / 1024**2:.2f} MB"
    print(
        f"{label:<18} | TTFT {result.ttft:.4f}s | TPOT/ITL {result.tpot:.4f}s "
        f"| E2E {result.e2e:.4f}s | {throughput:.2f} tok/s | "
        f"KV {result.cache_bytes / 1024**2:.2f} MB | Python peak "
        f"{result.peak_python_bytes / 1024**2:.2f} MB | RSS Δ {rss}"
    )


def main():
    device = select_device()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    inputs_by_length = {
        length: input_ids.to(device)
        for length, input_ids in make_random_input_ids(model).items()
    }

    print(f"Model: {MODEL_ID}")
    print(f"Device: {device}")
    print(f"Input lengths: {INPUT_LENGTHS}")
    print(f"Generation lengths: {GENERATION_LENGTHS}")
    print(f"Warm-up runs per input length: {WARMUP_RUNS} ({WARMUP_GENERATION_LENGTH} tokens)\n")

    for input_ids in inputs_by_length.values():
        for _ in range(WARMUP_RUNS):
            run_once(model, input_ids, True, WARMUP_GENERATION_LENGTH, device)
            run_once(model, input_ids, False, WARMUP_GENERATION_LENGTH, device)

    cached_results = {}
    uncached_results = {}
    for input_length, input_ids in inputs_by_length.items():
        for generation_length in GENERATION_LENGTHS:
            setup = (input_length, generation_length)
            cached_results[setup] = run_once(model, input_ids, True, generation_length, device)
            uncached_results[setup] = run_once(model, input_ids, False, generation_length, device)

    print("Measured results by test setup")
    print("-" * 190)
    for input_length in INPUT_LENGTHS:
        for generation_length in GENERATION_LENGTHS:
            setup = (input_length, generation_length)
            print(f"Input length: {input_length}; generation length: {generation_length}")
            print_result("  WITH KV cache", cached_results[setup])
            print_result("  WITHOUT KV cache", uncached_results[setup])

    print("\nAverage by generation length across input lengths")
    print("-" * 190)
    for generation_length in GENERATION_LENGTHS:
        cached = average([cached_results[(i, generation_length)] for i in INPUT_LENGTHS])
        uncached = average([uncached_results[(i, generation_length)] for i in INPUT_LENGTHS])
        print(f"Generation length: {generation_length}")
        print_result("  WITH KV cache", cached)
        print_result("  WITHOUT KV cache", uncached)

    print("\nAverage by input length across generation lengths")
    print("-" * 190)
    for input_length in INPUT_LENGTHS:
        cached = average([cached_results[(input_length, g)] for g in GENERATION_LENGTHS])
        uncached = average([uncached_results[(input_length, g)] for g in GENERATION_LENGTHS])
        print(f"Input length: {input_length}")
        print_result("  WITH KV cache", cached)
        print_result("  WITHOUT KV cache", uncached)

    # Each combination receives equal weight in this aggregate: four input
    # lengths multiplied by three generation lengths.
    print("\nOverall average across all input/generation combinations")
    print("-" * 190)
    cached = average(list(cached_results.values()))
    uncached = average(list(uncached_results.values()))
    print_result("WITH KV cache", cached)
    print_result("WITHOUT KV cache", uncached)
    print("\nNote: KV memory is the final key/value tensor footprint. RSS Δ is best-effort and requires psutil.")


if __name__ == "__main__":
    main()
