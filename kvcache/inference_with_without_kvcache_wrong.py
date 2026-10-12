import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize setup
model_id = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

# Use a longer prompt to better demonstrate the quadratic cost of skipping the cache
prompt = (
    "The ultimate goal of artificial intelligence is to build highly autonomous systems "
    "that excel at human-like reasoning, adaptive learning, structural problem solving, "
    "and natural language comprehension across all domains of science."
)
inputs = tokenizer(prompt, return_tensors="pt")
initial_input_ids = inputs["input_ids"]
tokens_to_generate = 10

print(f"Prompt length: {initial_input_ids.shape[1]} tokens.\n" + "="*60)

# Helper function to compute memory footprint of cache tensors
def get_cache_memory_bytes(cache_obj):
    # DynamicCache stores keys and values in separate lists, while GPT-2 may
    # return the legacy tuple: ((key_layer_0, value_layer_0), ...).
    if hasattr(cache_obj, "key_cache") and hasattr(cache_obj, "value_cache"):
        layer_caches = zip(cache_obj.key_cache, cache_obj.value_cache)
    else:
        layer_caches = cache_obj

    return sum(
        tensor.nelement() * tensor.element_size()
        for key, value in layer_caches
        for tensor in (key, value)
    )


# =====================================================================
# EXPERIMENT 1: WITH KV CACHE (Token-by-Token)
# =====================================================================
print("\n--- Running Inference WITH KV Cache ---")

total_tokens_generated = 0

# Prefill Stage
start_time = time.perf_counter()
outputs = model(input_ids=initial_input_ids, use_cache=True) #let model auto-create kv cache
ttft_cache = time.perf_counter() - start_time
kv_cache = outputs.past_key_values #keep updated kv values in kv_cache
total_tokens_generated += 1

print(f"At the end of Refill, KV shapes: {kv_cache[0][0].shape}, {kv_cache[0][1].shape}")
print(f"kv cache size: {get_cache_memory_bytes(kv_cache)} @ {total_tokens_generated} tokens generated")



# Get first token
next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
decode_latencies_cache = []

# Decode Stage
for _ in range(tokens_to_generate - 1):
    step_start = time.perf_counter()
    # Pass ONLY the single newest token + the historical cache matrix
    outputs = model(input_ids=next_token_id, past_key_values=kv_cache, use_cache=True)
    kv_cache = outputs.past_key_values
    total_tokens_generated += 1

    print(f"kv cache size: {get_cache_memory_bytes(kv_cache)} @ {total_tokens_generated} tokens generated")

    decode_latencies_cache.append(time.perf_counter() - step_start)
    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

# Metrics calculation
itl_cache = sum(decode_latencies_cache) / len(decode_latencies_cache)  # Inter-Token Latency
e2e_cache = ttft_cache + sum(decode_latencies_cache)
memory_cache_bytes = get_cache_memory_bytes(kv_cache)

# =====================================================================
# EXPERIMENT 2: WITHOUT KV CACHE (Full Re-computation)
# =====================================================================
print("--- Running Inference WITHOUT KV Cache ---")

# Prefill Stage (processing original prompt)
start_time = time.perf_counter()
outputs = model(input_ids=initial_input_ids, use_cache=False)
ttft_no_cache = time.perf_counter() - start_time

# Get first token
next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
# Append token to create a growing context window tensor
rolling_input_ids = torch.cat([initial_input_ids, next_token_id], dim=-1)
decode_latencies_no_cache = []

# Decode Stage
for _ in range(tokens_to_generate - 1):
    step_start = time.perf_counter()
    # CRUCIAL DIFFERENCE: We re-feed the entire sequence length from scratch every time
    outputs = model(input_ids=rolling_input_ids, use_cache=False)
    decode_latencies_no_cache.append(time.perf_counter() - step_start)
    
    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    rolling_input_ids = torch.cat([rolling_input_ids, next_token_id], dim=-1)

# Metrics calculation
itl_no_cache = sum(decode_latencies_no_cache) / len(decode_latencies_no_cache)
e2e_no_cache = ttft_no_cache + sum(decode_latencies_no_cache)
memory_no_cache_bytes = 0  # No cache tensors are preserved in memory after the forward pass

# =====================================================================
# PERFORMANCE ANALYSIS SUMMARY
# =====================================================================
print("\n" + "="*60 + "\nBENCHMARK COMPARISON REPORT\n" + "="*60)
print(f"{'Metric':<32} | {'WITH KV Cache':<15} | {'WITHOUT KV Cache':<15}")
print("-"*68)
print(f"{'Time to First Token (TTFT)':<32} | {ttft_cache:.4f} sec      | {ttft_no_cache:.4f} sec")
print(f"{'Inter-Token Latency (ITL)':<32} | {itl_cache:.4f} sec      | {itl_no_cache:.4f} sec")
print(f"{'End-to-End Latency (E2E)':<32} | {e2e_cache:.4f} sec      | {e2e_no_cache:.4f} sec")
print(f"{'KV Cache Size (Peak)':<32} | {memory_cache_bytes / 1024:.2f} KB     | {memory_no_cache_bytes:.2f} KB")
