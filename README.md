# Inference fundamentals
This repo contains code to help understand fundamentals of LLM inference, including KV cache, prefill and decode, etc., in a gentle way, without diving too deep into how Transformers work (Read the "Attention Is All You Need" paper thoroughly, and sketch out the pipeline with pencil and paper if you need a refresher).
* **kvcache** - Using a 124M-parameter community version of OpenAI GPT2 model to demonstrate token generation with and without KV cache. Showing impact of KV cache on key inference metrics such as TTFT, ITL and E2E Latency. My approach: first I use an inaccurately implemented code to draw partially correct and partially incorrect conclusions on KV cache's effectiveness; then, I point out where the issue lies and provide a correct implementation to re-examine inference with KV cache where the wrong conclusions were previously drawn.

## Key Concepts to Study

### KV Cache
A buffer that caches the Key and Value projections of past tokens during autoregressive decoding, enabling each new token to attend to the full history without re-processing it.

When a KV cache is used, inference proceeds in two distinct stages: prefill, and decode.

During the prefill phase, the model processes the entire input prompt in one pass and computes K and V vectors for every token. Instead of discarding them, the model stores them in the KV cache.

During the decode phase, generating each new token requires attention over all previous tokens. Without a cache, the model would recompute K and V for the entire prefix at every step—an $O(n^2)$ operation per token, where $n$ is the length of the current input token sequence. With the cache, the model only computes K and V for the single new token that was generated at the previous step, appends them to the cache, and reads the rest—an O(n) operation.
```text
The Trade-Off
Aspect	Impact
---------------
Speed	Dramatically faster decoding (linear instead of quadratic scaling)
Memory	Grows linearly with batch size × layers × heads x sequence length
Bottleneck	Decode becomes memory-bandwidth-bound, not compute-bound
```

So, KV Cache trades memory for speed: it eliminates redundant computation during decoding, making real-time generation feasible, but its growing memory footprint becomes the primary bottleneck for long-context and high-throughput serving.


### Prefill
Prefill Phase: This is the initial processing stage where the model ingests the entire input prompt (all its tokens) in a single parallel pass. During this phase, it computes and stores the Key (K) and Value (V) vectors in KV cache for every token in the prompt. This phase is compute-bound because it involves large, dense matrix multiplications. The time it takes directly determines the Time to First Token (TTFT) .

### Decode
Decode Phase: This is the autoregressive generation stage where the model produces the output sequence one token at a time (except the first token, which is produced in Prefill phase). For each new token, it computes K and V only for that single new token, appends them to the KV cache, and then uses the current token's Query (Q) to attend over all cached K and V vectors. This phase is memory-bound because the dominant cost is reading the large KV cache and model weights from memory for relatively little computation. The speed of this phase determines the Inter-Token Latency (ITL) or Time Per Output Token (TPOT) .

## Key Inference Metrics

Here is a list of the key inference-time metrics for LLMs.
```text
Key LLM Inference Metrics
Metric  Definition  Unit  Primary Phase
------------------------------------------------
TTFT (Time to First Token)	Delay from request submission to the arrival of the first output token. Dominated by the prefill phase.	ms,s	Prefill

ITL (Inter-Token Latency)	Average time between consecutive output tokens during generation. Also called TPOT.	ms/token	Decode

TPOT (Time Per Output Token)	Same as ITL; the average time to generate each subsequent token after the first.	ms/token	Decode

E2E Latency (End-to-End Latency)	Total time from request submission to the final output token. Approx. TTFT + (output_tokens − 1) × ITL.	ms,s	Both

TPS (Tokens Per Second)	Rate at which tokens are generated after the first. The inverse of ITL.	tokens/s	Decode

Throughput	Total tokens or requests processed per second across all concurrent users.	tokens/s or req/s	Both

QPS (Queries Per Second)	Number of complete requests served per second under a given latency constraint.	req/s	Both

Scheduler Queue Time	Time a request waits in the scheduler queue before processing begins. Indicates system load.	ms,s	Prefill

Cold Start Latency	Time from process launch to the first successful response, including all initialization costs.	s	Startup

Memory Footprint	Peak Memory Usage	Maximum GPU/CPU memory consumed during inference, including model weights and KV cache.	GB	Both

KV Cache Size	Memory occupied by cached K and V tensors, growing linearly with sequence length and batch size.	GB	Decode

Cost per Token	Monetary Cost per Token	Monetary cost of generating one token, accounting for hardware and energy.	$/token	Both
```
TTFT, ITL, E2E Latency, TPS, and Memory Footprint of KV Cache are what we study in this repo.

## openai-community/gpt2

This is the model used in this repo for token generation. It is the smallest version of GPT2 model, with 124M parameters. It is available on Hugging Face here:
[openai-community/gpt2](https://huggingface.co/openai-community/gpt2)

```text
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D()
          (c_proj): Conv1D()
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D()
          (c_proj): Conv1D()
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)

```
## Project Structure
```text
.
├── README.md
├── kvcache
│   ├── __init__.py
│   ├── inference_with_without_kvcache_fixed.html   #Correct measurement (exported from a run on Google Colab)
│   ├── inference_with_without_kvcache_fixed.ipynb  #Correct measurement (downloaded from Google Colab)
│   ├── inference_with_without_kvcache_fixed.py     #Correct measurement
│   └── inference_with_without_kvcache_wrong.py     #Wrong measurement
├── pyproject.toml
└── uv.lock
```


## How to Run
### Clone to Local

`git clone https://github.com/lichtjiang/inference-fundamentals.git`

### Dependency Installation
`cd inference-fundamentals`

`uv sync`

### Run Experiments

You can run a script in one of the following ways:

`uv run kvcache/<a_script>.py`

`uv run -m kvcache.<a_script>`

**Note**: 

You may need to upgrade dependencies if you see warnings like the following when you run scripts in this repo (these may or may not cause experiments to fail though):

```text
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.5.3 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
```

#### Recommended Steps
##### Step 1

Take a look at `inference_with_without_kvcache_wrong.py`, which generates 10 new tokens twice, once with KV Cache and once without KV Cache.

A little more detail. This script first enables KV Cache, then generates the first token. This is prefill.

Then the script repeatedly generate one new token at a time until another 9 tokens are generated. This is decode stage.

After all 10 tokens are generated, the script disables KV Cache and starts to generate another 10 tokens.

***You may have spotted issues in this script. But I'll get back to that later.***

Next, run the script.

You may see an output similar to the following:

```text
Prompt length: 37 tokens.
============================================================

--- Running Inference WITH KV Cache ---
At the end of Refill, KV shapes: torch.Size([1, 12, 37, 64]), torch.Size([1, 12, 37, 64])
kv cache size: 2727936 @ 1 tokens generated
kv cache size: 2801664 @ 2 tokens generated
kv cache size: 2875392 @ 3 tokens generated
kv cache size: 2949120 @ 4 tokens generated
kv cache size: 3022848 @ 5 tokens generated
kv cache size: 3096576 @ 6 tokens generated
kv cache size: 3170304 @ 7 tokens generated
kv cache size: 3244032 @ 8 tokens generated
kv cache size: 3317760 @ 9 tokens generated
kv cache size: 3391488 @ 10 tokens generated
--- Running Inference WITHOUT KV Cache ---

============================================================
BENCHMARK COMPARISON REPORT
============================================================
Metric                           | WITH KV Cache   | WITHOUT KV Cache
--------------------------------------------------------------------
Time to First Token (TTFT)       | 1.2312 sec      | 0.0849 sec
Inter-Token Latency (ITL)        | 0.0426 sec      | 0.0912 sec
End-to-End Latency (E2E)         | 1.6148 sec      | 0.9054 sec
KV Cache Size (Peak)             | 3312.00 KB     | 0.00 KB
```
This tells us a few things about inference:

-  TTFT with KV Cache is longer than that without KV Cache (`1.2312 sec w/ KV cache vs 0.0849 sec w/o KV cache`). (***tip***: this is a false conclusion. I will come back to this.)
- Generation of each successive token after the first one takes less time using KV Cache than without using KV Cache.
- KV Cache consumes extra memory.
- E2E Latency is longer with KV Cache than no kV Cache is used. (***tip***: this is another false claimation.)

A quick check will reveal that each new token causes KV Cache to increase by `73728` bytes. This is how the number is obtained: 

`12 (layers) x 12 (heads/layer)  x 64 (head-dimension) x 2 (K & V) x 4 (byes)`

To understand the above calculation, we know that the small GPT2 model uses:

  - 12 transformer layers
  - 12 attention heads per layer
  - 64 dimensions per head
  - float32, which uses 4 bytes per element

For each newly generated token, the model stores one new key vector and one new value vector per head in every layer. Therefore, each token adds:

```text
  12 × 12 × 64 × 2 = 18,432 float32 values
  18,432 × 4 bytes = 73,728 bytes
```

The cache tensors typically have shapes like:

```text
  [batch_size, num_heads, sequence_length, head_dim]
  [1, 12, sequence_length, 64]
```

Both keys and values grow along sequence_length, so the cache grows linearly by 73,728 bytes per token.

Total consumed memory by KV Cache is thus: `12 x 12 x 64 x 2 x 4 x (37+9)/1024 = 3312KB` (see "WITH KV Cache" column of the last row from above table).

##### Step 2
Is E2E Latency with KV Cache really bigger than inference without using KV Cache?

First, update the script `inference_with_without_kvcache_wrong.py` to generate 100 tokens by changing line 18 from `tokens_to_generate = 10` to `tokens_to_generate = 100`

This is metrics summary:

```text
============================================================
BENCHMARK COMPARISON REPORT
============================================================
Metric                           | WITH KV Cache   | WITHOUT KV Cache
--------------------------------------------------------------------
Time to First Token (TTFT)       | 1.2690 sec      | 0.0975 sec
Inter-Token Latency (ITL)        | 0.0521 sec      | 0.1701 sec
End-to-End Latency (E2E)         | 6.4244 sec      | 16.9410 sec
KV Cache Size (Peak)             | 9792.00 KB     | 0.00 KB
```

With KV cache, TTFT (1.2690 sec) is still bigger than that without KV cache (0.0975 sec).

But E2E Latency is much lower when KV cache is used (6.4244 sec), less than half of that without KV cache (16.9410 sec).

So, we may confirm that KV cache makes TTFT longer and then arrive at this updated conclusion:

- For short responses (10 output tokens), E2E Latency is longer with KV Cache than no kV Cache is used.
- For long responses (100 output tokens), E2E Latency is greatly reduced with KV cache, compared to inference with no KV cache.

***Tip***: this is still misleading!

##### Step 3
Does KV cache really negatively impact TTFT regardless of output length? And What about KV cache's effect on E2E Latency with respect to output length?

To answer these questions, I do the following things.

First, I reverse the order of generation with and without KV Cache. So, this time, I perform generation without KV Cache first. The following is metrics output:


```text
============================================================
BENCHMARK COMPARISON REPORT
============================================================
Metric                           | WITH KV Cache   | WITHOUT KV Cache
--------------------------------------------------------------------
Time to First Token (TTFT)       | 0.0901 sec      | 1.4546 sec
Inter-Token Latency (ITL)        | 0.0464 sec      | 0.0887 sec
End-to-End Latency (E2E)         | 0.5079 sec      | 2.2525 sec
KV Cache Size (Peak)             | 3312.00 KB     | 0.00 KB

```
As can be seen, now TTFT of inference without KV Cache and that with KV Cache swapped from previous experiment: 1.4546 sec vs 0.0901 sec. TTFT with KV Cache beats no KV Cache!

What happened? Why is that?

It seems that the first inference always spends longer in generating the first token, regardless of whether KV Cache is used. But this latency is a result of multiple operations than just a pure generation of the first token.

So, we need to consider the following costs during a cold start before a model is even starting to generate the very first token:

- Lazy Initialization of Libraries such as PyTorch, TensorFlow, and JAX, the cost of which can be orders of magnitude larger than the actual inference time.
- OS and Runtime Effects to address operations such as page faults, CPU cache warming, and framework thread pool initializations, etc.
- Autotuning and Algorithm Selection by libraries like cuBLAS, cuDNN, and MPSGraph, etc.
- KV Cache allocation itself.
- etc.

Due to this, we have to take cost of such cold start out of the equation regardless whether we do inference with or without KV Cache in the first place.

To this goal, we have to perform something called "warmup" before starting to serve real generation requests and performance measurement.

Simply speaking, "warmup" is therefore a deliberate, throwaway phase of execution that runs before we start measuring. Its purpose is to eliminate one-time costs that would otherwise pollute our benchmark numbers. 

What this means is that we run the model one or more times without recording any metrics, then discarding those runs. Only after the model is "warmed up" do we start the actual timed measurement. In frameworks like vLLM, TensorRT-LLM, and MLPerf, warmup is a formal part of the benchmark protocol.

Now, how do we warmup the system?

Empirically, we run a few requests (e.g. 5 ~ 50) before measurement. A principled approach is to run warmup iterations until the measurements converge statistically, then discard everything before that point. For example, we wait for the coefficient of variation (CV) of TTFT to drop below 5% across a rolling window of 30 requests. If the gate doesn't pass within a time window, e.g. 60 seconds, the system records a warning and proceeds anyway.

##### Step 4
With understanding of what contributes to the latency of seeing the first token, it is obvious that we need a warmup before tracking metrics in our code for measurement. That's exactly what the script `inference_with_without_kvcache_fixed.py` does. The script warms up the system, then performs multiple inference runs with and without KV Cache, against a combination of different lengths of input and output.

Here's the results of one run on Google Colab using T4 GPU:

```text
Model: openai-community/gpt2
Device: cuda:0
Input lengths: [16, 128]
Generation lengths: [10, 500, 800]
Warm-up runs per input length: 1 (10 tokens)

Measured results by test setup
------------------------------
Input length: 16; generation length: 10
  WITH KV cache    | TTFT 0.0360s | TPOT/ITL 0.0299s | E2E 0.3049s | 32.80 tok/s | KV 1.76 MB | Python peak 0.04 MB | RSS Δ 0.03 MB
  WITHOUT KV cache | TTFT 0.0174s | TPOT/ITL 0.0170s | E2E 0.1705s | 58.64 tok/s | KV 0.00 MB | Python peak 0.04 MB | RSS Δ 0.01 MB

Input length: 16; generation length: 500
  WITH KV cache    | TTFT 0.0191s | TPOT/ITL 0.0173s | E2E 8.6390s | 57.88 tok/s | KV 36.21 MB | Python peak 0.38 MB | RSS Δ 0.63 MB
  WITHOUT KV cache | TTFT 0.0177s | TPOT/ITL 0.0277s | E2E 13.8319s | 36.15 tok/s | KV 0.00 MB | Python peak 0.36 MB | RSS Δ 1.03 MB

Input length: 16; generation length: 800
  WITH KV cache    | TTFT 0.0313s | TPOT/ITL 0.0193s | E2E 15.4640s | 51.73 tok/s | KV 57.30 MB | Python peak 0.38 MB | RSS Δ 0.20 MB
  WITHOUT KV cache | TTFT 0.0187s | TPOT/ITL 0.0371s | E2E 29.6581s | 26.97 tok/s | KV 0.00 MB | Python peak 0.37 MB | RSS Δ 0.59 MB

Input length: 128; generation length: 10
  WITH KV cache    | TTFT 0.0214s | TPOT/ITL 0.0187s | E2E 0.1900s | 52.63 tok/s | KV 9.63 MB | Python peak 0.04 MB | RSS Δ 0.01 MB
  WITHOUT KV cache | TTFT 0.0189s | TPOT/ITL 0.0189s | E2E 0.1888s | 52.96 tok/s | KV 0.00 MB | Python peak 0.04 MB | RSS Δ 0.00 MB

Input length: 128; generation length: 500
  WITH KV cache    | TTFT 0.0196s | TPOT/ITL 0.0191s | E2E 9.5447s | 52.39 tok/s | KV 44.09 MB | Python peak 0.37 MB | RSS Δ 0.19 MB
  WITHOUT KV cache | TTFT 0.0206s | TPOT/ITL 0.0355s | E2E 17.7254s | 28.21 tok/s | KV 0.00 MB | Python peak 0.36 MB | RSS Δ 0.01 MB

Input length: 128; generation length: 800
  WITH KV cache    | TTFT 0.0209s | TPOT/ITL 0.0194s | E2E 15.4834s | 51.67 tok/s | KV 65.18 MB | Python peak 0.38 MB | RSS Δ 0.01 MB
  WITHOUT KV cache | TTFT 0.0207s | TPOT/ITL 0.0461s | E2E 36.8582s | 21.70 tok/s | KV 0.00 MB | Python peak 0.37 MB | RSS Δ 0.04 MB

Average by generation length across input lengths
-------------------------------------------------
Generation length: 10
  WITH KV cache    | TTFT 0.0287s | TPOT/ITL 0.0243s | E2E 0.2474s | 40.41 tok/s | KV 5.70 MB | Python peak 0.04 MB | RSS Δ 0.02 MB
  WITHOUT KV cache | TTFT 0.0182s | TPOT/ITL 0.0179s | E2E 0.1797s | 55.65 tok/s | KV 0.00 MB | Python peak 0.04 MB | RSS Δ 0.00 MB

Generation length: 500
  WITH KV cache    | TTFT 0.0193s | TPOT/ITL 0.0182s | E2E 9.0918s | 54.99 tok/s | KV 40.15 MB | Python peak 0.37 MB | RSS Δ 0.41 MB
  WITHOUT KV cache | TTFT 0.0191s | TPOT/ITL 0.0316s | E2E 15.7786s | 31.69 tok/s | KV 0.00 MB | Python peak 0.36 MB | RSS Δ 0.52 MB

Generation length: 800
  WITH KV cache    | TTFT 0.0261s | TPOT/ITL 0.0193s | E2E 15.4737s | 51.70 tok/s | KV 61.24 MB | Python peak 0.38 MB | RSS Δ 0.10 MB
  WITHOUT KV cache | TTFT 0.0197s | TPOT/ITL 0.0416s | E2E 33.2581s | 24.05 tok/s | KV 0.00 MB | Python peak 0.37 MB | RSS Δ 0.31 MB

Average by input length across generation lengths
-------------------------------------------------
Input length: 16
  WITH KV cache    | TTFT 0.0288s | TPOT/ITL 0.0222s | E2E 8.1359s | 1.23 tok/s | KV 31.76 MB | Python peak 0.27 MB | RSS Δ 0.29 MB
  WITHOUT KV cache | TTFT 0.0180s | TPOT/ITL 0.0273s | E2E 14.5535s | 0.69 tok/s | KV 0.00 MB | Python peak 0.26 MB | RSS Δ 0.54 MB

Input length: 128
  WITH KV cache    | TTFT 0.0206s | TPOT/ITL 0.0191s | E2E 8.4060s | 1.19 tok/s | KV 39.63 MB | Python peak 0.26 MB | RSS Δ 0.07 MB
  WITHOUT KV cache | TTFT 0.0201s | TPOT/ITL 0.0335s | E2E 18.2575s | 0.55 tok/s | KV 0.00 MB | Python peak 0.26 MB | RSS Δ 0.02 MB

Overall average across all input/generation combinations
--------------------------------------------------------
WITH KV cache      | TTFT 0.0247s | TPOT/ITL 0.0206s | E2E 8.2710s | 1.21 tok/s | KV 35.70 MB | Python peak 0.26 MB | RSS Δ 0.18 MB
WITHOUT KV cache   | TTFT 0.0190s | TPOT/ITL 0.0304s | E2E 16.4055s | 0.61 tok/s | KV 0.00 MB | Python peak 0.26 MB | RSS Δ 0.28 MB

Note: KV memory is the final key/value tensor footprint. RSS Δ is best-effort and requires psutil.
```

From above report, we can draw final conclusions as follows:

- TTFT is nearly identical with and without KV Cache: e.g., 0.0247s with KV Cache vs 0.0190s without KV Cache. This is expected because both methods must process the entire input prompt once during prefill. KV caching mainly helps during subsequent token generation.
- TTFT does not vary with changes in generation length.
- TPOT/ITL does not vary much with increase in generation length when KV Cache is used.
- TOPT/ITL increases sublinearly with increase in generation length when no KV Cache is used.
- TPOT/ITL is the clearest benefit of KV caching when the number of tokens to generate gets longer:

      - KV cache: 0.0299 s/token for generating 10 tokens
      - No cache: 0.0170 s/token for generating 10 tokens
      - KV cache: 0.0173 s/token for generating 500 tokens
      - No cache: 0.0277 s/token for generating 500 tokens
      - KV cache: 0.0193 s/token for generating 800 tokens
      - No cache: 0.0371 s/token for generating 800 tokens      

    Without a KV Cache, the model repeatedly recomputes attention over the entire growing sequence. With a cache, each new token attends to previously computed K/V tensors. The gap generally becomes larger for longer prompts and longer generations because the no-cache method performs increasingly more redundant computation, E.g. TPOT/ITL without KV Cache is ~2x over that with KV Cache, when input length is 16 and output length is 800. 
    
    Benifiting from a lower TPOT/ITL, TPS with KV Cache is seen with a similar pattern, e.g. 51.67 tok/s with KV Cache vs 21.70 tok/s without KV Cache, about 1.5 time gain.

- E2E latency is substantially lower with caching:
  - KV cache: ~15s (input length: 128, output length: 800)
  - No cache: ~37s (input length: 128, output length: 800)

  E2E latency includes the initial TTFT plus all decode steps, so the improvement accumulates over the generated sequence.

- KV caching requires persistent memory. In this run, the final KV tensors occupy approximately 65.18 MB when input length is 128 and output length is 800 (calculation: `12 (layers) * 12 (headers) * 64 (dimension per head) * 2 (K&V) * 4 (bytes) * 927 (total tokens - 1) /1024/1024`). The no-cache version retains no KV tensors, so its reported cache size is zero.

- The memory tradeoff is therefore:
  - KV cache: more memory, lower generation latency
  - No cache: less retained cache memory, much higher computation and latency

- RSS is a noisy process-level measurement. Python garbage collection, PyTorch’s allocator, and memory reuse can cause the process RSS at the end of a run to be lower than at the beginning (though not seen in above run of our experiment).

- Python peak is very small because it measures Python-managed allocations, not the native memory used by PyTorch tensors. The model weights, activations, and KV tensors are mostly allocated outside the Python heap, so this metric should not be interpreted as total inference memory.

- The reported values under "Overall average across all input/generation combinations" are averages over multiple input/output-length combinations in our experiments. They summarize overall behavior but hide important variation. For learning purpose, the per-configuration results are more informative when showing how the cache advantage changes with prompt length and generated-token count.

