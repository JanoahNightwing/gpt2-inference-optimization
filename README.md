# gpt2-inference-optimization

Benchmarking and optimizing GPT-2 inference on a T4 GPU: fp16, KV cache, torch.compile, and INT8 quantization, with throughput, memory, and perplexity results.

## Overview

I measured how fast GPT-2 small (124M parameters) generates text on a Colab T4 GPU, then tested optimizations one at a time and checked that each one didn't hurt output quality. The main finding is that decoding at small batch sizes is limited by CPU and kernel-launch overhead, not GPU math, and the profiler data below backs that up.

## Setup

- Model: GPT-2 small (Hugging Face), greedy decoding, one fixed 5-word prompt unless noted
- Main benchmark: 300 new tokens, 10 warm-up runs, then 10 timed runs, with `torch.cuda.synchronize()` before each timer read
- Peak memory is reset before each measurement
- Hardware: Google Colab T4 GPU

## Results

| config | batch | TTFT (ms) | tokens/sec | peak mem (MB) |
|---|---|---|---|---|
| fp32, cache on | 1 | 13.0 | 94 | 533 |
| fp32, cache on | 8 | 13.6 | 687 | 701 |
| fp32, cache on | 32 | 14.8 | 2173 | 1270 |
| fp32, cache off | 1 | 11.2 | 72 | 530 |
| fp32, cache off | 8 | 12.5 | 101 | 677 |
| fp32, cache off | 32 | 16.0 | 105 | 1179 |
| fp16, cache on | 1 | 14.4 | 91 | 280 |
| fp16, cache on | 8 | 12.6 | 720 | 370 |
| fp16, cache on | 32 | 13.2 | 2864 | 662 |
| fp16, cache off | 1 | 11.9 | 95 | 278 |
| fp16, cache off | 8 | 12.3 | 372 | 354 |
| fp16, cache off | 32 | 12.1 | 415 | 614 |
| fp16, static cache, torch.compile | 1 | 14.7 | 176 | 534 |
| fp16, static cache, torch.compile | 8 | 14.8 | 2249 | 622 |
| fp16, static cache, torch.compile | 32 | 15.8 | 5410 | 913 |

## Quality check (perplexity)

WikiText-2 test set, non-overlapping 1024-token chunks (lower is better):

| precision | perplexity |
|---|---|
| fp32 | 29.95 |
| fp16 | 29.96 |
| int8 (bitsandbytes) | 30.11 |

fp16 matches fp32 (difference of 0.01) at half the memory. Values are a bit higher than sliding-window numbers reported elsewhere because of the chunking.

## Long prompt (prefill)

Same setup, but with a ~500-token prompt, 100 new tokens, eager mode (no compile).

| config | prompt | TTFT (ms) | tokens/sec | peak mem (MB) |
|---|---|---|---|---|
| fp32, batch 1 | ~500 tok | 39.1 | 61 | 588 |
| fp32, batch 8 | ~500 tok | 290.6 | 494 | 1077 |
| fp16, batch 1 | ~500 tok | 14.1 | 66 | 301 |
| fp16, batch 8 | ~500 tok | 66.4 | 728 | 554 |

## Profiling

I profiled fp16 decoding at batch 1 (50 new tokens, cache on) with `torch.profiler`.

- Total GPU kernel time was about 140 ms, while total CPU time was about 558 ms. The profiler adds its own overhead to wall-clock time, so I used its CPU total as the cross-check: it lines up with the roughly 550 ms implied by the unprofiled benchmark (91 tokens/sec).
- That works out to about 2.8 ms of GPU work per token against roughly 11 ms per token end to end, so the GPU was busy about a quarter of the time.
- The work is spread across many tiny kernels per token: 48 matrix multiplies (`addmm`), 25 layer norms, and 12 attention calls. Each matrix multiply took about 25 µs on the GPU but about 57 µs of CPU time to launch.
- One large kernel (about 437 µs per call, once per token) accounts for about 16% of GPU time. It is likely the output projection onto the vocabulary, but I did not verify that.

## INT8 quantization

Loaded with bitsandbytes 8-bit. These rows use a lighter setup than the main benchmark (100 new tokens, 1 warm-up run, 3 timed runs) because INT8 was very slow to benchmark. The fp16 reference rows come from an earlier run with the same 100-token length but 3 warm-up runs, so treat the comparison as approximate.

| config | batch | TTFT (ms) | tokens/sec | peak mem (MB) |
|---|---|---|---|---|
| fp16 (reference) | 1 | 12.8 | 91 | 273 |
| int8 (bitsandbytes) | 1 | 37.4 | 34 | 180 |
| fp16 (reference) | 8 | 14.4 | 716 | 313 |
| int8 (bitsandbytes) | 8 | 40.1 | 234 | 220 |

INT8 cut peak memory by about 30 to 34% and raised perplexity by 0.15 (about 0.5%), but it was roughly 2.7x to 3.1x slower than fp16. This is likely because the bitsandbytes INT8 path adds extra work around each matrix multiply, and GPT-2 small is too small for the memory savings to pay that cost back. I did not profile the INT8 path to confirm this.

## Findings

- **fp16 roughly halves memory** (533 MB to 280 MB at batch 1) with no measurable quality loss.
- **The KV cache gives a large speedup at batch 32**: about 21x in fp32 and 7x in fp16. Without it, work per token grows with sequence length.
- **Decode is overhead-bound, not compute-bound.** With fp16 and the cache on, each step takes about 11 ms at batch 1, 8, and 32, even though batch 32 does 32x more math. The profiler shows the GPU idle about three quarters of the time, waiting on many small kernel launches. This is why fp16 barely helps decode speed.
- **torch.compile gave 1.9x to 3.1x higher throughput** than eager fp16 (91 to 176 tokens/sec at batch 1, 720 to 2249 at batch 8, 2864 to 5410 at batch 32). That is about 5.7 ms per step at batch 1, down from about 11 ms. `mode="reduce-overhead"` (CUDA graphs) crashed because of the in-place static KV cache update, and the working version skipped CUDA graphs, so the gain comes from kernel fusion reducing launch overhead, not from graph replay. Peak memory rose (280 to 534 MB at batch 1), likely because the static cache is preallocated.
- **Correctness check:** the compiled model generated token-for-token identical output to the eager fp16 model on a 50-token greedy generation.
- **Prefill is compute-bound, so fp16 helps a lot there.** With a ~500-token prompt, fp16 cut time to first token by about 2.8x at batch 1 (39 to 14 ms) and about 4.4x at batch 8 (291 to 66 ms), versus fp32. At batch 1 the fp16 TTFT is still near the per-call overhead floor, so the gap is smaller than the compute alone would suggest.
- **INT8 saved memory but lost speed** on this model (see the INT8 section).
- TTFT is about 12 to 15 ms in the main benchmark because the prompt is only 5 tokens, so it is not informative there.

## Limitations

- One GPU (T4), one small model, and one prompt for the main benchmark, so absolute numbers will differ on other hardware.
- The INT8 comparison uses a lighter benchmark setup than the main table, as noted above.
- Perplexity uses non-overlapping 1024-token chunks, so values are not directly comparable to published sliding-window numbers.
- The correctness check covers one prompt and 50 tokens.
- I did not profile the compiled or INT8 paths, so the explanations for their memory and speed behavior are likely causes, not confirmed ones.

## Mistakes I fixed along the way

- My first run had a warm-up outlier, so I increased warm-up from 3 to 10 runs.
- I originally didn't reset the peak memory counter, which made fp16 memory look identical to fp32.
- `torch.compile` with `mode="reduce-overhead"` crashed on the static KV cache, so I switched to the default compile path.
- My first INT8 benchmark used the same heavy settings as the other configs and stalled for several minutes, so I reduced the token count and number of runs for that row.

## How to run

Open the notebook in Google Colab with a T4 GPU runtime and run the cells in order.

## Next steps

- Write a custom Triton kernel (for example a fused LayerNorm + residual add) and benchmark it against PyTorch
- Get CUDA graphs working with the static cache
- Profile the compiled and INT8 paths to confirm why they behave the way they do
