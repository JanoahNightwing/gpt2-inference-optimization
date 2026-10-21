# gpt2-inference-optimization
Benchmarking and optimizing GPT-2 inference on a T4 GPU: fp16, KV cache, and torch.compile, with throughput, memory, and perplexity results.
## Overview

I measured how fast GPT-2 small (124M parameters) generates text on a Colab T4 GPU,
then tested optimizations one at a time. So far: half precision (fp16) and the KV cache.

## Setup

- Model: GPT-2 small (Hugging Face), greedy decoding
- Prompt: 5 words, generating 300 new tokens
- 10 warm-up runs, then 10 timed runs, with `torch.cuda.synchronize()` before each timer read
- Peak memory is reset before each measurement

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

## Findings

- **fp16 roughly halves memory** (533 MB to 280 MB at batch 1).
- **The KV cache gives a large speedup at batch 32**: about 21x in fp32 and 7x in fp16.
  Without it, work per token grows with sequence length.
- **Decode is overhead-bound, not compute-bound.** With fp16 and the cache on, each step
  takes about 11 ms at batch 1, 8, and 32, even though batch 32 does 32x more math.
  The GPU is waiting on Python and kernel-launch overhead, so fp16 barely helps speed.
  This motivates `torch.compile`.
- TTFT is about 12-15 ms everywhere because the prompt is only 5 tokens, so it is not informative here.

## Mistakes I fixed along the way

- My first run had a warm-up outlier, so I increased warm-up from 3 to 10 runs.
- I originally didn't reset the peak memory counter, which made fp16 memory look identical to fp32.

## Next steps

- Perplexity on WikiText-2 to confirm optimizations don't hurt quality
- `torch.compile`
- INT8 quantization
- Longer prompts to measure prefill cost
