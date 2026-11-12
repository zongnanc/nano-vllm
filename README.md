<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Chunked prefill mixed with decode, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Scheduling

Each step first gives every running sequence its next token, or the next chunk of a prompt that is still being prefilled. It then spends what is left of `max_num_batched_tokens` prefilling waiting prompts, cutting the last one into a chunk if it doesn't fit. Prefill and decode share one forward pass, so running sequences keep generating while new requests are prefilled. Steps that only decode use CUDA graphs.

A smaller `max_num_batched_tokens` keeps inter-token latency low while prompts are being prefilled. A larger one prefills faster.

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |

`bench.py` sends every request at once. To measure serving latency, `bench_online.py` runs the same workload with requests arriving over time and reports time to first token (TTFT) and inter-token latency (ITL):
```bash
python bench_online.py --request-rate 4    # Poisson arrivals, requests/s
```

**Test Configuration:**
- Hardware: NVIDIA A10G (24GB)
- Model: Qwen3-0.6B
- Workload: same as above, `max_num_batched_tokens=16384`
- Baseline: prefill-first scheduler (upstream `bb823b3`); Mixed: prefill and decode in the same step (this fork)

**Performance Results:**
| Rate (req/s) | Scheduler | Throughput (tokens/s) | Time to first token, p50 (ms) | Inter-token latency, p99 (ms) |
|--------------|-----------|-----------------------|-------------------------------|-------------------------------|
| offline      | Baseline  | 3088                  | –                             | –                             |
| offline      | Mixed     | 3019                  | –                             | –                             |
| 4            | Baseline  | 1790                  | 35.9                          | 42.7                          |
| 4            | Mixed     | 1795                  | 36.2                          | 31.4                          |
| 8            | Baseline  | 2670                  | 41.4                          | 86.2                          |
| 8            | Mixed     | 2684                  | 59.9                          | 69.1                          |

Throughput is unchanged. Inter-token latency p99 drops 20–26% because running sequences no longer stall while a new prompt is prefilled. Time to first token rises at 8 req/s because a step containing prefill runs eager (no CUDA graph) and takes longer than a pure decode step.

## Tests

The scheduler, block manager and sequence logic is tested on CPU, no GPU needed:
```bash
pip install pytest numpy xxhash
pytest
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)