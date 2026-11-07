import argparse
import os
import time
from collections import deque
from random import expovariate, randint, seed

import numpy as np

from nanovllm import LLM, SamplingParams


def summarize(name: str, seconds: np.ndarray):
    ms = seconds * 1000
    p50, p99 = np.percentile(ms, [50, 99])
    print(f"{name} (ms): mean {ms.mean():.1f}, p50 {p50:.1f}, p99 {p99:.1f}, max {ms.max():.1f}")


def main():
    parser = argparse.ArgumentParser(description="Online benchmark: requests arrive over time instead of all at once.")
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--request-rate", type=float, default=4.0, help="requests/s with Poisson arrivals; inf sends all at t=0")
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed(args.seed)
    max_input_len = 1024
    max_output_len = 1024

    path = os.path.expanduser(args.model)
    llm = LLM(path, enforce_eager=False, max_model_len=4096, max_num_batched_tokens=args.max_num_batched_tokens)

    # same workload as bench.py, plus arrival times
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(args.num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len)) for _ in range(args.num_seqs)]
    arrivals = [0.0]
    for _ in range(args.num_seqs - 1):
        arrivals.append(arrivals[-1] + (expovariate(args.request_rate) if args.request_rate != float("inf") else 0.0))

    llm.generate(["Benchmark: "], SamplingParams())
    requests = deque(zip(arrivals, prompt_token_ids, sampling_params))
    active = []    # (seq, arrival time, time each output token was produced)
    finished = []
    start = time.perf_counter()
    while requests or active:
        now = time.perf_counter() - start
        while requests and requests[0][0] <= now:
            arrival, prompt, sp = requests.popleft()
            llm.add_request(prompt, sp)
            active.append((llm.scheduler.waiting[-1], arrival, []))
        if llm.is_finished():    # idle until the next request arrives
            time.sleep(max(requests[0][0] - now, 0))
            continue
        llm.step()
        now = time.perf_counter() - start
        still_active = []
        for seq, arrival, token_times in active:
            token_times.extend([now] * (seq.num_completion_tokens - len(token_times)))
            (finished if seq.is_finished else still_active).append((seq, arrival, token_times))
        active = still_active

    duration = max(token_times[-1] for _, _, token_times in finished)
    num_tokens = sum(len(token_times) for _, _, token_times in finished)
    ttft = np.array([token_times[0] - arrival for _, arrival, token_times in finished])
    itl = np.concatenate([np.diff(token_times) for _, _, token_times in finished])
    print(f"Requests: {args.num_seqs}, Rate: {args.request_rate}req/s, max_num_batched_tokens: {args.max_num_batched_tokens}")
    print(f"Total: {num_tokens}tok, Time: {duration:.2f}s, Throughput: {num_tokens / duration:.2f}tok/s")
    summarize("TTFT", ttft)
    summarize("ITL", itl)


if __name__ == "__main__":
    main()
