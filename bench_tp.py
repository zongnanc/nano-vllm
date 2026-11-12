import argparse, os, time
from random import randint, seed
from nanovllm import LLM, SamplingParams

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    args = p.parse_args()

    seed(0)
    num_seqs, max_in, max_out = 256, 1024, 1024
    llm = LLM(os.path.expanduser(args.model), enforce_eager=args.eager, max_model_len=4096, tensor_parallel_size=args.tp, max_num_seqs=args.max_num_seqs)
    prompts = [[randint(0, 10000) for _ in range(randint(100, max_in))] for _ in range(num_seqs)]
    sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_out)) for _ in range(num_seqs)]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompts, sps, use_tqdm=False)
    t = time.time() - t
    total = sum(sp.max_tokens for sp in sps)
    print(f"TP={args.tp} Total: {total}tok, Time: {t:.2f}s, Throughput: {total / t:.2f}tok/s")

if __name__ == "__main__":
    main()
