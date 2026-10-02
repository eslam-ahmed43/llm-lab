"""Closed-loop load generator for any OpenAI-compatible /v1/completions server.

Measures TTFT, per-request latency, aggregate throughput and peak GPU memory
across a concurrency sweep, using identical prompts for every backend.
"""
import argparse
import asyncio
import json
import statistics
import subprocess
import threading
import time

import aiohttp
from transformers import AutoTokenizer


def make_prompt(tok, n_tokens):
    # Build a prompt of roughly n_tokens tokens (re-encoding may shift it by a few tokens)
    base = "The quick brown fox jumps over the lazy dog. "
    ids = tok(base * (n_tokens // 5 + 10), add_special_tokens=False)["input_ids"][:n_tokens]
    return tok.decode(ids)


class GpuSampler:
    """Polls nvidia-smi in a background thread and records peak memory in MiB."""

    def __init__(self, gpu_index=0, interval=0.25):
        self.gpu_index, self.interval = gpu_index, interval
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output([
                    "nvidia-smi", f"--id={self.gpu_index}",
                    "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                ])
                self.peak_mib = max(self.peak_mib, int(out.decode().strip()))
            except Exception:
                pass  # a failed poll must never kill the benchmark
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


async def one_request(session, url, model, prompt, max_tokens):
    payload = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,  # force a fixed output length
    }
    t0 = time.perf_counter()
    ttft, n_out = None, 0
    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if ttft is None and chunk.get("choices") and chunk["choices"][0].get("text"):
                ttft = time.perf_counter() - t0
            if chunk.get("usage"):
                n_out = chunk["usage"]["completion_tokens"]
    return ttft, time.perf_counter() - t0, n_out


async def run_level(url, model, prompt, max_tokens, concurrency, n_requests):
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def guarded():
            async with sem:
                return await one_request(session, url, model, prompt, max_tokens)

        t0 = time.perf_counter()
        results = await asyncio.gather(*[guarded() for _ in range(n_requests)])
        wall = time.perf_counter() - t0
    return results, wall


def summarize(results, wall, concurrency, peak_mib):
    ttfts = sorted(r[0] for r in results if r[0] is not None)
    if not ttfts:
        raise RuntimeError("No successful requests, check the server log")
    lats = sorted(r[1] for r in results)
    total_out = sum(r[2] for r in results)
    pct = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {
        "concurrency": concurrency,
        "n_requests": len(results),
        "ttft_p50_s": statistics.median(ttfts),
        "ttft_p95_s": pct(ttfts, 0.95),
        "latency_p50_s": statistics.median(lats),
        "throughput_tok_s": total_out / wall,
        "peak_gpu_mib": peak_mib,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend-name", required=True)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--requests-per-level", type=int, default=4,
                    help="requests per level = max(this * concurrency, 8)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    prompt = make_prompt(tok, args.prompt_len)
    url = f"{args.base_url}/v1/completions"

    # Warmup so kernels are compiled before measuring
    asyncio.run(run_level(url, args.model, prompt, 16, 2, 4))

    rows = []
    for c in args.concurrency:
        with GpuSampler() as gpu:
            results, wall = asyncio.run(
                run_level(url, args.model, prompt, args.max_tokens, c,
                          max(c * args.requests_per_level, 8)))
        row = summarize(results, wall, c, gpu.peak_mib)
        print(row, flush=True)
        rows.append(row)

    with open(args.out, "w") as f:
        json.dump({"backend": args.backend_name, "model": args.model,
                   "prompt_len": args.prompt_len, "max_tokens": args.max_tokens,
                   "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
