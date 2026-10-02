"""Transformers serving baseline with static batching.

Requests are queued. A worker waits briefly, groups up to --max-batch of them and runs ONE
model.generate() call for the whole group (left-padded). The next batch starts only after
the previous one has fully finished (no continuous batching). Tokens are streamed per request.
"""
import argparse
import asyncio
import json
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--max-batch", type=int, default=16)
ap.add_argument("--max-wait-ms", type=float, default=20.0)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"  # required for batched decoder-only generation
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
try:
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
except Exception:
    # Older transformers versions use torch_dtype instead of dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
model = model.to("cuda").eval()

pending = None  # asyncio.Queue, created inside the event loop


class Req(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 128
    ignore_eos: bool = False


class BatchStreamer:
    """Duck-typed streamer: generate() calls put() once per step with one token id per row."""

    def __init__(self, loop, queues, limits):
        self.loop, self.queues, self.limits = loop, queues, limits
        self.counts = [0] * len(queues)

    def put(self, value):
        if value.dim() > 1:  # the first call carries the prompt ids, skip it
            return
        for i, tid in enumerate(value.tolist()):
            if self.counts[i] < self.limits[i]:
                self.counts[i] += 1
                self.loop.call_soon_threadsafe(self.queues[i].put_nowait, tid)

    def end(self):
        for q in self.queues:
            self.loop.call_soon_threadsafe(q.put_nowait, None)


def run_batch(batch, loop):
    reqs = [b[0] for b in batch]
    queues = [b[1] for b in batch]
    try:
        inputs = tok([r.prompt for r in reqs], return_tensors="pt", padding=True).to("cuda")
        max_new = max(r.max_tokens for r in reqs)
        streamer = BatchStreamer(loop, queues, [r.max_tokens for r in reqs])
        with torch.inference_mode():
            model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                min_new_tokens=max_new if any(r.ignore_eos for r in reqs) else 0,
                pad_token_id=tok.pad_token_id, streamer=streamer,
            )
    except Exception as e:
        print("batch failed:", repr(e), flush=True)
        for q in queues:  # never leave a client hanging
            loop.call_soon_threadsafe(q.put_nowait, None)


async def batch_worker():
    loop = asyncio.get_running_loop()
    while True:
        batch = [await pending.get()]
        deadline = loop.time() + args.max_wait_ms / 1000
        while len(batch) < args.max_batch:
            timeout = deadline - loop.time()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(pending.get(), timeout))
            except asyncio.TimeoutError:
                break
        print(f"running batch of {len(batch)}", flush=True)
        await loop.run_in_executor(None, run_batch, batch, loop)


@asynccontextmanager
async def lifespan(app):
    global pending
    pending = asyncio.Queue()
    task = asyncio.create_task(batch_worker())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/v1/completions")
async def completions(req: Req):
    out_q = asyncio.Queue()
    await pending.put((req, out_q))

    async def gen():
        n = 0
        while True:
            tid = await out_q.get()
            if tid is None:
                break
            n += 1
            yield "data: " + json.dumps({"choices": [{"text": tok.decode([tid])}]}) + "\n\n"
        yield "data: " + json.dumps({"choices": [], "usage": {"completion_tokens": n}}) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
