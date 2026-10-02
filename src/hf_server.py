"""Minimal Transformers serving baseline: one generation at a time, no batching."""
import argparse
import asyncio
import json
import threading
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--port", type=int, default=8000)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
try:
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
except Exception:
    # Older transformers versions use torch_dtype instead of dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
model = model.to("cuda").eval()

gen_lock = asyncio.Lock()  # serialize generations, mimics a naive server
app = FastAPI()


class Req(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 128
    ignore_eos: bool = False


@app.get("/health")
def health():
    return {"ok": True}


async def stream_response(req: Req):
    loop = asyncio.get_running_loop()
    async with gen_lock:
        inputs = tok(req.prompt, return_tensors="pt").to("cuda")
        n_prompt = inputs["input_ids"].shape[1]
        streamer = TextIteratorStreamer(tok, skip_prompt=True)
        kwargs = dict(
            **inputs, max_new_tokens=req.max_tokens, do_sample=False,
            min_new_tokens=req.max_tokens if req.ignore_eos else 0,
            pad_token_id=tok.eos_token_id, streamer=streamer,
        )
        holder = {}

        def work():
            try:
                with torch.inference_mode():
                    holder["out"] = model.generate(**kwargs)
            except Exception as e:
                holder["err"] = e
                streamer.end()  # unblock the consumer loop

        t = threading.Thread(target=work)
        t.start()
        it = iter(streamer)
        while True:
            piece = await loop.run_in_executor(None, next, it, None)
            if piece is None:
                break
            if piece:
                yield "data: " + json.dumps({"choices": [{"text": piece}]}) + "\n\n"
        await loop.run_in_executor(None, t.join)
        n_out = holder["out"].shape[1] - n_prompt if "out" in holder else 0
    yield "data: " + json.dumps({"choices": [], "usage": {"completion_tokens": int(n_out)}}) + "\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(req: Req):
    return StreamingResponse(stream_response(req), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
