"""Thin client for a local Ollama server (chat/instruct models).

Uses /api/generate with the model's chat template applied (raw=False), so
models are queried the way a practitioner would query them. Returns the full
response metadata (not just text) so truncation (`done_reason == "length"`)
can be detected and reported rather than silently scored as wrong.
"""
import json
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"


def generate(model, prompt, temperature=0.0, num_predict=300, raw=False,
             timeout=600, seed=0, num_ctx=8192, think=False, retries=3):
    """Returns a dict with keys: response, done_reason, eval_count,
    prompt_eval_count.

    think=False disables the hidden reasoning mode of hybrid-thinking models
    (e.g. qwen3); without it, such models spend the token budget inside a
    <think> block and never emit an answer. Ignored by models that don't
    support thinking.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "raw": raw,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "seed": seed,
            "num_ctx": num_ctx,
        },
    }
    if think is not None:
        payload["think"] = think
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                OLLAMA_URL,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                d = json.loads(resp.read())
            return {
                "response": d.get("response", ""),
                "done_reason": d.get("done_reason"),
                "eval_count": d.get("eval_count"),
                "prompt_eval_count": d.get("prompt_eval_count"),
            }
        except urllib.error.HTTPError as e:
            # Some models reject the `think` field; retry without it.
            if e.code == 400 and "think" in payload:
                payload.pop("think")
                continue
            last_err = e
        except Exception as e:
            last_err = e
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Ollama request failed after {retries} attempts: {last_err}")


def chat(model, messages, temperature=0.0, num_predict=300, timeout=600, seed=0,
         num_ctx=8192, think=False, retries=3):
    """Like generate(), but sends a multi-turn message list to /api/chat."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict,
                    "seed": seed, "num_ctx": num_ctx},
    }
    if think is not None:
        payload["think"] = think
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                OLLAMA_URL.replace("/api/generate", "/api/chat"),
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                d = json.loads(resp.read())
            return {
                "response": d.get("message", {}).get("content", ""),
                "done_reason": d.get("done_reason"),
                "eval_count": d.get("eval_count"),
                "prompt_eval_count": d.get("prompt_eval_count"),
            }
        except urllib.error.HTTPError as e:
            if e.code == 400 and "think" in payload:
                payload.pop("think")
                continue
            last_err = e
        except Exception as e:
            last_err = e
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Ollama request failed after {retries} attempts: {last_err}")


def is_server_up():
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False
