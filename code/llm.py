"""minimal OpenAI-compatible client with an on-disk cache and a per-call cost log.

Keys are read from the JSON file named by $LLM_KEYS at import time and never written anywhere.
The cache and the usage log live in ./cache, so the daily spend of this
project can be audited on its own.
"""
import hashlib, json, os, time, threading
import requests

SEMVIEW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = json.load(open(os.environ.get("LLM_KEYS", "llm_keys.json")))
CACHE_DIR = os.path.join(SEMVIEW, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
# Endpoints as of 2026-11-03: DeepSeek serves deepseek-flash and deepseek-v4-pro (thinking can be disabled);
# GLM key "glm" only reaches the free, rate-limited glm-4.5-flash and key "glm2" only glm-4.5-air;
# Kimi k2.6/k3 need temperature 0.6 with thinking disabled and are rate-limited.
MODELS = {"glm-flash": ("glm", "glm-4.5-flash"), "glm-air": ("glm2", "glm-4.5-air"),
          "ds-flash": ("deepseek", "deepseek-flash"), "ds-pro": ("deepseek", "deepseek-v4-pro"),
          "ds-flash-think": ("deepseek", "deepseek-flash"), "ds-pro-think": ("deepseek", "deepseek-v4-pro"),
          "kimi-k2.6": ("kimi", "kimi-k2.6"), "kimi-k3": ("kimi", "kimi-k3")}
OFF = {"thinking": {"type": "disabled"}}
FIXED = {"glm-flash": OFF, "glm-air": OFF, "ds-flash": OFF, "ds-pro": OFF,
         "ds-flash-think": {"reasoning_effort": "low"}, "ds-pro-think": {"reasoning_effort": "high"},
         "kimi-k2.6": {**OFF, "temperature": 0.6}, "kimi-k3": {**OFF, "temperature": 0.6}}
# USD per 1M tokens (input, output), same sources as the lead's label.py (DeepSeek peak rate, 2026-11-03)
PRICES = {"glm-flash": (0.0, 0.0), "glm-air": (0.20, 1.10), "ds-flash": (0.30, 1.20), "ds-pro": (1.32, 3.96),
          "ds-flash-think": (0.30, 1.20), "ds-pro-think": (1.32, 3.96), "kimi-k2.6": (0.95, 4.00), "kimi-k3": (3.00, 15.00)}
_lock = threading.Lock()


def cost(model, usage):
    pi, po = PRICES[model]
    return (usage.get("prompt_tokens", 0) * pi + usage.get("completion_tokens", 0) * po) / 1e6


def chat(model, messages, temperature=0.0, max_tokens=512, retries=8, **kw):
    prov, name = MODELS[model]
    for k, v in FIXED.get(model, {}).items():
        if k == "temperature": temperature = v
        else: kw.setdefault(k, v)
    body = {"model": name, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, **kw}
    key = hashlib.sha256(json.dumps([prov, body], sort_keys=True).encode()).hexdigest()
    path = os.path.join(CACHE_DIR, key[:2], key + ".json")
    if os.path.exists(path):
        return {**json.load(open(path)), "cached": True}
    cfg = KEYS[prov]
    err = None
    for i in range(retries):
        try:
            r = requests.post(cfg["base_url"].rstrip("/") + "/chat/completions",
                              headers={"Authorization": "Bearer " + cfg["api_key"]}, json=body, timeout=300)
            if r.status_code == 200:
                d = r.json()
                out = {"text": d["choices"][0]["message"]["content"] or "", "usage": d.get("usage", {}), "model": name}
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with _lock:
                    json.dump(out, open(path, "w"))
                    with open(os.path.join(CACHE_DIR, "usage.log"), "a") as f:
                        f.write(json.dumps({"t": time.time(), "labeler_model": model, "model": name,
                                            "usd": cost(model, out["usage"]), **out["usage"]}) + "\n")
                return out
            err = f"{r.status_code} {r.text[:200]}"
        except Exception as e:
            err = repr(e)
        time.sleep(min(60, 2 ** i) + (5 if "429" in str(err) else 0))
    raise RuntimeError(err)


def spent_today(model=None):
    """USD spent by this project today (local time), optionally for one catalog model."""
    p = os.path.join(CACHE_DIR, "usage.log")
    if not os.path.exists(p): return 0.0
    t0 = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
    s = 0.0
    for l in open(p):
        r = json.loads(l)
        if r["t"] >= t0 and (model is None or MODELS.get(r["labeler_model"], (0, 0))[1] == MODELS[model][1]):
            s += r["usd"]
    return s
