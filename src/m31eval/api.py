import json
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit, urlunsplit


class CapabilityError(RuntimeError):
    pass


class Server:
    def __init__(self, base_url, model=None, api_key_env="OPENAI_API_KEY", timeout=120):
        self.base = base_url.rstrip("/")
        if not self.base.endswith("/v1"):
            self.base += "/v1"
        self.root = self.base[:-3]
        self.model = model
        self.api_key = os.environ.get(api_key_env, "EMPTY")
        self.timeout = timeout
        self.deadline = None

    def request(self, path, payload=None, root=False):
        url = (self.root if root else self.base) + path
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json",
                                      "Authorization": f"Bearer {self.api_key}"})
        try:
            timeout = self.timeout
            if self.deadline is not None:
                timeout = min(timeout, self.deadline - time.monotonic())
                if timeout <= 0:
                    raise TimeoutError("Evaluation time budget exhausted")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as exc:
            detail = exc.read(1500).decode(errors="replace")
            raise CapabilityError(f"HTTP {exc.code} at {path}: {detail}") from exc

    def discover(self):
        models = self.request("/models")["data"]
        if self.model is None:
            if len(models) != 1:
                raise CapabilityError("Specify --model: server exposes multiple models")
            self.model = models[0]["id"]
        if self.model not in [m["id"] for m in models]:
            raise CapabilityError(f"Model not served: {self.model}")
        return models

    def tokenize(self, row):
        payload = {"model": self.model, "add_special_tokens": False}
        if "messages" in row:
            payload.update(messages=row["messages"], add_generation_prompt=True)
            if row.get("tools"):
                payload["tools"] = row["tools"]
            if row.get("chat_template_kwargs"):
                payload["chat_template_kwargs"] = row["chat_template_kwargs"]
        else:
            payload["prompt"] = row["prompt"]
        response = self.request("/tokenize", payload, root=True)
        return response["tokens"]

    def detokenize(self, tokens):
        return self.request("/detokenize", {"model": self.model, "tokens": tokens}, root=True)["prompt"]

    def generate(self, tokens, max_tokens=32, logprobs=None, requested=None, score=False):
        payload = {"model": self.model, "prompt": tokens, "max_tokens": max_tokens,
                   "temperature": 1.0 if score else 0.0, "top_p": 1.0, "top_k": -1,
                   "return_token_ids": True, "return_tokens_as_token_ids": True,
                   "skip_special_tokens": False, "add_special_tokens": False, "seed": 31}
        if logprobs is not None:
            payload["logprobs"] = logprobs
        if requested is not None:
            payload["logprob_token_ids"] = list(requested)
        response = self.request("/completions", payload)
        choice = response["choices"][0]
        ids = choice.get("token_ids")
        lp = choice.get("logprobs") or {}
        if ids is None:
            try:
                ids = [int(t.removeprefix("token_id:")) for t in lp["tokens"]]
            except (KeyError, ValueError):
                raise CapabilityError("Server must return token_ids or token_id:* logprob tokens")
        distributions = []
        for step in lp.get("top_logprobs", []):
            if step is None:
                distributions.append(None)
                continue
            try:
                distributions.append({int(k.removeprefix("token_id:")): v for k, v in step.items()
                                      if k.startswith("token_id:")})
            except (ValueError, AttributeError) as exc:
                raise CapabilityError("Logprobs lack stable token ID keys") from exc
            if not distributions[-1]:
                raise CapabilityError("return_tokens_as_token_ids was ignored")
        return {"token_ids": ids, "text": choice.get("text", ""), "distributions": distributions,
                "finish_reason": choice.get("finish_reason"), "usage": response.get("usage", {})}

    def chat(self, row, max_tokens=32):
        messages = row.get("messages") or [{"role": "user", "content": row["prompt"]}]
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": 0, "top_p": 1, "seed": 31}
        if row.get("tools"):
            payload["tools"] = row["tools"]
        response = self.request("/chat/completions", payload)
        choice = response["choices"][0]
        msg = choice["message"]
        return {"message": {k: msg.get(k) for k in ["content", "reasoning_content", "tool_calls"]},
                "finish_reason": choice.get("finish_reason"), "usage": response.get("usage", {})}

    def responses(self, row, max_tokens=32):
        """Weak parsed-output check for Responses-only deployments; never token KL."""
        items = []
        for message in row.get("messages", [{"role": "user", "content": row.get("prompt", "")} ]):
            if message["role"] == "tool":
                items.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                              "output": message.get("content", "")})
            else:
                content = message.get("content") or ""
                if message.get("reasoning_content"):
                    content = message["reasoning_content"] + "\n" + content
                if content:
                    items.append({"role": message["role"], "content": content})
                for call in message.get("tool_calls", []):
                    items.append({"type": "function_call", "call_id": call["id"], **call["function"]})
        payload = {"model": self.model, "input": items, "max_output_tokens": max_tokens, "temperature": 0}
        if row.get("tools"):
            payload["tools"] = [{"type": "function", **t["function"]} for t in row["tools"]]
        response = self.request("/responses", payload)
        normalized = []
        for item in response.get("output", []):
            normalized.append({k: v for k, v in item.items() if k not in ["id", "call_id", "status"]})
        return {"output": normalized, "status": response.get("status"), "usage": response.get("usage")}
