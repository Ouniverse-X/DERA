"""OpenAI-compatible DeepSeek client with JSON extraction and call logging."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

def extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        value = None
        for i, character in enumerate(text):
            if character != "{": continue
            try: candidate, _ = decoder.raw_decode(text[i:])
            except json.JSONDecodeError: continue
            if isinstance(candidate, dict): value = candidate; break
    if not isinstance(value, dict):
        raise ValueError("LLM response does not contain a JSON object")
    return value


class DeepSeekClient:
    def __init__(self, model: str | None = None, base_url: str | None = None, log_dir: Path | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("install openai>=2.44.0") from exc
        key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key: raise RuntimeError("DEEPSEEK_API_KEY or OPENAI_API_KEY is required")
        kwargs: dict[str, Any] = {"api_key": key}
        url = base_url or os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.deepseek.com"
        if url: kwargs["base_url"] = url
        self.client = OpenAI(**kwargs)
        self.model = model or os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4-flash"
        self.log_dir = log_dir
        if log_dir: log_dir.mkdir(parents=True, exist_ok=True)

    def json(self, system: str, user: str, call_name: str, retries: int = 6) -> dict[str, Any]:
        last_error: Exception | None = None
        started = time.monotonic()
        accumulated_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(retries + 1):
            text = ""
            response_metadata: dict[str, Any] = {}
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                response_usage = getattr(response, "usage", None)
                for key in accumulated_usage:
                    accumulated_usage[key] += int(getattr(response_usage, key, 0) or 0)
                choice = response.choices[0]
                message = choice.message
                content = message.content or ""
                reasoning = getattr(message, "reasoning_content", None) or ""
                text = content or reasoning
                response_metadata = {"finish_reason": getattr(choice, "finish_reason", None), "content_length": len(content), "reasoning_length": len(reasoning)}
                value = extract_json(text)
                break
            except Exception as exc:
                last_error = exc
                if self.log_dir:
                    failure = self.log_dir / f"failed-{call_name}-attempt-{attempt + 1}.json"
                    failure.write_text(json.dumps({"model": self.model, "call_name": call_name, "attempt": attempt + 1, "error": repr(exc), "raw_response": text, **response_metadata}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                if attempt >= retries: raise
                if response_metadata:
                    if text:
                        messages = messages + [{"role": "assistant", "content": text}]
                    messages = messages + [{"role": "user", "content": "Your previous response was empty or was not a valid JSON object. Return only one complete JSON object matching the requested schema; do not use Markdown or explanatory text."}]
                time.sleep(2 ** attempt)
        else:
            raise last_error or RuntimeError("LLM call failed")
        if self.log_dir:
            index = len(list(self.log_dir.glob("*.json")))
            payload = {
                "model": self.model,
                "call_name": call_name,
                "elapsed_seconds": time.monotonic() - started,
                "attempts": attempt + 1,
                "usage": accumulated_usage,
                "system": system,
                "user": user,
                "response": value,
            }
            (self.log_dir / f"{index:04d}-{call_name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            logs = []
            for path in self.log_dir.glob("*.json"):
                try:
                    item = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(item.get("usage"), dict):
                    logs.append(item)
            totals = {
                key: sum(int(item["usage"].get(key) or 0) for item in logs)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            totals.update({
                "calls": len(logs),
                "elapsed_seconds": sum(float(item.get("elapsed_seconds", 0.0)) for item in logs),
            })
            (self.log_dir / "usage_summary.json").write_text(json.dumps(totals, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return value
