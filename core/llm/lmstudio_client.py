from __future__ import annotations
import requests, os

class LMStudioClient:
    def __init__(self, base_url: str = None, model: str = None, temperature: float = 0.2, max_tokens: int = 1200):
        self.base_url = base_url or os.environ.get("CATCHUP_LLM_BASE_URL", "http://localhost:1234/v1")
        self.model = model or os.environ.get("CATCHUP_LLM_MODEL", "llama-3.1-8b-instruct")
        self.temperature = float(os.environ.get("CATCHUP_LLM_TEMPERATURE", temperature))
        self.max_tokens = int(os.environ.get("CATCHUP_LLM_MAX_TOKENS", max_tokens))

    def chat(self, messages: list[dict]) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
