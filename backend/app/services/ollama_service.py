"""All HTTP communication with the local Ollama server lives here."""

import re
from typing import Any

import requests

from app.config import Settings, get_settings


class OllamaError(Exception):
    """Base error for Ollama failures. `status_code` is the HTTP code the API should return."""

    status_code: int = 502

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class OllamaUnavailableError(OllamaError):
    status_code = 503


class OllamaModelNotFoundError(OllamaError):
    status_code = 503


class OllamaTimeoutError(OllamaError):
    status_code = 504


class OllamaInvalidResponseError(OllamaError):
    status_code = 502


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _normalize_model_name(name: str) -> str:
    """Ollama treats `model` and `model:latest` as the same model."""
    return name if ":" in name else f"{name}:latest"


class OllamaService:
    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model
        self._connect_timeout = settings.ollama_connect_timeout
        self._status_timeout = settings.ollama_status_timeout
        self._generation_timeout = settings.ollama_generation_timeout

    def _request(self, method: str, path: str, read_timeout: float, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = requests.request(
                method, url, timeout=(self._connect_timeout, read_timeout), **kwargs
            )
        except requests.exceptions.ConnectTimeout as exc:
            raise OllamaUnavailableError(f"Timed out connecting to Ollama at {self.base_url}") from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaTimeoutError(
                f"Ollama did not respond within {read_timeout:.0f} seconds"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise OllamaUnavailableError(f"Could not connect to Ollama at {self.base_url}") from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaUnavailableError(f"Request to Ollama failed: {exc}") from exc

        if response.status_code == 404:
            detail = self._error_detail(response)
            raise OllamaModelNotFoundError(detail or f"Model '{self.model}' was not found in Ollama")
        if not response.ok:
            detail = self._error_detail(response)
            raise OllamaInvalidResponseError(
                f"Ollama returned HTTP {response.status_code}" + (f": {detail}" if detail else "")
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaInvalidResponseError("Ollama returned a non-JSON response") from exc
        if not isinstance(data, dict):
            raise OllamaInvalidResponseError("Ollama returned an unexpected JSON payload")
        return data

    @staticmethod
    def _error_detail(response: requests.Response) -> str:
        try:
            body = response.json()
            if isinstance(body, dict) and isinstance(body.get("error"), str):
                return body["error"]
        except ValueError:
            pass
        return response.text.strip()[:500]

    def list_models(self) -> list[str]:
        data = self._request("GET", "/api/tags", self._status_timeout)
        models = data.get("models")
        if not isinstance(models, list):
            raise OllamaInvalidResponseError("Ollama /api/tags response is missing 'models'")
        names: list[str] = []
        for item in models:
            if isinstance(item, dict):
                name = item.get("name") or item.get("model")
                if isinstance(name, str):
                    names.append(name)
        return names

    def is_model_available(self, installed_models: list[str]) -> bool:
        target = _normalize_model_name(self.model)
        return any(_normalize_model_name(name) == target for name in installed_models)

    def chat(
        self,
        message: str,
        *,
        system: str | None = None,
        response_format: dict[str, Any] | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a single-turn chat. `response_format` maps to Ollama's `format`
        (either "json" or a JSON schema) for structured output."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # Disable qwen3 "thinking" output to keep CPU inference fast and responses clean.
            "think": False,
        }
        if response_format is not None:
            payload["format"] = response_format
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if options:
            payload["options"] = options
        data = self._request("POST", "/api/chat", self._generation_timeout, json=payload)

        msg = data.get("message")
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, str):
            raise OllamaInvalidResponseError("Ollama chat response is missing 'message.content'")

        # Fallback in case an Ollama version ignores `think` and inlines reasoning.
        return _THINK_BLOCK.sub("", content).strip()


def get_ollama_service() -> OllamaService:
    return OllamaService(get_settings())
