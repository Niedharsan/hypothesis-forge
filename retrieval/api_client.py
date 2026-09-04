from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import requests

from utils.run_logger import log_event
from utils.security import redact_sensitive_text


class CachedAPIClient:
    def __init__(self, cache_dir: str | Path = "data/cache/api", min_interval_seconds: float = 0.2):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval_seconds = min_interval_seconds
        self._last_call_at = 0.0

    def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        cache_namespace: str = "api",
        max_retries: int = 3,
    ) -> dict:
        cache_path = self._cache_path(url, params, cache_namespace)
        cache_meta = {"url": url, "cache_namespace": cache_namespace, "cache_key": cache_path.stem}
        if cache_path.exists():
            log_event("api", "cache_hit", cache_meta)
            return json.loads(cache_path.read_text(encoding="utf-8"))

        log_event("api", "cache_miss", cache_meta)
        data = self._request_json_with_retries(
            method="GET",
            url=url,
            params=params,
            headers=headers,
            json_payload=None,
            cache_namespace=cache_namespace,
            timeout=15,
            max_retries=max_retries,
        )
        cache_path.write_text(json.dumps(data), encoding="utf-8")
        return data

    def get_text(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        cache_namespace: str = "api_text",
        max_retries: int = 3,
    ) -> str:
        cache_path = self._cache_path(url, params, cache_namespace).with_suffix(".txt")
        cache_meta = {"url": url, "cache_namespace": cache_namespace, "cache_key": cache_path.stem}
        if cache_path.exists():
            log_event("api", "cache_hit", cache_meta)
            return cache_path.read_text(encoding="utf-8")
        log_event("api", "cache_miss", cache_meta)
        text = self._request_text_with_retries(
            method="GET",
            url=url,
            params=params,
            headers=headers,
            cache_namespace=cache_namespace,
            timeout=20,
            max_retries=max_retries,
        )
        cache_path.write_text(text, encoding="utf-8")
        return text

    def post_json(
        self,
        url: str,
        payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        cache_namespace: str = "api",
        max_retries: int = 3,
    ) -> dict:
        cache_path = self._cache_path(url, {"params": params or {}, "payload": payload or {}}, cache_namespace)
        cache_meta = {"url": url, "cache_namespace": cache_namespace, "cache_key": cache_path.stem}
        if cache_path.exists():
            log_event("api", "cache_hit", cache_meta)
            return json.loads(cache_path.read_text(encoding="utf-8"))

        log_event("api", "cache_miss", cache_meta)
        request_headers = dict(headers or {})
        request_headers.setdefault("Content-Type", "application/json")
        data = self._request_json_with_retries(
            method="POST",
            url=url,
            params=params,
            headers=request_headers,
            json_payload=payload or {},
            cache_namespace=cache_namespace,
            timeout=20,
            max_retries=max_retries,
        )
        cache_path.write_text(json.dumps(data), encoding="utf-8")
        return data

    def _request_json_with_retries(
        self,
        *,
        method: str,
        url: str,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
        json_payload: dict[str, Any] | None,
        cache_namespace: str,
        timeout: int,
        max_retries: int,
    ) -> dict:
        delay = 1.0
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            self._rate_limit()
            try:
                log_event(
                    "api",
                    "request_start",
                    {"method": method, "url": url, "cache_namespace": cache_namespace, "attempt": attempt + 1},
                )
                if method == "GET":
                    response = requests.get(url, params=params, headers=headers, timeout=timeout)
                else:
                    response = requests.post(url, params=params, headers=headers, json=json_payload or {}, timeout=timeout)

                if _should_retry_status(response.status_code) and attempt < max_retries:
                    retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
                    sleep_s = retry_after if retry_after is not None else _jitter(delay)
                    log_event(
                        "api",
                        "request_retry",
                        {
                            "method": method,
                            "url": url,
                            "cache_namespace": cache_namespace,
                            "attempt": attempt + 1,
                            "status_code": response.status_code,
                            "sleep_s": round(sleep_s, 3),
                        },
                        status="error",
                    )
                    time.sleep(sleep_s)
                    delay *= 2
                    continue
                response.raise_for_status()
                data = response.json()
                log_event(
                    "api",
                    "request_end",
                    {
                        "method": method,
                        "url": url,
                        "cache_namespace": cache_namespace,
                        "attempt": attempt + 1,
                        "status_code": response.status_code,
                    },
                )
                return data
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt < max_retries:
                    sleep_s = _jitter(delay)
                    log_event(
                        "api",
                        "request_retry",
                        {
                            "method": method,
                            "url": url,
                            "cache_namespace": cache_namespace,
                            "attempt": attempt + 1,
                            "error": redact_sensitive_text(exc),
                            "sleep_s": round(sleep_s, 3),
                        },
                        status="error",
                    )
                    time.sleep(sleep_s)
                    delay *= 2
                    continue
                break
            except requests.HTTPError as exc:
                last_error = exc
                status_code = getattr(exc.response, "status_code", None)
                log_event(
                    "api",
                    "request_error",
                    {
                        "method": method,
                        "url": url,
                        "cache_namespace": cache_namespace,
                        "attempt": attempt + 1,
                        "status_code": status_code,
                        "error": redact_sensitive_text(exc),
                    },
                    status="error",
                )
                raise RuntimeError(f"API request failed: {url}: HTTP {status_code or 'error'}") from exc
            except Exception as exc:
                last_error = exc
                log_event(
                    "api",
                    "request_error",
                    {
                        "method": method,
                        "url": url,
                        "cache_namespace": cache_namespace,
                        "attempt": attempt + 1,
                        "error": redact_sensitive_text(exc),
                    },
                    status="error",
                )
                raise
        error_message = redact_sensitive_text(last_error) if last_error else "unknown error"
        log_event(
            "api",
            "request_failed_after_retries",
            {"method": method, "url": url, "cache_namespace": cache_namespace, "error": error_message},
            status="error",
        )
        raise RuntimeError(f"API request failed after retries: {url}: {error_message}")

    def _request_text_with_retries(
        self,
        *,
        method: str,
        url: str,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
        cache_namespace: str,
        timeout: int,
        max_retries: int,
    ) -> str:
        delay = 1.0
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            self._rate_limit()
            try:
                log_event(
                    "api",
                    "request_start",
                    {"method": method, "url": url, "cache_namespace": cache_namespace, "attempt": attempt + 1},
                )
                response = requests.get(url, params=params, headers=headers, timeout=timeout)
                if _should_retry_status(response.status_code) and attempt < max_retries:
                    retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
                    sleep_s = retry_after if retry_after is not None else _jitter(delay)
                    log_event(
                        "api",
                        "request_retry",
                        {
                            "method": method,
                            "url": url,
                            "cache_namespace": cache_namespace,
                            "attempt": attempt + 1,
                            "status_code": response.status_code,
                            "sleep_s": round(sleep_s, 3),
                        },
                        status="error",
                    )
                    time.sleep(sleep_s)
                    delay *= 2
                    continue
                response.raise_for_status()
                log_event(
                    "api",
                    "request_end",
                    {"method": method, "url": url, "cache_namespace": cache_namespace, "attempt": attempt + 1, "status_code": response.status_code},
                )
                return response.text
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt < max_retries:
                    sleep_s = _jitter(delay)
                    log_event(
                        "api",
                        "request_retry",
                        {
                            "method": method,
                            "url": url,
                            "cache_namespace": cache_namespace,
                            "attempt": attempt + 1,
                            "error": redact_sensitive_text(exc),
                            "sleep_s": round(sleep_s, 3),
                        },
                        status="error",
                    )
                    time.sleep(sleep_s)
                    delay *= 2
                    continue
                break
            except requests.HTTPError as exc:
                status_code = getattr(exc.response, "status_code", None)
                log_event(
                    "api",
                    "request_error",
                    {
                        "method": method,
                        "url": url,
                        "cache_namespace": cache_namespace,
                        "attempt": attempt + 1,
                        "status_code": status_code,
                        "error": redact_sensitive_text(exc),
                    },
                    status="error",
                )
                raise RuntimeError(f"API request failed: {url}: HTTP {status_code or 'error'}") from exc
            except Exception as exc:
                last_error = exc
                log_event(
                    "api",
                    "request_error",
                    {
                        "method": method,
                        "url": url,
                        "cache_namespace": cache_namespace,
                        "attempt": attempt + 1,
                        "error": redact_sensitive_text(exc),
                    },
                    status="error",
                )
                raise
        error_message = redact_sensitive_text(last_error) if last_error else "unknown error"
        log_event(
            "api",
            "request_failed_after_retries",
            {"method": method, "url": url, "cache_namespace": cache_namespace, "error": error_message},
            status="error",
        )
        raise RuntimeError(f"API request failed after retries: {url}: {error_message}")

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_call_at
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self._last_call_at = time.monotonic()

    def _cache_path(self, url: str, params: dict[str, Any] | None, namespace: str) -> Path:
        key = json.dumps({"url": url, "params": params or {}}, sort_keys=True)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        path = self.cache_dir / namespace
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{digest}.json"


def _should_retry_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= int(status_code) <= 599


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _jitter(delay: float) -> float:
    return max(0.1, delay + random.uniform(0, delay * 0.25))


def clean_text(value: str | None) -> str | None:
    if not value:
        return None
    return " ".join(str(value).replace("<jats:p>", " ").replace("</jats:p>", " ").split())


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def year_from_text(value: str | None) -> int | None:
    if not value:
        return None
    for token in str(value).replace("-", " ").split():
        if token.isdigit() and len(token) == 4:
            return int(token)
    return None
