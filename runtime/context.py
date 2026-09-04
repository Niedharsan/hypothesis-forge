from __future__ import annotations

import contextvars
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RunLimits:
    max_llm_calls_per_run: int = 8

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "RunLimits":
        cfg = ((config or {}).get("runtime") or {}).get("limits") or {}
        return cls(max_llm_calls_per_run=int(cfg.get("max_llm_calls_per_run", 8)))


@dataclass
class DryRunSettings:
    mock_llm_outputs: bool = True
    mock_response_tokens: int = 600

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "DryRunSettings":
        cfg = ((config or {}).get("runtime") or {}).get("dry_run") or {}
        return cls(
            mock_llm_outputs=bool(cfg.get("mock_llm_outputs", True)),
            mock_response_tokens=int(cfg.get("mock_response_tokens", 600)),
        )


@dataclass
class RuntimeContext:
    mode: str = "normal"
    limits: RunLimits = field(default_factory=RunLimits)
    dry_run: DryRunSettings = field(default_factory=DryRunSettings)
    pricing: dict[str, Any] = field(default_factory=dict)

    @property
    def is_dry_run(self) -> bool:
        return self.mode == "dry_run"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "limits": asdict(self.limits),
            "dry_run": asdict(self.dry_run),
            "pricing": self.pricing,
        }


_current_runtime = contextvars.ContextVar("runtime", default=RuntimeContext())
_llm_call_count = contextvars.ContextVar("llm_call_count", default=0)
_llm_call_count_seed = contextvars.ContextVar("llm_call_count_seed", default=0)


def seed_llm_call_count(value: int) -> None:
    """Seed the next runtime configuration with calls already used by this run.

    The seed is one-shot and context-local so concurrent FastAPI requests do not
    share budget state. The checkpoint API supplies the persisted cumulative
    call count before entering the next stage.
    """
    _llm_call_count_seed.set(max(0, int(value)))


def configure_runtime(config: dict[str, Any] | None = None, *, mode_override: str | None = None) -> RuntimeContext:
    config = config or {}
    mode = str(mode_override or ((config.get("runtime") or {}).get("mode")) or "normal").lower()
    if mode not in {"normal", "dry_run"}:
        mode = "normal"
    rt = RuntimeContext(
        mode=mode,
        limits=RunLimits.from_config(config),
        dry_run=DryRunSettings.from_config(config),
        pricing=((config.get("runtime") or {}).get("pricing") or {}),
    )
    seed = max(0, int(_llm_call_count_seed.get()))
    _current_runtime.set(rt)
    _llm_call_count.set(seed)
    _llm_call_count_seed.set(0)
    return rt


def current_runtime() -> RuntimeContext:
    return _current_runtime.get()


def increment_llm_call_count() -> int:
    value = _llm_call_count.get() + 1
    _llm_call_count.set(value)
    return value


def current_llm_call_count() -> int:
    return _llm_call_count.get()
