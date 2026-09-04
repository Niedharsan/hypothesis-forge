from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_PLACEHOLDER_RE = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")


class PromptTemplateError(ValueError):
    """Raised when a prompt template cannot be rendered safely."""


@dataclass(frozen=True)
class PromptTemplate:
    """Lightweight markdown prompt template with placeholder validation."""

    name: str
    text: str

    @property
    def placeholders(self) -> set[str]:
        return set(_PLACEHOLDER_RE.findall(self.text))

    def validate(self, variables: Mapping[str, Any]) -> None:
        missing = sorted(self.placeholders - set(variables.keys()))
        if missing:
            raise PromptTemplateError(
                f"Prompt '{self.name}' missing variables: {', '.join(missing)}"
            )

    def render(self, **variables: Any) -> str:
        self.validate(variables)
        rendered = self.text
        for key, value in variables.items():
            rendered = re.sub(r"{{\s*" + re.escape(key) + r"\s*}}", _stringify(value), rendered)
        return rendered


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_prompt_path(name: str | Path, prompt_root: str | Path = "prompts") -> Path:
    raw = Path(name)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        root = _project_root()
        candidates.append(root / raw)
        candidates.append(root / prompt_root / raw)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    tried = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Prompt template not found: {name}. Tried:\n{tried}")


def load_prompt_template(name: str, prompt_root: str | Path = "prompts") -> PromptTemplate:
    """Load a markdown prompt template by relative path."""
    path = _resolve_prompt_path(name, prompt_root=prompt_root)
    return PromptTemplate(name=str(name), text=path.read_text(encoding="utf-8"))


def render_prompt(name: str, prompt_root: str | Path = "prompts", **variables: Any) -> str:
    return load_prompt_template(name, prompt_root=prompt_root).render(**variables)


def load_prompt(path: str | Path, **variables: Any) -> str:
    """Compatibility helper: load a prompt file and optionally render variables.

    Existing code can call ``load_prompt('v31/foo.md', x='...')`` while newer
    code can use ``render_prompt``. Missing variables are validated when any
    placeholders exist.
    """
    template = load_prompt_template(str(path))
    if variables:
        return template.render(**variables)
    if template.placeholders:
        return template.text
    return template.text


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)
