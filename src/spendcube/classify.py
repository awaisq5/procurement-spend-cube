"""Spend classification in three layers, cheapest and most explainable first.

1. Rules: keywords in the line description.
2. Supplier profile: a supplier whose classified spend is clearly one category
   lends that category to its vague lines ("Monthly fee", "Services as per PO").
3. Language model (optional): only for what is left, sending descriptions
   only. Supplier names and amounts never leave the machine.

Every line records which layer classified it and with what confidence, so a
buyer can see why a line landed where it did.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd

from .taxonomy import BY_ID, CATEGORIES, PATTERNS, UNCLASSIFIED
from .text import normalise_description


def classify_description(description: str) -> tuple[str, float]:
    """Category by keyword rules. Ties between categories stay unclassified."""
    text = normalise_description(description)
    scores: dict[str, int] = {}
    for category_id, patterns in PATTERNS:
        hits = sum(1 for p in patterns if p.search(text))
        if hits:
            scores[category_id] = hits
    if not scores:
        return UNCLASSIFIED, 0.0
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return UNCLASSIFIED, 0.0
    return ranked[0][0], 0.95 if len(ranked) == 1 else 0.8


def apply_rules(lines: pd.DataFrame) -> pd.DataFrame:
    results = lines.description.map(classify_description)
    out = lines.copy()
    out["category_id"] = [r[0] for r in results]
    out["confidence"] = [r[1] for r in results]
    out["source"] = ["rule" if c != UNCLASSIFIED else "none" for c in out.category_id]
    return out


def apply_supplier_profile(lines: pd.DataFrame, min_share: float = 0.8, min_lines: int = 3) -> pd.DataFrame:
    """Give a supplier's dominant category to its unclassified lines."""
    classified = lines[lines.category_id != UNCLASSIFIED]
    by_supplier = classified.groupby(["supplier_id", "category_id"]).amount_eur.agg(["sum", "size"]).reset_index()
    totals = by_supplier.groupby("supplier_id")[["sum", "size"]].sum()
    top = by_supplier.sort_values("sum", ascending=False).drop_duplicates("supplier_id").set_index("supplier_id")
    share = top["sum"] / totals.loc[top.index, "sum"]
    eligible = top[(share >= min_share) & (totals.loc[top.index, "size"] >= min_lines)]

    out = lines.copy()
    mask = (out.category_id == UNCLASSIFIED) & out.supplier_id.isin(eligible.index)
    out.loc[mask, "category_id"] = out.loc[mask, "supplier_id"].map(eligible.category_id)
    out.loc[mask, "confidence"] = out.loc[mask, "supplier_id"].map(share.round(2) * 0.85)
    out.loc[mask, "source"] = "supplier"
    return out


# Language model layer ------------------------------------------------------------------------


@dataclass(frozen=True)
class LlmConfig:
    """Any OpenAI-compatible chat endpoint: OpenRouter, Ollama, or a company gateway."""

    base_url: str
    model: str
    api_key: str | None = None
    batch_size: int = 40
    timeout: float = 60.0


PRESETS = {
    "openrouter": ("https://openrouter.ai/api/v1", "deepseek/deepseek-chat"),
    "ollama": ("http://localhost:11434/v1", "llama3.1"),
}


def _prompt(items: list[tuple[str, str]]) -> list[dict[str, str]]:
    taxonomy = "\n".join(f"{c.id}: {c.group} / {c.name}" for c in CATEGORIES)
    lines = "\n".join(f"{key}: {desc}" for key, desc in items)
    return [
        {
            "role": "system",
            "content": (
                "You classify indirect procurement spend. Descriptions may be English or German. "
                "Use only these category ids:\n"
                f"{taxonomy}\n"
                f"If a description gives no real clue, answer {UNCLASSIFIED}. "
                'Reply with JSON only: {"results": [{"id": "...", "category": "...", "confidence": 0.0}]}. '
                "Confidence is your probability that the category is right."
            ),
        },
        {"role": "user", "content": lines},
    ]


class LlmClassifier:
    """Batches descriptions, validates answers against the taxonomy, caches on disk."""

    def __init__(self, config: LlmConfig, cache_path: Path | None = None, client: httpx.Client | None = None):
        self.config = config
        self.cache_path = cache_path
        self.cache: dict[str, tuple[str, float]] = {}
        if cache_path and cache_path.exists():
            self.cache = {k: tuple(v) for k, v in json.loads(cache_path.read_text()).items()}
        headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
        self.client = client or httpx.Client(base_url=config.base_url, headers=headers, timeout=config.timeout)
        self.calls = 0

    @staticmethod
    def _key(description: str) -> str:
        return hashlib.sha1(normalise_description(description).encode()).hexdigest()[:16]

    def classify(self, descriptions: Iterable[str]) -> dict[str, tuple[str, float]]:
        """Map each description to (category id, confidence)."""
        unique = {self._key(d): d for d in descriptions}
        todo = [(k, d) for k, d in unique.items() if k not in self.cache]
        for start in range(0, len(todo), self.config.batch_size):
            self._ask(todo[start : start + self.config.batch_size])
        self._save()
        return {d: self.cache.get(k, (UNCLASSIFIED, 0.0)) for k, d in unique.items()}

    def _ask(self, batch: list[tuple[str, str]]) -> None:
        self.calls += 1
        response = self.client.post(
            "/chat/completions",
            json={
                "model": self.config.model,
                "messages": _prompt(batch),
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        try:
            results = json.loads(content).get("results", [])
        except (json.JSONDecodeError, AttributeError):
            results = []
        asked = {k for k, _ in batch}
        for item in results if isinstance(results, list) else []:
            key, category = str(item.get("id", "")), str(item.get("category", ""))
            if key not in asked:
                continue  # ignore ids the model invented
            if category not in BY_ID:
                category = UNCLASSIFIED  # ignore categories outside the taxonomy
            try:
                confidence = min(max(float(item.get("confidence", 0.5)), 0.0), 1.0)
            except (TypeError, ValueError):
                confidence = 0.5
            self.cache[key] = (category, confidence if category != UNCLASSIFIED else 0.0)

    def _save(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, indent=0, sort_keys=True))


def apply_llm(lines: pd.DataFrame, classifier: LlmClassifier, min_confidence: float = 0.6) -> pd.DataFrame:
    out = lines.copy()
    todo = out.category_id == UNCLASSIFIED
    if not todo.any():
        return out
    answers = classifier.classify(out.loc[todo, "description"].tolist())
    for idx in out.index[todo]:
        category, confidence = answers[out.at[idx, "description"]]
        if category != UNCLASSIFIED and confidence >= min_confidence:
            out.at[idx, "category_id"] = category
            out.at[idx, "confidence"] = round(confidence * 0.9, 2)
            out.at[idx, "source"] = "llm"
    return out
