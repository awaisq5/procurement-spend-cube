"""Measure supplier matching and classification against ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import pandas as pd

from .taxonomy import UNCLASSIFIED


@dataclass(frozen=True)
class Scores:
    #: Of the spelling pairs we merged, how many are really the same supplier.
    match_precision: float
    #: Of the spelling pairs that are the same supplier, how many we merged.
    match_recall: float
    #: Share of classified lines whose category is right.
    class_precision: float
    #: Share of all lines that got the right category.
    class_coverage: float
    by_source: dict[str, tuple[int, float]]


def evaluate(lines: pd.DataFrame, truth: pd.DataFrame) -> Scores:
    df = lines.merge(truth, on=["po_number"], how="inner")

    names = df[["supplier_name", "supplier_country", "supplier_id", "true_supplier"]].drop_duplicates()
    rows = list(names.itertuples(index=False))
    tp = fp = fn = 0
    for a, b in combinations(rows, 2):
        predicted = a.supplier_id == b.supplier_id
        actual = a.true_supplier == b.true_supplier
        tp += predicted and actual
        fp += predicted and not actual
        fn += actual and not predicted

    classified = df[df.category_id != UNCLASSIFIED]
    correct = classified.category_id == classified.true_category
    by_source = {
        str(src): (len(g), float((g.category_id == g.true_category).mean())) for src, g in classified.groupby("source")
    }
    return Scores(
        match_precision=tp / (tp + fp) if tp + fp else 1.0,
        match_recall=tp / (tp + fn) if tp + fn else 1.0,
        class_precision=float(correct.mean()) if len(classified) else 0.0,
        class_coverage=float((df.category_id == df.true_category).mean()),
        by_source=by_source,
    )
