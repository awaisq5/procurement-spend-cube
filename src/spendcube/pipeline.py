"""From a raw spend export to clean, classified lines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .classify import LlmClassifier, apply_llm, apply_rules, apply_supplier_profile
from .suppliers import match_suppliers
from .taxonomy import BY_ID, UNCLASSIFIED

REQUIRED = ["po_number", "posting_date", "supplier_name", "supplier_country", "description", "amount", "currency"]


class InputError(ValueError):
    pass


@dataclass(frozen=True)
class Result:
    lines: pd.DataFrame
    suppliers: pd.DataFrame
    supplier_map: pd.DataFrame
    review: pd.DataFrame
    llm_calls: int = 0


def load_lines(path: str | Path, fx: dict[str, float]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"po_number": str, "contract_id": str}, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise InputError(f"The export is missing the column{'s' if len(missing) > 1 else ''} {', '.join(missing)}.")
    df["amount"] = pd.to_numeric(df.amount, errors="coerce")
    if df.amount.isna().any():
        row = int(df.amount.isna().to_numpy().argmax()) + 2
        raise InputError(f"Row {row}: amount is not a number.")
    unknown = sorted(set(df.currency) - set(fx))
    if unknown:
        raise InputError(f"No exchange rate for {', '.join(unknown)}. Add it to the rates file.")
    df["posting_date"] = pd.to_datetime(df.posting_date, errors="coerce")
    if df.posting_date.isna().any():
        row = int(df.posting_date.isna().to_numpy().argmax()) + 2
        raise InputError(f"Row {row}: posting_date is not a date (expected YYYY-MM-DD).")
    df["amount_eur"] = df.amount * df.currency.map(fx)
    if "contract_id" not in df.columns:
        df["contract_id"] = ""
    df["on_contract"] = df.contract_id.str.strip() != ""
    df["description"] = df.description.astype(str)
    df["supplier_name"] = df.supplier_name.astype(str).str.strip()
    return df


def load_fx(path: str | Path) -> dict[str, float]:
    rates = pd.read_csv(path)
    return dict(zip(rates.currency.str.upper(), rates.eur_per_unit.astype(float), strict=True))


def run(lines: pd.DataFrame, llm: LlmClassifier | None = None, review_below: float = 0.6) -> Result:
    match = match_suppliers(lines)
    df = lines.merge(
        match.mapping[["supplier_name", "supplier_country", "supplier_id", "supplier"]],
        on=["supplier_name", "supplier_country"],
        how="left",
    )
    df = apply_rules(df)
    df = apply_supplier_profile(df)
    if llm is not None:
        df = apply_llm(df, llm)

    df["category"] = df.category_id.map(lambda c: BY_ID[c].name if c in BY_ID else "Unclassified")
    df["category_group"] = df.category_id.map(lambda c: BY_ID[c].group if c in BY_ID else "Unclassified")
    df["quarter"] = df.posting_date.dt.year.astype(str) + "-Q" + df.posting_date.dt.quarter.astype(str)

    review = (
        df[(df.category_id == UNCLASSIFIED) | (df.confidence < review_below)]
        .sort_values("amount_eur", ascending=False)
        .loc[
            :,
            [
                "po_number",
                "posting_date",
                "supplier",
                "description",
                "amount_eur",
                "category_id",
                "confidence",
                "source",
            ],
        ]
    )
    return Result(
        lines=df,
        suppliers=match.suppliers,
        supplier_map=match.mapping,
        review=review,
        llm_calls=llm.calls if llm else 0,
    )
