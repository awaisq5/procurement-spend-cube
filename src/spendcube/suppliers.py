"""Supplier normalisation: many spellings, one supplier.

Spend exported from an ERP lists the same supplier under several names:
legal-form variants, upper case, transliterated umlauts, typos and
abbreviations. They are merged in two steps:

1. Exact match on a normalised key (legal form, punctuation, location removed).
2. Fuzzy match between keys, within the same country, joined with union-find.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz, process

from .text import supplier_key


@dataclass(frozen=True)
class SupplierMatch:
    #: raw supplier name and country -> supplier id
    mapping: pd.DataFrame
    #: one row per resolved supplier
    suppliers: pd.DataFrame


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def match_suppliers(lines: pd.DataFrame, threshold: float = 90.0) -> SupplierMatch:
    """Resolve raw supplier names in ``lines`` (columns supplier_name, supplier_country, amount_eur)."""
    raw = (
        lines.groupby(["supplier_name", "supplier_country"], as_index=False)
        .agg(amount_eur=("amount_eur", "sum"), lines=("amount_eur", "size"))
        .assign(key=lambda d: d.supplier_name.map(supplier_key))
    )

    keys = raw[["key", "supplier_country"]].drop_duplicates().reset_index(drop=True)
    uf = _UnionFind(len(keys))
    for _country, group in keys.groupby("supplier_country"):
        idx = group.index.to_numpy()
        names = group.key.tolist()
        if len(names) < 2:
            continue
        # token_sort_ratio ignores word order; ratio catches single-character typos.
        scores = process.cdist(names, names, scorer=fuzz.token_sort_ratio, score_cutoff=threshold)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if scores[i, j] >= threshold and _compatible(names[i], names[j]):
                    uf.union(idx[i], idx[j])

    keys["cluster"] = [uf.find(i) for i in range(len(keys))]
    raw = raw.merge(keys, on=["key", "supplier_country"])

    # The display name is the cleanest spelling: not all capitals, no location suffix, then the most used.
    raw["clean"] = ~raw.supplier_name.str.isupper() & ~raw.supplier_name.str.contains(r"\(", regex=True)
    canonical = (
        raw.sort_values(["clean", "lines", "amount_eur"], ascending=[False, False, False])
        .groupby("cluster")
        .agg(
            supplier=("supplier_name", "first"),
            country=("supplier_country", "first"),
            amount_eur=("amount_eur", "sum"),
            lines=("lines", "sum"),
            variants=("supplier_name", "nunique"),
        )
        .reset_index()
        .sort_values("amount_eur", ascending=False)
        .reset_index(drop=True)
    )
    canonical["supplier_id"] = [f"S{i + 1:04d}" for i in range(len(canonical))]
    raw = raw.merge(canonical[["cluster", "supplier_id", "supplier"]], on="cluster")
    mapping = raw[["supplier_name", "supplier_country", "supplier_id", "supplier", "key"]]
    return SupplierMatch(mapping=mapping, suppliers=canonical.drop(columns="cluster"))


def _compatible(a: str, b: str) -> bool:
    """Guard against merging different suppliers that share most words.

    "nordwind logistik" and "nordwind elektro" score high on token similarity
    but differ in the word that says what they do. Require the longest
    distinguishing words to be near-identical.
    """
    ta, tb = set(a.split()), set(b.split())
    only_a, only_b = ta - tb, tb - ta
    if not only_a and not only_b:
        return True
    if not only_a or not only_b:
        # One name is the other plus extra words, e.g. "kessler partner" vs "kessler partner beratung".
        return True
    a_rest, b_rest = " ".join(sorted(only_a)), " ".join(sorted(only_b))
    # In short words one letter is a large share of the identity: "Schulz" and "Scholz" are different people.
    threshold = 85 if max(len(a_rest), len(b_rest)) <= 6 else 75
    return fuzz.ratio(a_rest, b_rest) >= threshold
