"""Rolling suppliers up to their parent group.

Spelling variants are one problem; corporate structure is another. A group
buys through subsidiaries in several countries, under names that share little
("Northbridge Consulting Inc" belongs to the same group as "Kessler & Partner
Beratung GmbH"), and joint ventures belong to two groups at once. Spend per
supplier hides how much really goes to one group, and with it the real
concentration and contract coverage.

The roll-up uses a group map: one row per supplier, its group, and for joint
ventures the ownership share. Suppliers not in the map are their own group.
The map cannot be inferred from names alone, so the module also suggests
likely members to review: suppliers in different countries whose names start
with the same distinctive word.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .text import supplier_key


class GroupMapError(ValueError):
    pass


@dataclass(frozen=True)
class GroupRollup:
    #: supplier_id -> group, with the share of the supplier's spend that goes to that group
    membership: pd.DataFrame
    #: suppliers that look related but are not mapped to the same group
    suggestions: pd.DataFrame
    #: map rows that matched no supplier in the data
    unmatched: list[str]


def load_group_map(path: str | Path) -> pd.DataFrame:
    gm = pd.read_csv(path, dtype=str, keep_default_na=False)
    gm.columns = [c.strip().lower() for c in gm.columns]
    missing = [c for c in ("supplier", "group") if c not in gm.columns]
    if missing:
        raise GroupMapError(f"The group map needs the column{'s' if len(missing) > 1 else ''} {', '.join(missing)}.")
    if "country" not in gm.columns:
        gm["country"] = ""
    gm["share"] = pd.to_numeric(gm.get("share", pd.Series(["1"] * len(gm))).replace("", "1"), errors="coerce")
    if gm["share"].isna().any() or ((gm["share"] <= 0) | (gm["share"] > 1)).any():
        row = int(((gm["share"].isna()) | (gm["share"] <= 0) | (gm["share"] > 1)).to_numpy().argmax()) + 2
        raise GroupMapError(f"Group map row {row}: share must be a number above 0 and at most 1.")
    gm["key"] = gm["supplier"].map(supplier_key)
    totals = gm.groupby(["key", "country"])["share"].sum()
    over = totals[totals > 1.0001]
    if len(over):
        raise GroupMapError(f"Shares for {over.index[0][0]!r} add up to {over.iloc[0]:.2f}; they must not exceed 1.")
    return gm


def roll_up(supplier_map: pd.DataFrame, suppliers: pd.DataFrame, group_map: pd.DataFrame | None) -> GroupRollup:
    """Assign every resolved supplier to one or more groups.

    ``supplier_map`` comes from supplier matching (raw name -> supplier_id, key),
    ``suppliers`` has one row per resolved supplier (supplier_id, supplier, country).
    """
    keys = supplier_map[["supplier_id", "key", "supplier_country"]].drop_duplicates()
    rows: list[dict] = []
    unmatched: list[str] = []
    mapped_ids: set[str] = set()

    if group_map is not None and len(group_map):
        for g in group_map.itertuples():
            hits = keys[(keys.key == g.key) & ((g.country == "") | (keys.supplier_country == g.country))]
            if hits.empty:
                unmatched.append(g.supplier)
                continue
            for sid in hits.supplier_id.unique():
                rows.append({"supplier_id": sid, "group": g.group, "share": float(g.share)})
                mapped_ids.add(sid)

    membership = pd.DataFrame(rows, columns=["supplier_id", "group", "share"])
    # Partially mapped suppliers (a JV with one listed owner) keep the rest themselves.
    if len(membership):
        covered = membership.groupby("supplier_id")["share"].sum()
        rest = covered[covered < 0.9999]
        names = suppliers.set_index("supplier_id")["supplier"]
        membership = pd.concat(
            [
                membership,
                pd.DataFrame({"supplier_id": rest.index, "group": names[rest.index].values, "share": 1 - rest.values}),
            ],
            ignore_index=True,
        )
    alone = suppliers[~suppliers.supplier_id.isin(mapped_ids)]
    membership = pd.concat(
        [membership, pd.DataFrame({"supplier_id": alone.supplier_id, "group": alone.supplier, "share": 1.0})],
        ignore_index=True,
    )
    membership["joint_venture"] = membership.groupby("supplier_id")["supplier_id"].transform("size") > 1

    return GroupRollup(membership=membership, suggestions=_suggest(suppliers, membership), unmatched=unmatched)


def _suggest(suppliers: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Suppliers in different countries whose names start with the same distinctive word
    and that share no group yet."""
    first = suppliers.assign(
        word=suppliers.supplier.map(lambda n: (supplier_key(n).split() or [""])[0]),
        supplier_country=suppliers.country,
    )
    first = first[first.word.str.len() >= 5]
    groups_of = membership.groupby("supplier_id")["group"].agg(set)
    names = suppliers.set_index("supplier_id")
    out = []
    for word, grp in first.groupby("word"):
        if grp.supplier_country.nunique() < 2:
            continue
        sets = [groups_of[s] for s in grp.supplier_id]
        if set.intersection(*sets):
            continue
        out.append(
            {
                "word": word,
                "suppliers": "; ".join(
                    f"{names.loc[s, 'supplier']} ({names.loc[s, 'country']})" for s in grp.supplier_id
                ),
                "spend": float(names.loc[grp.supplier_id, "amount_eur"].sum()),
            }
        )
    return pd.DataFrame(out, columns=["word", "suppliers", "spend"]).sort_values("spend", ascending=False)
