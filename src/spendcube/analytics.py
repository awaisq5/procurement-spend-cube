"""The spend cube itself: SQL over the classified lines, run in DuckDB."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd


@dataclass(frozen=True)
class Cube:
    totals: dict
    categories: pd.DataFrame
    suppliers: pd.DataFrame
    pareto: pd.DataFrame
    quarters: pd.DataFrame
    merges: pd.DataFrame


def build_cube(lines: pd.DataFrame, supplier_map: pd.DataFrame) -> Cube:
    con = duckdb.connect()
    con.register("lines", lines)
    con.register("supplier_map", supplier_map)

    totals = (
        con.sql(
            """
        SELECT
          SUM(amount_eur)                                                   AS spend,
          COUNT(*)                                                          AS lines,
          COUNT(DISTINCT supplier_id)                                       AS suppliers,
          (SELECT COUNT(DISTINCT supplier_name || '|' || supplier_country) FROM lines) AS raw_names,
          COALESCE(SUM(amount_eur) FILTER (WHERE category_id <> 'UNCLASSIFIED'), 0) / SUM(amount_eur) AS classified_share,
          COALESCE(SUM(amount_eur) FILTER (WHERE source = 'rule'), 0)     / SUM(amount_eur) AS rule_share,
          COALESCE(SUM(amount_eur) FILTER (WHERE source = 'supplier'), 0) / SUM(amount_eur) AS supplier_share,
          COALESCE(SUM(amount_eur) FILTER (WHERE source = 'llm'), 0)      / SUM(amount_eur) AS llm_share,
          COALESCE(SUM(amount_eur) FILTER (WHERE on_contract), 0) / SUM(amount_eur) AS on_contract_share,
          MIN(posting_date) AS first_date,
          MAX(posting_date) AS last_date
        FROM lines
        """
        )
        .df()
        .iloc[0]
        .to_dict()
    )

    # Per category: concentration (HHI on a 0-10,000 scale) and contract coverage.
    categories = con.sql(
        """
        WITH by_supplier AS (
          SELECT category_id, category, category_group, supplier_id, supplier,
                 SUM(amount_eur) AS spend, BOOL_OR(on_contract) AS has_contract,
                 SUM(amount_eur) FILTER (WHERE on_contract) AS contract_spend
          FROM lines WHERE category_id <> 'UNCLASSIFIED'
          GROUP BY ALL
        ),
        shares AS (
          SELECT *, spend / SUM(spend) OVER (PARTITION BY category_id) AS share,
                 ROW_NUMBER() OVER (PARTITION BY category_id ORDER BY spend DESC) AS rank
          FROM by_supplier
        )
        SELECT category_id, ANY_VALUE(category) AS category, ANY_VALUE(category_group) AS category_group,
               SUM(spend) AS spend,
               COUNT(*) AS suppliers,
               MAX(share) AS top_share,
               ANY_VALUE(supplier) FILTER (WHERE rank = 1) AS top_supplier,
               ROUND(10000 * SUM(share * share)) AS hhi,
               COALESCE(SUM(contract_spend), 0) / SUM(spend) AS on_contract_share,
               BOOL_OR(has_contract) AS contract_exists
        FROM shares
        GROUP BY category_id
        ORDER BY spend DESC
        """
    ).df()
    findings = categories.apply(_finding, axis=1, result_type="expand")
    categories["finding"], categories["at_stake"] = findings[0], findings[1]

    suppliers = con.sql(
        """
        SELECT supplier_id, ANY_VALUE(supplier) AS supplier, ANY_VALUE(supplier_country) AS country,
               SUM(amount_eur) AS spend, COUNT(*) AS lines, COUNT(DISTINCT category_id) AS categories,
               BOOL_OR(on_contract) AS has_contract
        FROM lines GROUP BY supplier_id ORDER BY spend DESC
        """
    ).df()

    pareto = con.sql(
        """
        SELECT ROW_NUMBER() OVER (ORDER BY spend DESC) AS rank,
               SUM(spend) OVER (ORDER BY spend DESC) / SUM(spend) OVER () AS cumulative_share
        FROM (SELECT supplier_id, SUM(amount_eur) AS spend FROM lines GROUP BY supplier_id)
        ORDER BY rank
        """
    ).df()

    quarters = con.sql(
        """
        SELECT quarter, category_group, SUM(amount_eur) AS spend
        FROM lines GROUP BY ALL ORDER BY quarter, spend DESC
        """
    ).df()

    merges = con.sql(
        """
        SELECT m.supplier_id, ANY_VALUE(m.supplier) AS supplier,
               LIST(DISTINCT m.supplier_name ORDER BY m.supplier_name) AS spellings,
               COUNT(DISTINCT m.supplier_name) AS variants,
               ANY_VALUE(s.spend) AS spend
        FROM supplier_map m JOIN (SELECT supplier_id, SUM(amount_eur) AS spend FROM lines GROUP BY 1) s USING (supplier_id)
        GROUP BY m.supplier_id HAVING COUNT(DISTINCT m.supplier_name) > 1
        ORDER BY spend DESC
        """
    ).df()

    # Tail: suppliers that together make up the smallest 5% of spend.
    tail_cut = pareto.cumulative_share > 0.95
    totals["tail_suppliers"] = int(tail_cut.sum())
    totals["suppliers_for_80"] = int((pareto.cumulative_share < 0.80).sum() + 1)
    totals["maverick_spend"] = float(
        con.sql(
            """
            SELECT COALESCE(SUM(l.amount_eur), 0) FROM lines l
            WHERE NOT l.on_contract AND l.category_id IN (
              SELECT DISTINCT category_id FROM lines WHERE on_contract)
            """
        ).fetchone()[0]
    )
    con.close()
    return Cube(totals, categories, suppliers, pareto, quarters, merges)


MATERIAL_EUR = 100_000


def _finding(row: pd.Series) -> tuple[str, float]:
    """One plain sentence a category manager can act on, and the euros it concerns.

    Small categories get none. A single supplier under contract is a deliberate
    choice, so concentration is only flagged where no contract governs it.
    """
    if row.spend < MATERIAL_EUR:
        return "", 0.0
    off_contract = row.spend * (1 - row.on_contract_share)
    if not row.contract_exists and (row.suppliers == 1 or row.top_share >= 0.9):
        return (
            f"{row.top_supplier} supplies {row.top_share:.0%} with no contract: no agreed terms, no fallback.",
            row.spend,
        )
    if not row.contract_exists:
        return f"No contract in place. {row.suppliers} suppliers, the largest holds {row.top_share:.0%}.", row.spend
    if row.on_contract_share < 0.6:
        return f"A contract exists, but {1 - row.on_contract_share:.0%} of spend bypasses it.", off_contract
    if row.hhi < 1500 and row.suppliers >= 8:
        return f"Spread across {row.suppliers} suppliers: consolidating would add leverage.", row.spend
    return "", 0.0
