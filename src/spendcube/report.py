"""Write the spend cube as one self-contained HTML report plus CSV extracts."""

from __future__ import annotations

import base64
from datetime import date
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from .analytics import Cube
from .charts import coverage_colour, eur_short, pareto_svg, source_bar, treemap_svg
from .pipeline import Result
from .workbook import LINE_COLUMNS, write_workbook

ASSETS = Path(__file__).resolve().parent / "assets"


def _eur(n: float) -> str:
    return f"€{n:,.0f}"


def render(result: Result, cube: Cube, title: str = "Indirect spend review") -> str:
    env = Environment(loader=PackageLoader("spendcube", "templates"), autoescape=select_autoescape())
    env.filters["eur"] = _eur
    env.filters["eur_short"] = eur_short
    env.filters["pct"] = lambda x: f"{x:.0%}"
    env.filters["signed_pct"] = lambda x: "new" if x != x else f"{x:+.0%}"  # NaN: no spend last year
    env.filters["signed_eur"] = lambda n: f"{'+' if n >= 0 else '−'}€{abs(n):,.0f}"
    env.filters["day"] = lambda iso: f"{date.fromisoformat(iso).day} {date.fromisoformat(iso):%B %Y}"
    env.filters["colour"] = coverage_colour
    font = base64.b64encode((ASSETS / "PublicSans.woff2").read_bytes()).decode()
    t = cube.totals
    return env.get_template("report.html.j2").render(
        title=title,
        font=font,
        t=t,
        first=t["first_date"].strftime("%B %Y"),
        last=t["last_date"].strftime("%B %Y"),
        categories=cube.categories.to_dict("records"),
        findings=cube.categories[cube.categories.finding != ""]
        .sort_values("at_stake", ascending=False)
        .head(6)
        .to_dict("records"),
        merges=cube.merges.head(8).to_dict("records"),
        yoy=cube.year_on_year.head(10).to_dict("records"),
        yoy_period=t.get("yoy_period"),
        groups=cube.groups[cube.groups.members > 1].to_dict("records"),
        suggestions=result.groups.suggestions.to_dict("records"),
        scope=sorted(set(result.lines.category_group) - {"Unclassified"}),
        review=result.review.head(12).to_dict("records"),
        review_count=len(result.review),
        review_value=float(result.review.amount_eur.sum()),
        treemap=treemap_svg(cube.categories),
        pareto=pareto_svg(cube.pareto, t["suppliers_for_80"]),
        sources=source_bar(t),
        legend=[coverage_colour(s / 10) for s in range(11)],
        generated=date.today().isoformat(),
    )


def write_outputs(result: Result, cube: Cube, out_dir: Path, title: str = "Indirect spend review") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "index.html": render(result, cube, title),
    }
    for name, text in files.items():
        (out_dir / name).write_text(text, encoding="utf-8")
    result.lines[LINE_COLUMNS].to_csv(out_dir / "classified_lines.csv", index=False)
    result.supplier_map.to_csv(out_dir / "supplier_map.csv", index=False)
    result.review.to_csv(out_dir / "review_queue.csv", index=False)
    cube.categories.to_csv(out_dir / "categories.csv", index=False)
    if len(cube.year_on_year):
        cube.year_on_year.to_csv(out_dir / "year_on_year.csv", index=False)
    cube.groups.assign(member_names=cube.groups.member_names.map("; ".join)).to_csv(out_dir / "groups.csv", index=False)
    write_workbook(result, cube, out_dir / "spend_cube.xlsx")
    return [
        out_dir / n
        for n in [
            "index.html",
            "spend_cube.xlsx",
            "classified_lines.csv",
            "supplier_map.csv",
            "review_queue.csv",
            "categories.csv",
            *(["year_on_year.csv"] if len(cube.year_on_year) else []),
            "groups.csv",
        ]
    ]
