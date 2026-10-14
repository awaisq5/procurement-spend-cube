"""Write the spend cube as one self-contained HTML report plus CSV extracts."""

from __future__ import annotations

import base64
from datetime import date
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from .analytics import Cube
from .charts import coverage_colour, eur_short, pareto_svg, source_bar, treemap_svg
from .pipeline import Result

ASSETS = Path(__file__).resolve().parent / "assets"


def _eur(n: float) -> str:
    return f"€{n:,.0f}"


def render(result: Result, cube: Cube, title: str = "Indirect spend review") -> str:
    env = Environment(loader=PackageLoader("spendcube", "templates"), autoescape=select_autoescape())
    env.filters["eur"] = _eur
    env.filters["eur_short"] = eur_short
    env.filters["pct"] = lambda x: f"{x:.0%}"
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
    columns = [
        "po_number",
        "posting_date",
        "supplier_id",
        "supplier",
        "supplier_name",
        "supplier_country",
        "description",
        "amount",
        "currency",
        "amount_eur",
        "category_id",
        "category",
        "category_group",
        "confidence",
        "source",
        "on_contract",
    ]
    result.lines[columns].to_csv(out_dir / "classified_lines.csv", index=False)
    result.supplier_map.to_csv(out_dir / "supplier_map.csv", index=False)
    result.review.to_csv(out_dir / "review_queue.csv", index=False)
    cube.categories.to_csv(out_dir / "categories.csv", index=False)
    return [
        out_dir / n
        for n in ["index.html", "classified_lines.csv", "supplier_map.csv", "review_queue.csv", "categories.csv"]
    ]
