"""Hand-built SVG charts, so the report is one self-contained file with no chart library."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

import pandas as pd

EXPOSED = (194, 125, 26)  # ochre: spend outside contracts
NEUTRAL = (233, 228, 212)
COVERED = (31, 111, 80)  # forest: spend on contract
INK = "#102A23"


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float
    item: dict


def squarify(values: list[float], items: list[dict], x: float, y: float, w: float, h: float) -> list[Rect]:
    """Squarified treemap (Bruls, Huizing, van Wijk 2000): tiles as close to square as possible."""
    total = sum(values)
    if total <= 0:
        return []
    scale = w * h / total
    areas = [v * scale for v in values]
    rects: list[Rect] = []
    queue = list(zip(areas, items, strict=True))

    def worst(row: list[float], side: float) -> float:
        s = sum(row)
        return max(max(side * side * r / (s * s), (s * s) / (side * side * r)) for r in row)

    while queue:
        side = min(w, h)
        row = [queue[0]]
        queue = queue[1:]
        while queue and worst([a for a, _ in row + [queue[0]]], side) <= worst([a for a, _ in row], side):
            row.append(queue[0])
            queue = queue[1:]
        row_area = sum(a for a, _ in row)
        if w >= h:  # lay the row out as a column on the left
            col_w = row_area / h
            cy = y
            for a, item in row:
                rects.append(Rect(x, cy, col_w, a / col_w, item))
                cy += a / col_w
            x, w = x + col_w, w - col_w
        else:  # lay the row out across the top
            row_h = row_area / w
            cx = x
            for a, item in row:
                rects.append(Rect(cx, y, a / row_h, row_h, item))
                cx += a / row_h
            y, h = y + row_h, h - row_h
    return rects


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def coverage_colour(share: float) -> str:
    """Ochre for spend outside contracts, through neutral, to forest for full coverage."""
    share = min(max(share, 0.0), 1.0)
    rgb = _mix(EXPOSED, NEUTRAL, share / 0.5) if share < 0.5 else _mix(NEUTRAL, COVERED, (share - 0.5) / 0.5)
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _text_colour(hex_colour: str) -> str:
    r, g, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#FFFFFF" if luminance < 0.5 else INK


def _fits(text: str, font_size: float, room: float) -> bool:
    """Rough width check: Public Sans averages about 0.56 em per character."""
    return len(text) * font_size * 0.56 <= room


def eur_short(n: float) -> str:
    if n >= 1_000_000:
        return f"€{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"€{n / 1_000:.0f}k"
    return f"€{n:.0f}"


def treemap_svg(categories: pd.DataFrame, width: int = 1120, height: int = 500) -> str:
    rows = categories.sort_values("spend", ascending=False).to_dict("records")
    rects = squarify([r["spend"] for r in rows], rows, 0, 0, width, height)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Spend by category, coloured by contract coverage">'
    ]
    for r in rects:
        c = r.item
        fill = coverage_colour(c["on_contract_share"])
        fg = _text_colour(fill)
        label = escape(c["category"])
        tip = escape(
            f"{c['category']}: {eur_short(c['spend'])}, {c['suppliers']} suppliers, "
            f"{c['on_contract_share']:.0%} on contract"
        )
        parts.append(
            f'<g><title>{tip}</title><rect x="{r.x:.1f}" y="{r.y:.1f}" width="{r.w:.1f}" height="{r.h:.1f}" '
            f'fill="{fill}" stroke="#EEF2EF" stroke-width="2"/>'
        )
        name_size = 17 if r.w * r.h > 60_000 else 14
        detail_size = name_size - 2
        room = r.w - 18
        details = [
            f"{eur_short(c['spend'])}, {c['on_contract_share']:.0%} on contract",
            f"{eur_short(c['spend'])}, {c['on_contract_share']:.0%}",
            eur_short(c["spend"]),
        ]
        detail = next((d for d in details if _fits(d, detail_size, room)), None)
        if r.h > 46 and _fits(c["category"], name_size, room) and detail:
            parts.append(
                f'<text x="{r.x + 10:.1f}" y="{r.y + 10 + name_size:.1f}" fill="{fg}" font-size="{name_size}" '
                f'font-weight="650">{label}</text>'
                f'<text x="{r.x + 10:.1f}" y="{r.y + 14 + name_size * 2.1:.1f}" fill="{fg}" font-size="{detail_size}" '
                f'opacity="0.85">{escape(detail)}</text>'
            )
        elif r.h > 24 and _fits(eur_short(c["spend"]), 12, r.w - 10):
            parts.append(
                f'<text x="{r.x + 6:.1f}" y="{r.y + 17:.1f}" fill="{fg}" font-size="12" font-weight="600">'
                f"{eur_short(c['spend'])}</text>"
            )
        parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)


def pareto_svg(pareto: pd.DataFrame, highlight: int, width: int = 560, height: int = 300) -> str:
    """Cumulative share of spend against number of suppliers, largest first."""
    pad_l, pad_b, pad_t, pad_r = 46, 34, 12, 12
    n = int(pareto["rank"].max())
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b

    def x(rank: float) -> float:
        return pad_l + (rank / n) * pw

    def y(share: float) -> float:
        return pad_t + (1 - share) * ph

    points = " ".join(f"{x(r):.1f},{y(s):.1f}" for r, s in zip(pareto["rank"], pareto.cumulative_share, strict=True))
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Cumulative spend by supplier rank">']
    for s in (0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y(s):.1f}" y2="{y(s):.1f}" class="grid"/>')
        out.append(f'<text x="{pad_l - 8}" y="{y(s) + 4:.1f}" text-anchor="end" class="axis">{s:.0%}</text>')
    for r in range(0, n + 1, 25 if n > 60 else 10):
        out.append(f'<text x="{x(r):.1f}" y="{height - 12}" text-anchor="middle" class="axis">{r}</text>')
    out.append(f'<polyline points="{x(0):.1f},{y(0):.1f} {points}" fill="none" stroke="{INK}" stroke-width="2.5"/>')
    hx, hy = x(highlight), y(0.8)
    out.append(f'<line x1="{hx:.1f}" x2="{hx:.1f}" y1="{hy:.1f}" y2="{y(0):.1f}" class="marker"/>')
    out.append(f'<line x1="{pad_l}" x2="{hx:.1f}" y1="{hy:.1f}" y2="{hy:.1f}" class="marker"/>')
    out.append(f'<circle cx="{hx:.1f}" cy="{hy:.1f}" r="5" fill="#1F6F50"/>')
    out.append(
        f'<text x="{hx + 10:.1f}" y="{hy + 18:.1f}" class="callout">{highlight} suppliers hold 80% of spend</text>'
    )
    out.append("</svg>")
    return "".join(out)


def source_bar(totals: dict) -> list[dict]:
    """Segments for the 'how was this classified' bar."""
    segments = [
        ("Keyword rules", totals["rule_share"], "#1F6F50"),
        ("Supplier profile", totals["supplier_share"], "#5E9E82"),
        ("Language model", totals["llm_share"], "#9CC5B2"),
        ("Needs review", 1 - totals["classified_share"], "#C27D1A"),
    ]
    return [{"label": label, "share": share, "colour": colour} for label, share, colour in segments if share > 0.0005]
