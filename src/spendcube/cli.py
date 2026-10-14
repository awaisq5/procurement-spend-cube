"""spendcube: from a raw spend export to a classified spend cube.

spendcube run data/spend_lines.csv --fx data/fx_rates.csv --out report
spendcube run data/spend_lines.csv --fx data/fx_rates.csv --out report --llm openrouter
spendcube evaluate data/spend_lines.csv --fx data/fx_rates.csv --truth data/ground_truth.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from .analytics import build_cube
from .classify import PRESETS, LlmClassifier, LlmConfig
from .evaluate import evaluate
from .pipeline import InputError, load_fx, load_lines, run
from .report import write_outputs


def _llm(args: argparse.Namespace) -> LlmClassifier | None:
    if not args.llm:
        return None
    base_url, model = PRESETS[args.llm]
    api_key = os.environ.get("OPENROUTER_API_KEY") if args.llm == "openrouter" else None
    if args.llm == "openrouter" and not api_key:
        raise InputError("Set OPENROUTER_API_KEY to use --llm openrouter, or use --llm ollama for a local model.")
    config = LlmConfig(base_url=base_url, model=args.model or model, api_key=api_key)
    return LlmClassifier(config, cache_path=Path(args.cache))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spendcube", description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("lines", help="spend export CSV")
        p.add_argument("--fx", required=True, help="CSV with currency, eur_per_unit")
        p.add_argument("--llm", choices=sorted(PRESETS), help="classify leftovers with a language model")
        p.add_argument("--model", help="override the preset model name")
        p.add_argument("--cache", default=".spendcube-cache.json", help="where language model answers are cached")
        if name == "run":
            p.add_argument("--out", default="report", help="output folder")
            p.add_argument("--title", default="Indirect spend review")
        else:
            p.add_argument("--truth", required=True, help="CSV with po_number, true_supplier, true_category")
    args = parser.parse_args(argv)

    try:
        lines = load_lines(args.lines, load_fx(args.fx))
        result = run(lines, _llm(args))
        if args.command == "evaluate":
            truth = pd.read_csv(args.truth, dtype={"po_number": str})
            s = evaluate(result.lines, truth)
            print(f"Supplier matching   precision {s.match_precision:.1%}   recall {s.match_recall:.1%}")
            print(f"Classification      precision {s.class_precision:.1%}   coverage {s.class_coverage:.1%}")
            for source, (n, acc) in sorted(s.by_source.items()):
                print(f"  {source:<10} {n:>5} lines   {acc:.1%} correct")
            return 0
        cube = build_cube(result.lines, result.supplier_map)
        files = write_outputs(result, cube, Path(args.out), args.title)
        t = cube.totals
        print(f"{len(lines):,} lines, €{t['spend']:,.0f}")
        print(f"{t['raw_names']} supplier spellings resolved to {t['suppliers']} suppliers")
        print(f"{t['classified_share']:.1%} of spend classified, {len(result.review)} lines to review")
        if result.llm_calls:
            print(f"{result.llm_calls} language model requests")
        print("Wrote " + ", ".join(str(f) for f in files))
        return 0
    except (InputError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
