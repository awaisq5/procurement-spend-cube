import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from spendcube.analytics import build_cube
from spendcube.classify import LlmClassifier, LlmConfig, apply_llm, apply_supplier_profile, classify_description
from spendcube.evaluate import evaluate
from spendcube.groups import GroupMapError, load_group_map, roll_up
from spendcube.pipeline import InputError, load_fx, load_lines, run
from spendcube.report import render
from spendcube.suppliers import match_suppliers
from spendcube.text import supplier_key

DATA = Path(__file__).resolve().parent.parent / "data"
REQUIRED_COLS = [
    "po_number",
    "posting_date",
    "supplier_name",
    "supplier_country",
    "description",
    "amount",
    "currency",
    "amount_eur",
    "contract_id",
    "on_contract",
]


def frame(names, country="DE", amounts=None):
    return pd.DataFrame(
        {
            "supplier_name": names,
            "supplier_country": [country] * len(names),
            "amount_eur": amounts or [100.0] * len(names),
        }
    )


class TestSupplierKeys:
    @pytest.mark.parametrize(
        "raw",
        [
            "Müller Engineering GmbH & Co. KG",
            "MUELLER ENGINEERING GMBH",
            "Mueller Engg. GmbH u. Co. KG",
            "Müller Engineering (Celle)",
        ],
    )
    def test_variants_reduce_to_the_same_core(self, raw):
        assert supplier_key(raw).startswith("mueller engineering")

    def test_keeps_words_that_identify_the_business(self):
        assert supplier_key("Becker Elektro GmbH") == "becker elektro"


class TestMatching:
    def test_merges_spellings_of_one_supplier(self):
        names = ["Tidewell Engineering Ltd", "TIDEWELL ENGINEERING LTD", "Tidewell Engineeirng Ltd", "Tidewell Engg."]
        result = match_suppliers(frame(names, "GB"))
        assert result.suppliers.shape[0] == 1
        assert result.suppliers.variants.iloc[0] == 4

    def test_keeps_different_trades_of_the_same_family_name_apart(self):
        result = match_suppliers(frame(["Becker Elektro", "Becker Druck GmbH", "Becker Events"]))
        assert result.suppliers.shape[0] == 3

    def test_does_not_merge_across_countries(self):
        lines = pd.concat([frame(["Atlas Forwarding Ltd"], "GB"), frame(["Atlas Forwarding GmbH"], "DE")])
        assert match_suppliers(lines).suppliers.shape[0] == 2

    def test_display_name_is_the_spelling_with_most_spend(self):
        result = match_suppliers(frame(["SWIFTPOST KURIER", "Swiftpost Kurier GmbH"], amounts=[10.0, 500.0]))
        assert result.suppliers.supplier.iloc[0] == "Swiftpost Kurier GmbH"


class TestRules:
    @pytest.mark.parametrize(
        ("text", "category"),
        [
            ("Gebäudereinigung März", "FM-CLN"),
            ("Office cleaning Feb", "FM-CLN"),
            ("Arbeitnehmerüberlassung Juni", "LB-CTR"),
            ("Wartung Klimaanlage Halle 3", "FM-MNT"),
            ("Air freight Houston", "LG-FWD"),
            ("Lizenzverlängerung 2026", "IT-SW"),
        ],
    )
    def test_english_and_german(self, text, category):
        assert classify_description(text)[0] == category

    def test_vague_lines_stay_unclassified(self):
        assert classify_description("Leistungen gem. Bestellung")[0] == "UNCLASSIFIED"

    def test_ties_stay_unclassified(self):
        assert classify_description("Hotel and flight Aberdeen")[0] == "UNCLASSIFIED"

    def test_supplier_profile_fills_vague_lines(self):
        lines = pd.DataFrame(
            {
                "supplier_id": ["S1"] * 5,
                "amount_eur": [100.0] * 5,
                "category_id": ["FM-CLN"] * 4 + ["UNCLASSIFIED"],
                "confidence": [0.95] * 4 + [0.0],
                "source": ["rule"] * 4 + ["none"],
            }
        )
        out = apply_supplier_profile(lines)
        assert out.category_id.iloc[-1] == "FM-CLN"
        assert out.source.iloc[-1] == "supplier"


class TestLanguageModel:
    def make(self, answer, sent, tmp_path):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            sent.append(body)
            ids = [line.split(":")[0] for line in body["messages"][1]["content"].splitlines()]
            results = answer(ids)
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"results": results})}}]})

        client = httpx.Client(base_url="http://llm.test", transport=httpx.MockTransport(handler))
        return LlmClassifier(LlmConfig("http://llm.test", "test-model"), tmp_path / "cache.json", client)

    def lines(self):
        return pd.DataFrame(
            {
                "supplier_name": ["Secret Supplier GmbH"],
                "amount_eur": [98765.43],
                "description": ["Q3 charges"],
                "category_id": ["UNCLASSIFIED"],
                "confidence": [0.0],
                "source": ["none"],
            }
        )

    def test_uses_valid_answers_and_sends_only_descriptions(self, tmp_path):
        sent = []
        llm = self.make(lambda ids: [{"id": ids[0], "category": "FM-SEC", "confidence": 0.9}], sent, tmp_path)
        out = apply_llm(self.lines(), llm)
        assert out.category_id.iloc[0] == "FM-SEC"
        assert out.source.iloc[0] == "llm"
        payload = json.dumps(sent)
        assert "Secret Supplier" not in payload
        assert "98765" not in payload

    def test_rejects_categories_outside_the_taxonomy_and_invented_ids(self, tmp_path):
        sent = []
        llm = self.make(
            lambda ids: [
                {"id": ids[0], "category": "BANANAS", "confidence": 0.99},
                {"id": "made-up", "category": "FM-SEC", "confidence": 0.99},
            ],
            sent,
            tmp_path,
        )
        out = apply_llm(self.lines(), llm)
        assert out.category_id.iloc[0] == "UNCLASSIFIED"

    def test_ignores_low_confidence(self, tmp_path):
        llm = self.make(lambda ids: [{"id": ids[0], "category": "FM-SEC", "confidence": 0.3}], [], tmp_path)
        assert apply_llm(self.lines(), llm).category_id.iloc[0] == "UNCLASSIFIED"

    def test_caches_answers_between_runs(self, tmp_path):
        sent = []
        answer = lambda ids: [{"id": ids[0], "category": "FM-SEC", "confidence": 0.9}]  # noqa: E731
        apply_llm(self.lines(), self.make(answer, sent, tmp_path))
        second = self.make(answer, sent, tmp_path)
        apply_llm(self.lines(), second)
        assert len(sent) == 1
        assert second.calls == 0


@pytest.fixture(scope="module")
def result():
    lines = load_lines(DATA / "spend_lines.csv", load_fx(DATA / "fx_rates.csv"))
    return run(lines)


class TestPipeline:
    def test_accuracy_on_sample_data(self, result):
        truth = pd.read_csv(DATA / "ground_truth.csv", dtype={"po_number": str})
        scores = evaluate(result.lines, truth)
        assert scores.match_precision >= 0.99
        assert scores.match_recall >= 0.95
        assert scores.class_precision >= 0.98
        assert scores.class_coverage >= 0.85

    def test_cube_totals_reconcile_with_lines(self, result):
        cube = build_cube(result.lines, result.supplier_map)
        assert cube.totals["spend"] == pytest.approx(result.lines.amount_eur.sum())
        classified = result.lines[result.lines.category_id != "UNCLASSIFIED"].amount_eur.sum()
        assert cube.categories.spend.sum() == pytest.approx(classified)
        assert cube.pareto.cumulative_share.iloc[-1] == pytest.approx(1.0)

    def test_report_renders(self, result):
        html = render(result, build_cube(result.lines, result.supplier_map))
        assert "bought outside a contract" in html
        assert "<svg" in html

    def test_names_the_missing_exchange_rate(self, tmp_path):
        bad = tmp_path / "lines.csv"
        bad.write_text(
            "po_number,posting_date,supplier_name,supplier_country,description,amount,currency\n"
            "1,2026-01-05,A GmbH,DE,Toner,10,CHF\n"
        )
        with pytest.raises(InputError, match="CHF"):
            load_lines(bad, {"EUR": 1.0})


class TestGroups:
    def suppliers(self, rows):
        lines = pd.DataFrame(rows, columns=["supplier_name", "supplier_country", "amount_eur"])
        return match_suppliers(lines)

    def gm(self, text, tmp_path):
        p = tmp_path / "groups.csv"
        p.write_text(text)
        return load_group_map(p)

    def test_rolls_differently_named_subsidiaries_up_to_one_group(self, tmp_path):
        m = self.suppliers([("Kessler Beratung GmbH", "DE", 100.0), ("Northbridge Consulting Inc", "US", 50.0)])
        gm = self.gm(
            "supplier,country,group\nKessler Beratung GmbH,DE,Kessler Group\nNorthbridge Consulting,US,Kessler Group\n",
            tmp_path,
        )
        r = roll_up(m.mapping, m.suppliers, gm)
        assert set(r.membership.group) == {"Kessler Group"}
        assert r.unmatched == []

    def test_splits_a_joint_venture_between_its_owners(self, tmp_path):
        m = self.suppliers([("Alpha Beta JV AS", "NO", 100.0)])
        gm = self.gm(
            "supplier,country,group,share\nAlpha Beta JV AS,NO,Alpha,0.6\nAlpha Beta JV AS,NO,Beta,0.4\n", tmp_path
        )
        r = roll_up(m.mapping, m.suppliers, gm)
        assert dict(zip(r.membership.group, r.membership.share, strict=True)) == {"Alpha": 0.6, "Beta": 0.4}
        assert r.membership.joint_venture.all()

    def test_refuses_shares_above_one(self, tmp_path):
        with pytest.raises(GroupMapError, match="add up to 1.30"):
            self.gm("supplier,group,share\nX AS,A,0.7\nX AS,B,0.6\n", tmp_path)

    def test_reports_map_rows_that_match_nothing(self, tmp_path):
        m = self.suppliers([("Alpha AS", "NO", 1.0)])
        r = roll_up(m.mapping, m.suppliers, self.gm("supplier,group\nGhost Ltd,G\n", tmp_path))
        assert r.unmatched == ["Ghost Ltd"]

    def test_suggests_same_named_suppliers_in_other_countries(self):
        m = self.suppliers([("Brightwater Advisory Ltd", "GB", 10.0), ("Brightwater Advisory Inc", "US", 5.0)])
        r = roll_up(m.mapping, m.suppliers, None)
        assert "Brightwater Advisory Inc (US)" in r.suggestions.suppliers.iloc[0]

    def test_group_concentration_reveals_what_supplier_concentration_hides(self, result):
        grouped = run(result.lines[REQUIRED_COLS], group_map=load_group_map(DATA / "supplier_groups.csv"))
        cube = build_cube(grouped.lines, grouped.supplier_map, grouped.groups.membership)
        consulting = cube.categories.set_index("category").loc["Consulting"]
        assert consulting.group_top_share > consulting.top_share
        assert consulting.top_group == "Kessler Northbridge Group"
        assert "through 3 of them" in consulting.finding
        assert cube.totals["spend"] == pytest.approx(cube.groups.spend.sum())

    def test_scope_limits_the_cube(self, result):
        scoped = result.scoped(["Professional services", "Labour"])
        assert set(scoped.lines.category_group) == {"Professional services", "Labour"}
        cube = build_cube(scoped.lines, scoped.supplier_map, scoped.groups.membership)
        assert cube.totals["spend"] == pytest.approx(scoped.lines.amount_eur.sum())


def test_cli_runs_with_groups_and_scope(tmp_path, capsys):
    from spendcube.cli import main

    args = ["run", str(DATA / "spend_lines.csv"), "--fx", str(DATA / "fx_rates.csv"), "--out", str(tmp_path)]
    code = main([*args, "--groups", str(DATA / "supplier_groups.csv"), "--scope", "Professional services,Labour"])
    out = capsys.readouterr().out
    assert code == 0
    assert "in 22 groups" in out
    assert (tmp_path / "groups.csv").exists()
    assert main([*args, "--scope", "Consultng"]) == 2
