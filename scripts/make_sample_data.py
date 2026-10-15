"""Generate a synthetic indirect-spend export with known ground truth.

The export imitates what comes out of an ERP: one supplier under several
spellings, vague line descriptions, English and German text, four currencies.
``ground_truth.csv`` records the real supplier and category of every line so
matching and classification accuracy can be measured. All names are invented.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np

rng = np.random.default_rng(365)
OUT = Path(__file__).resolve().parent.parent / "data"

# name, country, {category: weight}, yearly lines, has contract
MAJOR = [
    ("Brightwater Advisory Ltd", "GB", {"PS-CON": 1}, 22, False),
    ("Kessler & Partner Beratung GmbH", "DE", {"PS-CON": 1}, 30, False),
    ("Northbridge Consulting Inc", "US", {"PS-CON": 0.8, "HR-TRN": 0.2}, 14, False),
    ("KP Advisory UK Ltd", "GB", {"PS-CON": 1}, 18, False),
    ("Brightwater Advisory Inc", "US", {"PS-CON": 1}, 10, False),
    ("Hartmann Rechtsanwälte PartG mbB", "DE", {"PS-LEG": 1}, 18, False),
    ("Calder Fyfe LLP", "GB", {"PS-LEG": 1}, 12, False),
    ("Albrecht Treuhand GmbH", "DE", {"PS-AUD": 1}, 10, True),
    ("Müller Engineering GmbH & Co. KG", "DE", {"PS-ENG": 1}, 60, True),
    ("Tidewell Engineering Ltd", "GB", {"PS-ENG": 1}, 34, True),
    ("Stavik Ingeniørtjenester AS", "NO", {"PS-ENG": 1}, 20, False),
    ("Tidewell Stavik Engineering JV AS", "NO", {"PS-ENG": 1}, 18, True),
    ("FlexForce Personaldienstleistungen GmbH", "DE", {"LB-CTR": 1}, 96, True),
    ("Granite Staffing Ltd", "GB", {"LB-CTR": 1}, 60, True),
    ("Nordlys Bemanning AS", "NO", {"LB-CTR": 1}, 30, False),
    ("Granite Staffing Middle East LLC", "AE", {"LB-CTR": 1}, 26, False),
    ("Talentbrücke GmbH", "DE", {"LB-REC": 1}, 12, False),
    ("Cloudmere Software Inc", "US", {"IT-SW": 1}, 24, True),
    ("Datavane GmbH", "DE", {"IT-SW": 0.7, "IT-SVC": 0.3}, 30, True),
    ("Byteworks Systemhaus GmbH", "DE", {"IT-HW": 0.8, "IT-SVC": 0.2}, 44, False),
    ("Kelvinet Telecom GmbH", "DE", {"IT-TEL": 1}, 24, True),
    ("Orbital IT Services Ltd", "GB", {"IT-SVC": 0.6, "IT-SW": 0.4}, 28, False),
    ("Glanzwerk Gebäudereinigung GmbH", "DE", {"FM-CLN": 1}, 24, True),
    ("Wachdienst Nord GmbH", "DE", {"FM-SEC": 1}, 24, True),
    ("Heiztechnik Schröder GmbH", "DE", {"FM-MNT": 1}, 40, False),
    ("Granitehall Facilities Ltd", "GB", {"FM-MNT": 0.6, "FM-CLN": 0.4}, 30, False),
    ("Energiewerk Allertal GmbH", "DE", {"FM-UTL": 1}, 24, True),
    ("Skyline Travel Management GmbH", "DE", {"TR-AIR": 0.55, "TR-HTL": 0.35, "TR-GND": 0.1}, 160, True),
    ("Hotel am Schlosspark", "DE", {"TR-HTL": 1}, 40, False),
    ("Taxizentrale Allertal", "DE", {"TR-GND": 1}, 50, False),
    ("Meridian Logistics GmbH", "DE", {"LG-FWD": 1}, 70, True),
    ("Atlas Forwarding Ltd", "GB", {"LG-FWD": 1}, 40, False),
    ("Swiftpost Kurier GmbH", "DE", {"LG-COU": 1}, 60, False),
    ("Werkzeug Krämer KG", "DE", {"MRO-TLS": 1}, 70, False),
    ("Gulf Industrial Supply LLC", "AE", {"MRO-TLS": 0.7, "MRO-PPE": 0.3}, 40, False),
    ("SafeGuard PSA GmbH", "DE", {"MRO-PPE": 1}, 36, True),
    ("Leuchtturm Events GmbH", "DE", {"MK-EVT": 1}, 10, False),
    ("Papyrus Bürobedarf GmbH", "DE", {"OF-SUP": 1}, 50, False),
    ("Kompass Training GmbH", "DE", {"HR-TRN": 1}, 20, False),
]

CURRENCY = {"DE": "EUR", "GB": "GBP", "US": "USD", "NO": "NOK", "AE": "AED"}
FX = {"EUR": 1.0, "GBP": 1.17, "USD": 0.90, "NOK": 0.085, "AED": 0.245}

# Typical line value in EUR (median of a lognormal).
MEDIAN = {
    "PS-CON": 18000,
    "PS-LEG": 9000,
    "PS-AUD": 25000,
    "PS-ENG": 14000,
    "LB-CTR": 11000,
    "LB-REC": 12000,
    "IT-SW": 6000,
    "IT-HW": 2500,
    "IT-TEL": 3000,
    "IT-SVC": 9000,
    "FM-CLN": 4000,
    "FM-SEC": 7000,
    "FM-MNT": 2000,
    "FM-UTL": 12000,
    "TR-AIR": 900,
    "TR-HTL": 400,
    "TR-GND": 120,
    "LG-FWD": 3500,
    "LG-COU": 150,
    "MRO-TLS": 700,
    "MRO-PPE": 600,
    "MK-EVT": 8000,
    "OF-SUP": 250,
    "HR-TRN": 2200,
}

TEMPLATES = {
    "PS-CON": [
        "Consulting services {m}",
        "Strategy workshop {city}",
        "Beratungsleistung {m}",
        "Process review phase {n}",
    ],
    "PS-LEG": [
        "Legal advice contract review",
        "Rechtsberatung Arbeitsrecht",
        "Trademark filing",
        "Litigation support {m}",
    ],
    "PS-AUD": ["Audit FY{y}", "Jahresabschluss {y} Prüfung", "Tax advice transfer pricing", "Steuerberatung {m}"],
    "PS-ENG": [
        "Engineering support {proj}",
        "Design review {proj}",
        "FEA stress analysis",
        "Konstruktionsleistung {proj}",
        "Drawing update {proj}",
    ],
    "LB-CTR": [
        "Contractor day rate {m}",
        "Zeitarbeit {m} Lager",
        "Arbeitnehmerüberlassung {m}",
        "Timesheet week {n}",
        "Temporary staff {city}",
    ],
    "LB-REC": ["Placement fee engineer", "Personalvermittlung Projektleiter", "Executive search fee"],
    "IT-SW": ["Software licences {y}", "SaaS subscription {m}", "Lizenzverlängerung {y}", "User seats annual"],
    "IT-HW": ["Laptops x{n}", "Monitor 27 inch x{n}", "Docking station", "Notebook Ersatz", "Server memory upgrade"],
    "IT-TEL": ["Mobilfunk {m}", "Mobile data plan {m}", "SIM cards", "Internet line {city}"],
    "IT-SVC": ["IT support {m}", "Managed services {m}", "Service desk {m}", "Implementation phase {n}"],
    "FM-CLN": ["Gebäudereinigung {m}", "Office cleaning {m}", "Window cleaning", "Reinigung Werkhalle {m}"],
    "FM-SEC": ["Wachdienst {m}", "Security guard services {m}", "Objektschutz Werk {m}"],
    "FM-MNT": ["Wartung Klimaanlage", "HVAC repair", "Heizung Reparatur", "Aufzug Wartung {m}", "Plumbing repair"],
    "FM-UTL": ["Strom {m}", "Erdgas {m}", "Electricity {m}", "Wasser {m}"],
    "TR-AIR": ["Flight {city} return", "Flug {city}", "Airfare {city}", "Air ticket change fee"],
    "TR-HTL": ["Hotel {city} {n} nights", "Übernachtung {city}", "Accommodation {city}"],
    "TR-GND": ["Taxi {city}", "Car rental {city}", "Mietwagen {city}", "Train ticket {city}"],
    "LG-FWD": ["Air freight {city}", "Sea freight container", "Spedition {city}", "Customs clearance {city}"],
    "LG-COU": ["Courier {city}", "Express delivery", "Kurierfahrt {city}", "Parcel service {m}"],
    "MRO-TLS": ["Werkzeug Satz", "Drill bits assorted", "Fasteners M12", "Consumables workshop", "Schrauben Sortiment"],
    "MRO-PPE": ["PPE gloves", "Safety glasses x{n}", "Sicherheitsschuhe", "Coveralls FR x{n}", "Helmets"],
    "MK-EVT": ["Trade show booth {city}", "Messe {city} Standbau", "Customer event {city}", "Brochure design"],
    "OF-SUP": ["Office supplies", "Toner cartridges", "Papier A4", "Büromaterial", "Printing business cards"],
    "HR-TRN": ["Training course {n} days", "Schulung Excel", "Seminar leadership", "Certification fee"],
}
VAGUE = [
    "Services as per PO",
    "Leistungen gem. Bestellung",
    "Invoice {n}",
    "Monthly fee {m}",
    "Pauschale {m}",
    "Rahmenvertrag Abruf {n}",
    "Abschlagsrechnung {n}",
    "As agreed",
    "Q{q} charges",
]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CITIES = ["Aberdeen", "Houston", "Dubai", "Stavanger", "Hannover", "Singapore", "Celle", "London"]

SURNAMES = [
    "Schulz",
    "Becker",
    "Hoffmann",
    "Koch",
    "Richter",
    "Wolf",
    "Neumann",
    "Schwarz",
    "Zimmermann",
    "Braun",
    "Krüger",
    "Lange",
    "Werner",
    "Krause",
    "Lehmann",
    "Köhler",
    "Maier",
    "Walter",
    "Fuchs",
    "Peters",
    "Scholz",
    "Möller",
    "Jung",
    "Hahn",
    "Vogel",
    "Friedrich",
    "Seidel",
    "Engel",
    "Roth",
    "Busch",
]
TRADES = {
    "Elektro": "FM-MNT",
    "Haustechnik": "FM-MNT",
    "Druck": "OF-SUP",
    "Werkzeuge": "MRO-TLS",
    "Transporte": "LG-FWD",
    "Reinigung": "FM-CLN",
    "Schulungen": "HR-TRN",
    "IT-Service": "IT-SVC",
    "Beratung": "PS-CON",
    "Events": "MK-EVT",
}
TAIL_TEMPLATES = {
    "FM-MNT": ["Reparatur Elektrik", "Wartung Anlage", "Heizung Service"],
    "OF-SUP": ["Druck Flyer", "Visitenkarten Druck", "Büromaterial"],
    "MRO-TLS": ["Werkzeug", "Schrauben", "Consumables"],
    "LG-FWD": ["Transport {city}", "Fracht {city}", "Spedition"],
    "FM-CLN": ["Reinigung Büro", "Sonderreinigung"],
    "HR-TRN": ["Schulung", "Seminar"],
    "IT-SVC": ["IT support", "Service desk call-out"],
    "PS-CON": ["Beratung", "Workshop"],
    "MK-EVT": ["Event catering", "Messe Material"],
}

START, END = date(2025, 1, 1), date(2026, 9, 30)


def fill(template: str) -> str:
    return template.format(
        m=MONTHS[rng.integers(12)],
        y=rng.choice([2025, 2026]),
        n=int(rng.integers(2, 40)),
        city=CITIES[rng.integers(len(CITIES))],
        proj=f"P-{rng.integers(1000, 9999)}",
        q=int(rng.integers(1, 5)),
    )


def spellings(name: str) -> list[str]:
    """The ways one supplier shows up across invoices and ERP master records."""
    legal = {"GmbH", "Ltd", "Inc", "KG", "LLP", "LLC", "AS", "PartG", "mbB", "&", "Co.", "GmbH,"}
    words = name.split()
    core = " ".join(w for w in words if w not in legal)
    options = {
        name.upper(),
        core,
        name.replace("ü", "ue").replace("ä", "ae").replace("ö", "oe").replace("ø", "o"),
        name.replace("ü", "u").replace("ä", "a").replace("ö", "o").replace("ø", "o"),
        name.replace("Engineering", "Engg.")
        .replace("Services", "Svcs")
        .replace("Personaldienstleistungen", "Personal"),
        f"{core} ({CITIES[rng.integers(len(CITIES))]})",
        name.replace("GmbH & Co. KG", "GmbH u. Co. KG").replace(" & ", " und "),
    }
    # One typo: swap two letters inside the longest word.
    longest = max(words, key=len)
    if len(longest) > 6:
        i = int(rng.integers(1, len(longest) - 3))
        typo = longest[:i] + longest[i + 1] + longest[i] + longest[i + 2 :]
        options.add(name.replace(longest, typo))
    options.discard(name)
    extra = list(rng.choice(sorted(options), size=min(len(options), int(rng.integers(1, 5))), replace=False))
    return [name, *extra]


def random_date() -> date:
    return START + timedelta(days=int(rng.integers(0, (END - START).days)))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows, truth = [], []
    po = 4_500_000

    def add(supplier_id: str, display: str, country: str, category: str, description: str, contract: str | None):
        nonlocal po
        po += int(rng.integers(1, 7))
        eur = float(rng.lognormal(np.log(MEDIAN[category]), 0.6))
        currency = CURRENCY[country]
        amount = round(eur / FX[currency], 2)
        rows.append(
            {
                "po_number": str(po),
                "po_line": 10,
                "posting_date": random_date().isoformat(),
                "supplier_name": display,
                "supplier_country": country,
                "description": description,
                "amount": amount,
                "currency": currency,
                "cost_center": f"CC{rng.integers(4100, 4180)}",
                "contract_id": contract or "",
            }
        )
        truth.append({"po_number": str(po), "po_line": 10, "true_supplier": supplier_id, "true_category": category})

    for idx, (name, country, cats, per_year, has_contract) in enumerate(MAJOR):
        sid = f"T{idx:03d}"
        names = spellings(name)
        weights = np.array([0.7] + [0.3 / (len(names) - 1)] * (len(names) - 1)) if len(names) > 1 else np.array([1.0])
        n_lines = int(per_year * 1.75 * rng.uniform(0.8, 1.2))
        contract = f"CW-{2400 + idx}" if has_contract else None
        cat_ids, cat_w = list(cats), np.array(list(cats.values()))
        for _ in range(n_lines):
            category = str(rng.choice(cat_ids, p=cat_w / cat_w.sum()))
            vague = rng.random() < 0.18
            desc = fill(str(rng.choice(VAGUE if vague else TEMPLATES[category])))
            line_contract = contract if contract and rng.random() < 0.9 else None
            add(sid, str(rng.choice(names, p=weights)), country, category, desc, line_contract)

    # Long tail: many small local suppliers, a few invoices each.
    seen = set()
    tail_id = 0
    while tail_id < 110:
        surname = SURNAMES[rng.integers(len(SURNAMES))]
        trade = list(TRADES)[rng.integers(len(TRADES))]
        if (surname, trade) in seen:
            continue
        seen.add((surname, trade))
        category = TRADES[trade]
        name = f"{surname} {trade}" + (" GmbH" if rng.random() < 0.5 else "")
        sid = f"L{tail_id:03d}"
        for _ in range(int(rng.integers(1, 6))):
            desc = fill(str(rng.choice(TAIL_TEMPLATES[category] + ["Rechnung {n}"])))
            add(sid, name, "DE", category, desc, None)
            rows[-1]["amount"] = round(float(rng.lognormal(np.log(600), 0.9)), 2)
        tail_id += 1

    order = np.argsort([r["posting_date"] for r in rows], kind="stable")
    with open(OUT / "spend_lines.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows[i] for i in order)
    with open(OUT / "ground_truth.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(truth[0]))
        writer.writeheader()
        writer.writerows(truth)
    with open(OUT / "fx_rates.csv", "w", newline="", encoding="utf-8") as f:
        f.write("currency,eur_per_unit\n")
        f.writelines(f"{c},{r}\n" for c, r in FX.items())

    # Corporate structure: subsidiaries under other names, an acquisition, and a 50/50 joint venture.
    # Brightwater's US subsidiary is deliberately left out, for the review suggestions to find.
    groups = [
        ("Kessler & Partner Beratung GmbH", "DE", "Kessler Northbridge Group", 1),
        ("Northbridge Consulting Inc", "US", "Kessler Northbridge Group", 1),
        ("KP Advisory UK Ltd", "GB", "Kessler Northbridge Group", 1),
        ("Granite Staffing Ltd", "GB", "Granite Workforce Group", 1),
        ("Granite Staffing Middle East LLC", "AE", "Granite Workforce Group", 1),
        ("Nordlys Bemanning AS", "NO", "Granite Workforce Group", 1),
        ("Tidewell Engineering Ltd", "GB", "Tidewell Group", 1),
        ("Tidewell Stavik Engineering JV AS", "NO", "Tidewell Group", 0.5),
        ("Tidewell Stavik Engineering JV AS", "NO", "Stavik Group", 0.5),
        ("Stavik Ingeniørtjenester AS", "NO", "Stavik Group", 1),
    ]
    with open(OUT / "supplier_groups.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["supplier", "country", "group", "share"])
        writer.writerows(groups)

    names = {r["supplier_name"] for r in rows}
    print(f"{len(rows)} lines, {len(names)} supplier spellings for {len(MAJOR) + tail_id} real suppliers")


if __name__ == "__main__":
    main()
