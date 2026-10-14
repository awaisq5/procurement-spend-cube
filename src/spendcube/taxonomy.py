"""Indirect spend taxonomy with keyword rules in English and German.

Keywords are matched against a normalised description (lower case, umlauts
folded, punctuation removed) on word boundaries; a trailing ``*`` matches any
word ending, which handles German compounds like ``Reinigungsleistung``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    id: str
    group: str
    name: str
    keywords: tuple[str, ...]


CATEGORIES: tuple[Category, ...] = (
    # Professional services
    Category(
        "PS-CON",
        "Professional services",
        "Consulting",
        ("consulting", "advisory", "strategy", "workshop", "beratung*", "berater*", "process review"),
    ),
    Category(
        "PS-LEG",
        "Professional services",
        "Legal",
        ("legal", "counsel", "litigation", "contract review", "rechtsberatung", "anwalt*", "notar*", "trademark"),
    ),
    Category(
        "PS-AUD",
        "Professional services",
        "Audit and tax",
        ("audit", "tax advice", "transfer pricing", "wirtschaftspruefung", "steuerberatung", "jahresabschluss"),
    ),
    Category(
        "PS-ENG",
        "Professional services",
        "Engineering services",
        ("engineering", "design review", "fea", "drawing*", "konstruktion*", "ingenieur*", "stress analysis"),
    ),
    # Labour
    Category(
        "LB-CTR",
        "Labour",
        "Contingent labour",
        (
            "contractor",
            "temporary staff",
            "temp staff",
            "zeitarbeit*",
            "arbeitnehmerueberlassung",
            "leiharbeit*",
            "day rate",
            "timesheet",
        ),
    ),
    Category(
        "LB-REC",
        "Labour",
        "Recruitment",
        ("recruitment", "headhunt*", "placement fee", "personalvermittlung", "executive search"),
    ),
    # IT
    Category(
        "IT-SW",
        "IT",
        "Software and SaaS",
        ("licen*", "subscription", "saas", "software", "lizenz*", "cloud storage", "user seats"),
    ),
    Category(
        "IT-HW",
        "IT",
        "Hardware",
        ("laptop*", "notebook*", "monitor*", "docking station", "server", "hardware", "tablet*"),
    ),
    Category(
        "IT-TEL",
        "IT",
        "Telecom",
        ("mobile", "sim card*", "telefon*", "mobilfunk*", "internet line", "mpls", "data plan"),
    ),
    Category(
        "IT-SVC",
        "IT",
        "IT services",
        ("it support", "managed service*", "helpdesk", "service desk", "implementation", "it-dienstleistung*"),
    ),
    # Facilities
    Category(
        "FM-CLN",
        "Facilities",
        "Cleaning",
        ("cleaning", "reinigung*", "gebaeudereinigung", "window cleaning", "janitorial"),
    ),
    Category(
        "FM-SEC",
        "Facilities",
        "Security services",
        ("security guard*", "wachdienst*", "objektschutz", "access control", "pfoertner*"),
    ),
    Category(
        "FM-MNT",
        "Facilities",
        "Building maintenance",
        (
            "hvac",
            "klimaanlage*",
            "heizung*",
            "wartung*",
            "elevator",
            "aufzug*",
            "facility maintenance",
            "plumbing",
            "sanitaer*",
        ),
    ),
    Category(
        "FM-UTL",
        "Facilities",
        "Utilities",
        ("electricity", "strom*", "gas supply", "erdgas", "wasser", "water supply", "energy"),
    ),
    # Travel
    Category(
        "TR-AIR",
        "Travel",
        "Air travel",
        ("flight", "airfare", "flug*", "air ticket", "economy class", "business class"),
    ),
    Category("TR-HTL", "Travel", "Hotels", ("hotel", "accommodation", "uebernachtung*", "room night*", "lodging")),
    Category(
        "TR-GND",
        "Travel",
        "Ground transport",
        ("taxi", "car rental", "mietwagen", "rental car", "train ticket", "bahn*", "shuttle"),
    ),
    # Logistics
    Category(
        "LG-FWD",
        "Logistics",
        "Freight forwarding",
        ("freight", "air freight", "sea freight", "fracht*", "spedition*", "customs clearance", "verzollung"),
    ),
    Category("LG-COU", "Logistics", "Courier", ("courier", "kurier*", "express delivery", "parcel", "paket*")),
    # MRO
    Category(
        "MRO-TLS",
        "MRO",
        "Tools and consumables",
        ("tool*", "werkzeug*", "drill bit*", "consumable*", "fastener*", "schrauben", "abrasive*"),
    ),
    Category(
        "MRO-PPE",
        "MRO",
        "Safety equipment",
        ("ppe", "psa", "safety glasses", "gloves", "handschuh*", "helmet*", "coverall*", "sicherheitsschuh*"),
    ),
    # Other indirect
    Category(
        "MK-EVT",
        "Marketing",
        "Marketing and events",
        ("trade show", "messe*", "event", "booth", "marketing", "brochure", "werbe*"),
    ),
    Category(
        "OF-SUP",
        "Office",
        "Office supplies and print",
        ("office supplies", "buerobedarf", "toner", "paper", "papier", "printing", "druck*", "stationery"),
    ),
    Category(
        "HR-TRN", "HR", "Training", ("training", "schulung*", "course", "seminar*", "certification", "weiterbildung")
    ),
)

BY_ID = {c.id: c for c in CATEGORIES}
UNCLASSIFIED = "UNCLASSIFIED"


def _compile(keyword: str) -> re.Pattern[str]:
    if keyword.endswith("*"):
        return re.compile(rf"\b{re.escape(keyword[:-1])}\w*")
    return re.compile(rf"\b{re.escape(keyword)}\b")


PATTERNS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = tuple(
    (c.id, tuple(_compile(k) for k in c.keywords)) for c in CATEGORIES
)
