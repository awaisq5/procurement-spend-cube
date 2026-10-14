"""Text normalisation shared by supplier matching and classification."""

from __future__ import annotations

import re
import unicodedata

_FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "æ": "ae", "ø": "o", "å": "a"})

# Legal forms carry no identity; they vary across invoices of the same supplier.
LEGAL_FORMS = {
    "gmbh",
    "mbh",
    "co",
    "kg",
    "ag",
    "se",
    "ohg",
    "ug",
    "ev",
    "partg",
    "mbb",
    "ltd",
    "limited",
    "inc",
    "incorporated",
    "llc",
    "llp",
    "plc",
    "corp",
    "corporation",
    "as",
    "asa",
    "ab",
    "bv",
    "nv",
    "sa",
    "sarl",
    "srl",
    "spa",
    "pty",
    "lp",
}
ABBREVIATIONS = {
    "engg": "engineering",
    "svcs": "services",
    "svc": "services",
    "intl": "international",
    "mgmt": "management",
    "grp": "group",
}


def fold(text: str) -> str:
    """Lower-case, umlauts written out (ü to ue), other accents removed."""
    text = text.lower().translate(_FOLD)
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def normalise_description(text: str) -> str:
    text = fold(text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def supplier_key(name: str) -> str:
    """Matching key for a supplier name: no legal form, no punctuation, no location suffix.

    ``"Müller Engineering GmbH & Co. KG (Celle)"`` and ``"MUELLER ENGG"`` both
    become ``"mueller engineering"``.
    """
    text = fold(name)
    text = re.sub(r"\(.*?\)", " ", text)  # "(Celle)", "(UK)"
    text = text.replace("&", " ").replace("+", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = [ABBREVIATIONS.get(t, t) for t in text.split()]
    tokens = [t for t in tokens if t not in LEGAL_FORMS and t not in {"und", "and", "the"}]
    return " ".join(tokens)
