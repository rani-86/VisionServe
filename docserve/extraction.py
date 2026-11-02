"""Rule-based key-value extraction from OCR rows, keyed by the predicted document type.

Each field has label wordings (all variants seen on documents of that type) and a
value pattern. Labels match exactly first, then fuzzily (edit distance) so OCR
typos such as "Baiance" or "Perrmanent" still anchor the value. The value is taken
from the same row after the label, or from the next row when the label sits alone. Values get position-aware OCR
fixes (O->0 in numeric fields, 0->O in alphabetic slots of a PAN, etc.).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

# Grouped (12,345.67 / 1,23,456.70) or plain (1234.50). OCR often inserts a space after a separator.
AMOUNT = r"(?:\d{1,3}(?:[,.]\s?\d{2,3})+|\d+)[.,]\s?\d{2}(?!\d)"
DATE = r"\d{1,2}\s?[/.-]\s?\d{1,2}\s?[/.-]\s?\d{2,4}"
MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}"
FUZZY_MIN_LEN = 8  # labels shorter than this (PAN, DOB, Date, TOTAL) must match exactly
FUZZY_MIN_SCORE = 88


@dataclass(frozen=True)
class FieldSpec:
    labels: tuple[str, ...]  # literal label text; a leading "^" means it must start the row
    value: str
    kind: str = "text"  # amount | date | digits | pan | ifsc | code | text


SPECS: dict[str, dict[str, FieldSpec]] = {
    "invoice": {
        "invoice_number": FieldSpec(("invoice no", "invoice number", "invoice #", "bill no"),
                                    r"INV\s?-?\s?[\dOIl]{5}", "code"),
        "invoice_date": FieldSpec(("invoice date", "date of issue", "dated"), DATE, "date"),
        "total_amount": FieldSpec(("total amount", "grand total", "amount due", "^total"), AMOUNT, "amount"),
    },
    "bank_statement": {
        "account_number": FieldSpec(("account number", "account no", "a/c no"), r"[\dOIl]{9,14}", "digits"),
        "ifsc": FieldSpec(("ifsc code", "ifsc"), r"[A-Z]{4}\s?[0O]\s?[\dOIl]{6}", "ifsc"),
        "closing_balance": FieldSpec(("closing balance", "balance as on date", "closing bal"), AMOUNT, "amount"),
    },
    "payslip": {
        "employee_id": FieldSpec(("employee id", "employee no", "emp code"), r"EMP\s?[\dOIl]{4}", "code"),
        "pay_period": FieldSpec(("pay period", "salary for the month of", "^month"), MONTH, "text"),
        "net_pay": FieldSpec(("net pay", "net salary", "take home pay"), AMOUNT, "amount"),
    },
    "id_card": {
        "id_number": FieldSpec(("permanent account number", "^pan", "id no"),
                               r"[A-Z0-9]{5}\s?[\dOIlSB]{4}\s?[A-Z0-9]", "pan"),
        "name": FieldSpec(("^name", "holder name"), r"[A-Z][A-Z .]{2,40}", "text"),
        "date_of_birth": FieldSpec(("date of birth", "dob"), DATE, "date"),
    },
    "receipt": {
        "receipt_number": FieldSpec(("receipt no", "bill #", "txn no"), r"[\dOIl]{6}", "digits"),
        "date": FieldSpec(("^date", "txn date"), DATE, "date"),
        "total_amount": FieldSpec(("total amount", "grand total", "amount due", "^total"), AMOUNT, "amount"),
    },
}


def _label_regex(label: str) -> re.Pattern:
    anchored = label.startswith("^")
    body = r"\s*".join(re.escape(w) for w in label.lstrip("^").split())
    return re.compile(("^" if anchored else "") + body, re.IGNORECASE)


_LABEL_RE = {lbl: _label_regex(lbl) for spec in (f for d in SPECS.values() for f in d.values()) for lbl in spec.labels}

_TO_DIGIT = str.maketrans({"O": "0", "o": "0", "D": "0", "I": "1", "l": "1", "|": "1", "S": "5", "B": "8", "Z": "2"})
_TO_ALPHA = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"})


def normalize_value(kind: str, raw: str) -> str:
    v = raw.strip(" :.-")
    if kind == "amount":
        v = re.sub(r"\s", "", v)
        # Last separator is the decimal point; everything before is thousands grouping.
        m = re.match(r"^(.*?)[.,](\d{2})$", v)
        if m:
            whole = re.sub(r"[^\d]", "", m.group(1))
            return f"{int(whole):,}.{m.group(2)}" if whole else v
        return v
    if kind == "date":
        parts = re.split(r"\s*[/.-]\s*", v)
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            return f"{int(parts[0]):02d}/{int(parts[1]):02d}/{parts[2]}"
        return v
    if kind == "digits":
        return re.sub(r"\s", "", v).translate(_TO_DIGIT)
    if kind == "code":
        v = re.sub(r"\s", "", v)
        m = re.match(r"^([A-Za-z]+)-?(.*)$", v)
        if m:
            sep = "-" if m.group(1).upper() == "INV" else ""
            return m.group(1).upper() + sep + m.group(2).translate(_TO_DIGIT)
        return v
    if kind == "pan":  # AAAAA9999A
        v = re.sub(r"\s", "", v).upper()
        if len(v) == 10:
            return v[:5].translate(_TO_ALPHA) + v[5:9].translate(_TO_DIGIT) + v[9].translate(_TO_ALPHA)
        return v
    if kind == "ifsc":  # AAAA0999999
        v = re.sub(r"\s", "", v).upper()
        if len(v) == 11:
            return v[:4].translate(_TO_ALPHA) + "0" + v[5:].translate(_TO_DIGIT)
        return v
    return re.sub(r"\s+", " ", v)


def _label_end(row: str, label: str, fuzzy: bool) -> int | None:
    """Index just past the label in `row`, or None. Fuzzy mode tolerates OCR typos in long labels."""
    m = _LABEL_RE[label].search(row)
    if m:
        return m.end()
    text = label.lstrip("^")
    if not fuzzy or len(text) < FUZZY_MIN_LEN:
        return None
    al = fuzz.partial_ratio_alignment(text, row.lower(), score_cutoff=FUZZY_MIN_SCORE)
    if al is None or (label.startswith("^") and al.dest_start > 1):
        return None
    end = al.dest_end
    while end < len(row) and row[end].isalpha():  # the alignment window can stop mid-word
        end += 1
    return end


def _find_after_label(rows: list[str], spec: FieldSpec) -> str | None:
    for fuzzy in (False, True):  # exact label matches anywhere on the page win over fuzzy ones
        for i, row in enumerate(rows):
            for label in spec.labels:
                end = _label_end(row, label, fuzzy)
                if end is None:
                    continue
                tail = row[end:].lstrip(" :#-.,;")
                vm = re.search(spec.value, tail)
                if vm:
                    return vm.group(0)
                if i + 1 < len(rows) and not tail.strip():  # value on the next line (e.g. PAN under its label)
                    vm = re.search(spec.value, rows[i + 1])
                    if vm:
                        return vm.group(0)
    return None


def extract_fields(doc_type: str, rows: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name, spec in SPECS.get(doc_type, {}).items():
        raw = _find_after_label(rows, spec)
        out[name] = normalize_value(spec.kind, raw) if raw else None
    return out


def normalize_truth(doc_type: str, name: str, value: str) -> str:
    """Ground truth goes through the same normaliser so comparisons are like for like."""
    return normalize_value(SPECS[doc_type][name].kind, value)
