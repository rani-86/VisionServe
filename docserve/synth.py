"""Synthetic document generator with ground truth for training and OCR evaluation.

Renders five lending-style document types (invoice, bank statement, payslip,
PAN-style ID card, retail receipt) with randomised layouts, label wording and
fonts, then degrades them like a flatbed scan or a phone photo. Each sample
carries the exact text lines and key-value fields that were drawn, so OCR CER/WER
and field accuracy can be measured.

Entity pools (people, companies, merchants) are drawn from split-specific
syllable seeds, so the test split never shares a name with training.
"""
from __future__ import annotations

import io
import random
import string
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DOC_TYPES = ["invoice", "bank_statement", "payslip", "id_card", "receipt"]

FIELDS: dict[str, list[str]] = {
    "invoice": ["invoice_number", "invoice_date", "total_amount"],
    "bank_statement": ["account_number", "ifsc", "closing_balance"],
    "payslip": ["employee_id", "pay_period", "net_pay"],
    "id_card": ["id_number", "name", "date_of_birth"],
    "receipt": ["receipt_number", "date", "total_amount"],
}

# Several wordings per label; extraction has to handle all of them.
LABELS: dict[str, list[str]] = {
    "invoice_number": ["Invoice No:", "Invoice #:", "Invoice Number:", "Bill No:"],
    "invoice_date": ["Invoice Date:", "Date of Issue:", "Dated:"],
    "total_amount": ["Total Amount:", "Grand Total:", "Amount Due:", "TOTAL:"],
    "account_number": ["Account Number:", "A/C No:", "Account No:"],
    "ifsc": ["IFSC:", "IFSC Code:"],
    "closing_balance": ["Closing Balance:", "Balance as on date:", "Closing Bal:"],
    "employee_id": ["Employee ID:", "Emp Code:", "Employee No:"],
    "pay_period": ["Pay Period:", "Salary for the month of:", "Month:"],
    "net_pay": ["Net Pay:", "Net Salary:", "Take Home Pay:"],
    "id_number": ["Permanent Account Number:", "PAN:", "ID No:"],
    "name": ["Name:", "Holder Name:"],
    "date_of_birth": ["Date of Birth:", "DOB:"],
    "receipt_number": ["Receipt No:", "Bill #:", "Txn No:"],
    "date": ["Date:", "Txn Date:"],
}

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
ITEMS = ["Steel rods", "Cement bags", "Office chairs", "Laptop stand", "Printer toner", "LED panels",
         "Copper wire", "Safety gloves", "Paint buckets", "Network cable", "Desk lamps", "Water pumps"]
GROCERY = ["Milk 1L", "Bread", "Eggs x12", "Rice 5kg", "Tea 250g", "Sugar 1kg", "Soap", "Biscuits",
           "Apples", "Cooking oil", "Detergent", "Coffee"]

FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"),
             Path("/usr/share/fonts/truetype/liberation")]
FONT_PAIRS = [("arial.ttf", "arialbd.ttf"), ("verdana.ttf", "verdanab.ttf"), ("times.ttf", "timesbd.ttf"),
              ("calibri.ttf", "calibrib.ttf"), ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
              ("DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf"),
              ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf")]

SPLIT_SEEDS = {"train": 101, "val": 202, "test": 303, "holdout": 404}  # holdout: final OCR report only


def available_fonts() -> list[tuple[Path, Path]]:
    pairs = [(d / a, d / b) for a, b in FONT_PAIRS for d in FONT_DIRS if (d / a).exists() and (d / b).exists()]
    if not pairs:
        raise RuntimeError("No TrueType fonts found; install fonts-dejavu or run on Windows")
    return pairs


@dataclass
class TextItem:
    text: str
    bbox: tuple[int, int, int, int]


@dataclass
class Sample:
    doc_type: str
    fields: dict[str, str]
    items: list[TextItem]
    image: np.ndarray  # BGR uint8, degraded
    clean: np.ndarray  # BGR uint8, before degradation
    capture: str  # scan | photo

    @property
    def lines(self) -> list[str]:
        return group_rows([(it.text, it.bbox) for it in self.items])


def group_rows(boxes: list[tuple[str, tuple[int, int, int, int]]]) -> list[str]:
    """Group text boxes into reading-order rows by vertical overlap, left to right inside a row."""
    rows: list[list[tuple[str, tuple[int, int, int, int]]]] = []
    for text, b in sorted(boxes, key=lambda t: (t[1][1] + t[1][3]) / 2):
        cy, h = (b[1] + b[3]) / 2, max(1, b[3] - b[1])
        if rows:
            last = rows[-1]
            ly = sum((x[1][1] + x[1][3]) / 2 for x in last) / len(last)
            lh = sum(max(1, x[1][3] - x[1][1]) for x in last) / len(last)
            if abs(cy - ly) < 0.5 * min(h, lh):
                last.append((text, b))
                continue
        rows.append([(text, b)])
    return [" ".join(t for t, _ in sorted(r, key=lambda x: x[1][0])) for r in rows]


class EntityPool:
    """Pronounceable fake names. Each split owns a disjoint slice of the syllable inventory."""

    def __init__(self, split: str):
        cons, vow = "bcdfghjklmnprstvwy", "aeiou"
        syll = sorted({a + b + c for a in cons for b in vow for c in ["", "n", "r", "l", "s"]})
        random.Random(7919).shuffle(syll)
        i = list(SPLIT_SEEDS).index(split)
        self.syllables = syll[i * 80:(i + 1) * 80]
        self.rng = random.Random(SPLIT_SEEDS[split])

    def word(self, n: int | None = None) -> str:
        n = n or self.rng.randint(2, 3)
        return "".join(self.rng.choice(self.syllables) for _ in range(n)).capitalize()

    def person(self) -> str:
        return f"{self.word()} {self.word()}"

    def company(self) -> str:
        return f"{self.word()} {self.rng.choice(['Traders', 'Industries', 'Pvt Ltd', 'Enterprises', 'Logistics', 'Supplies'])}"


class Canvas:
    def __init__(self, w: int, h: int, fonts: tuple[Path, Path], size: int, ox: int = 0, oy: int = 0,
                 img: Image.Image | None = None):
        self.img = img or Image.new("RGB", (w, h), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.img)
        self.fonts, self.size = fonts, size
        self.ox, self.oy = ox, oy  # offset of this canvas inside the page (for card / receipt layouts)
        self.items: list[TextItem] = []

    def font(self, size: int | None = None, bold: bool = False) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(str(self.fonts[1 if bold else 0]), size or self.size)

    def text(self, x: int, y: int, s: str, size: int | None = None, bold: bool = False, anchor: str = "la",
             fill=(20, 20, 20)) -> tuple[int, int, int, int]:
        f = self.font(size, bold)
        self.draw.text((x, y), s, font=f, fill=fill, anchor=anchor)
        b = self.draw.textbbox((x, y), s, font=f, anchor=anchor)
        self.items.append(TextItem(s, (b[0] + self.ox, b[1] + self.oy, b[2] + self.ox, b[3] + self.oy)))
        return b

    def hline(self, y: int, x0: int, x1: int, width: int = 2) -> None:
        self.draw.line([(x0, y), (x1, y)], fill=(40, 40, 40), width=width)

    def rect(self, box, width: int = 2, fill=None) -> None:
        self.draw.rectangle(box, outline=(40, 40, 40), width=width, fill=fill)


class DocumentGenerator:
    PAGE = (1000, 1414)  # A4 at ~120 dpi

    def __init__(self, split: str = "train", seed: int | None = None):
        self.split = split
        self.rng = random.Random(seed if seed is not None else SPLIT_SEEDS[split] + 1)
        self.np_rng = np.random.default_rng(self.rng.randint(0, 2**31))
        self.pool = EntityPool(split)
        self.pool.rng = self.rng
        self.fonts = available_fonts()

    # ---- value helpers -------------------------------------------------
    def money(self, lo: float, hi: float) -> str:
        return f"{self.rng.uniform(lo, hi):,.2f}"

    def date(self, y0: int = 2023, y1: int = 2026) -> str:
        return f"{self.rng.randint(1, 28):02d}/{self.rng.randint(1, 12):02d}/{self.rng.randint(y0, y1)}"

    def digits(self, n: int) -> str:
        return str(self.rng.randint(1, 9)) + "".join(self.rng.choice(string.digits) for _ in range(n - 1))

    def label(self, key: str) -> str:
        return self.rng.choice(LABELS[key])

    def page(self) -> Canvas:
        return Canvas(*self.PAGE, self.rng.choice(self.fonts), self.rng.randint(19, 23))

    # ---- document layouts ----------------------------------------------
    def invoice(self) -> tuple[Canvas, dict[str, str]]:
        c, m = self.page(), self.rng.randint(60, 90)
        W = self.PAGE[0]
        f = {"invoice_number": f"INV-{self.digits(5)}", "invoice_date": self.date(), "total_amount": ""}
        c.text(m, 60, self.pool.company(), size=34, bold=True)
        c.text(m, 110, f"GSTIN {self.digits(2)}{''.join(self.rng.choices(string.ascii_uppercase, k=5))}{self.digits(4)}")
        c.text(W - m, 60, "TAX INVOICE", size=30, bold=True, anchor="ra")
        y = 190
        for key in self.rng.sample(["invoice_number", "invoice_date"], 2):
            c.text(m, y, f"{self.label(key)} {f[key]}")
            y += 40
        c.text(m, y + 10, f"Bill To: {self.pool.company()}")
        y += 90
        cols = [m, int(W * 0.55), int(W * 0.7), W - m]
        c.hline(y, m, W - m)
        for x, h, a in zip(cols, ["Description", "Qty", "Rate", "Amount"], ["la", "la", "la", "ra"]):
            c.text(x, y + 10, h, bold=True, anchor=a)
        y += 50
        c.hline(y, m, W - m, 1)
        total = 0.0
        for item in self.rng.sample(ITEMS, self.rng.randint(3, 7)):
            q, r = self.rng.randint(1, 40), self.rng.uniform(50, 2500)
            total += q * r
            for x, v, a in zip(cols, [item, str(q), f"{r:,.2f}", f"{q * r:,.2f}"], ["la", "la", "la", "ra"]):
                c.text(x, y + 10, v, anchor=a)
            y += 42
        c.hline(y + 5, m, W - m)
        f["total_amount"] = f"{total * 1.18:,.2f}"
        c.text(cols[2], y + 30, f"GST 18%: {total * 0.18:,.2f}")
        c.text(cols[2] - 120, y + 75, f"{self.label('total_amount')} {f['total_amount']}", bold=True)
        c.text(m, y + 160, "Thank you for your business.", size=18)
        return c, f

    def bank_statement(self) -> tuple[Canvas, dict[str, str]]:
        c, m = self.page(), self.rng.randint(55, 85)
        W = self.PAGE[0]
        bank = f"{self.pool.word(2)} Bank"
        f = {"account_number": self.digits(self.rng.choice([11, 12])),
             "ifsc": "".join(self.rng.choices(string.ascii_uppercase, k=4)) + "0" + self.digits(6), "closing_balance": ""}
        c.text(m, 55, bank, size=36, bold=True)
        c.text(W - m, 65, "ACCOUNT STATEMENT", size=24, bold=True, anchor="ra")
        c.text(m, 140, f"Customer: {self.pool.person()}")
        c.text(m, 180, f"{self.label('account_number')} {f['account_number']}")
        c.text(int(W * 0.58), 180, f"{self.label('ifsc')} {f['ifsc']}")
        c.text(m, 220, f"Period: 01/{self.rng.randint(1, 9):02d}/2025 to 30/{self.rng.randint(10, 12):02d}/2025")
        y = 290
        cols = [m, int(W * 0.24), int(W * 0.62), int(W * 0.78), W - m]
        c.rect((m - 10, y - 8, W - m + 10, y + 40), fill=(225, 225, 225))
        for x, h, a in zip(cols, ["Date", "Narration", "Debit", "Credit", "Balance"], ["la", "la", "ra", "ra", "ra"]):
            c.text(x, y, h, bold=True, anchor=a)
        y += 55
        bal = self.rng.uniform(5000, 200000)
        for _ in range(self.rng.randint(8, 16)):
            amt = self.rng.uniform(100, 25000)
            debit = self.rng.random() < 0.6
            bal = bal - amt if debit and bal > amt else bal + amt
            narr = self.rng.choice(["UPI/", "NEFT/", "ATM WDL/", "IMPS/", "POS/"]) + self.pool.word(2).upper()
            row = [self.date(2025, 2025)[:5], narr, f"{amt:,.2f}" if debit else "", "" if debit else f"{amt:,.2f}",
                   f"{bal:,.2f}"]
            for x, v, a in zip(cols, row, ["la", "la", "ra", "ra", "ra"]):
                if v:
                    c.text(x, y, v, size=c.size - 2, anchor=a)
            y += 38
        c.hline(y + 5, m, W - m)
        f["closing_balance"] = f"{bal:,.2f}"
        c.text(m, y + 30, f"{self.label('closing_balance')} {f['closing_balance']}", bold=True)
        return c, f

    def payslip(self) -> tuple[Canvas, dict[str, str]]:
        c, m = self.page(), self.rng.randint(60, 90)
        W = self.PAGE[0]
        f = {"employee_id": f"EMP{self.digits(4)}", "pay_period": f"{self.rng.choice(MONTHS)} {self.rng.randint(2023, 2026)}",
             "net_pay": ""}
        c.text(W // 2, 60, self.pool.company(), size=32, bold=True, anchor="ma")
        c.text(W // 2, 110, "SALARY SLIP", size=26, bold=True, anchor="ma")
        y = 180
        c.text(m, y, f"Employee Name: {self.pool.person()}")
        c.text(int(W * 0.55), y, f"{self.label('employee_id')} {f['employee_id']}")
        c.text(m, y + 40, f"{self.label('pay_period')} {f['pay_period']}")
        c.text(int(W * 0.55), y + 40, f"Designation: {self.rng.choice(['Analyst', 'Engineer', 'Manager', 'Associate'])}")
        y += 120
        mid = W // 2
        c.rect((m - 10, y - 10, W - m + 10, y + 380), width=2)
        c.draw.line([(mid, y - 10), (mid, y + 380)], fill=(40, 40, 40), width=2)
        c.text(m, y, "Earnings", bold=True)
        c.text(mid + 20, y, "Deductions", bold=True)
        earn = {"Basic": self.rng.uniform(20000, 90000)}
        earn |= {"HRA": earn["Basic"] * 0.4, "Special Allowance": self.rng.uniform(2000, 20000),
                 "Conveyance": 1600.0}
        ded = {"Provident Fund": earn["Basic"] * 0.12, "Professional Tax": 200.0,
               "Income Tax": self.rng.uniform(0, 12000)}
        for i, (k, v) in enumerate(earn.items()):
            c.text(m, y + 50 + i * 40, k)
            c.text(mid - 20, y + 50 + i * 40, f"{v:,.2f}", anchor="ra")
        for i, (k, v) in enumerate(ded.items()):
            c.text(mid + 20, y + 50 + i * 40, k)
            c.text(W - m, y + 50 + i * 40, f"{v:,.2f}", anchor="ra")
        net = sum(earn.values()) - sum(ded.values())
        f["net_pay"] = f"{net:,.2f}"
        c.text(m, y + 330, f"Gross: {sum(earn.values()):,.2f}", bold=True)
        c.text(mid + 20, y + 330, f"Total Ded: {sum(ded.values()):,.2f}", bold=True)
        c.text(m, y + 430, f"{self.label('net_pay')} {f['net_pay']}", size=c.size + 3, bold=True)
        return c, f

    def id_card(self) -> tuple[Canvas, dict[str, str]]:
        page = Image.new("RGB", self.PAGE, (255, 255, 255))
        cw, ch = 680, 430
        ox, oy = self.rng.randint(80, 240), self.rng.randint(80, 400)
        card = Image.new("RGB", (cw, ch), tuple(self.rng.randint(225, 245) for _ in range(3)))
        c = Canvas(cw, ch, self.rng.choice(self.fonts), self.rng.randint(20, 23), ox, oy, img=card)
        f = {"id_number": "".join(self.rng.choices(string.ascii_uppercase, k=5)) + self.digits(4)
                          + self.rng.choice(string.ascii_uppercase),
             "name": self.pool.person().upper(), "date_of_birth": self.date(1960, 2004)}
        c.rect((0, 0, cw - 1, ch - 1), width=3)
        c.draw.rectangle((0, 0, cw, 60), fill=(60, 90, 150))
        c.text(cw // 2, 15, "INCOME TAX DEPARTMENT", size=24, bold=True, anchor="ma", fill=(255, 255, 255))
        c.rect((cw - 190, 85, cw - 30, 285), width=2, fill=(200, 200, 205))
        y = 90
        c.text(25, y, f"{self.label('name')} {f['name']}")
        c.text(25, y + 50, f"Father's Name: {self.pool.person().upper()}")
        c.text(25, y + 100, f"{self.label('date_of_birth')} {f['date_of_birth']}")
        c.text(25, y + 170, self.label("id_number"), size=c.size - 2)
        c.text(25, y + 205, f["id_number"], size=c.size + 8, bold=True)
        c.text(cw - 110, ch - 50, "Signature", size=16, anchor="ma")
        page.paste(card, (ox, oy))
        out = Canvas(*self.PAGE, c.fonts, c.size, img=page)
        out.items = c.items
        return out, f

    def receipt(self) -> tuple[Canvas, dict[str, str]]:
        page = Image.new("RGB", self.PAGE, (255, 255, 255))
        rw = self.rng.randint(400, 480)
        ox, oy = self.rng.randint(100, self.PAGE[0] - rw - 100), self.rng.randint(40, 120)
        n_items = self.rng.randint(4, 9)
        rh = 420 + n_items * 36
        strip = Image.new("RGB", (rw, rh), (250, 250, 246))
        c = Canvas(rw, rh, self.rng.choice(self.fonts), self.rng.randint(17, 19), ox, oy, img=strip)
        f = {"receipt_number": self.digits(6), "date": self.date(), "total_amount": ""}
        c.text(rw // 2, 20, f"{self.pool.word(2).upper()} MART", size=26, bold=True, anchor="ma")
        c.text(rw // 2, 60, f"{self.pool.word()} Road, {self.pool.word()}", anchor="ma")
        c.text(20, 105, f"{self.label('receipt_number')} {f['receipt_number']}")
        c.text(20, 135, f"{self.label('date')} {f['date']}")
        y = 175
        c.draw.line([(15, y), (rw - 15, y)], fill=(80, 80, 80), width=1)
        y += 12
        tot = 0.0
        for it in self.rng.sample(GROCERY, n_items):
            p = self.rng.uniform(20, 600)
            tot += p
            c.text(20, y, it)
            c.text(rw - 20, y, f"{p:,.2f}", anchor="ra")
            y += 36
        c.draw.line([(15, y + 5), (rw - 15, y + 5)], fill=(80, 80, 80), width=1)
        f["total_amount"] = f"{tot:,.2f}"
        c.text(20, y + 20, f"{self.label('total_amount')} {f['total_amount']}", size=c.size + 3, bold=True)
        c.text(rw // 2, y + 80, "THANK YOU, VISIT AGAIN", anchor="ma")
        page.paste(strip, (ox, oy))
        out = Canvas(*self.PAGE, c.fonts, c.size, img=page)
        out.items = c.items
        return out, f

    # ---- degradation ---------------------------------------------------
    def degrade(self, img: np.ndarray) -> tuple[np.ndarray, str]:
        rng, nr = self.rng, self.np_rng
        h, w = img.shape[:2]
        if rng.random() < 0.35:  # phone photo: page on a background, perspective + lighting gradient
            capture = "photo"
            pad = int(0.12 * max(h, w))
            bg_col = np.array([rng.randint(40, 120) for _ in range(3)], dtype=np.float32)
            canvas = np.clip(bg_col + nr.normal(0, 12, (h + 2 * pad, w + 2 * pad, 3)), 0, 255).astype(np.uint8)
            canvas[pad:pad + h, pad:pad + w] = img
            j = 0.06
            src = np.float32([[pad, pad], [pad + w, pad], [pad + w, pad + h], [pad, pad + h]])
            dst = src + np.float32([[rng.uniform(-j, j) * w, rng.uniform(-j, j) * h] for _ in range(4)])
            M = cv2.getPerspectiveTransform(src, dst)
            out = cv2.warpPerspective(canvas, M, (canvas.shape[1], canvas.shape[0]), borderMode=cv2.BORDER_REPLICATE)
            gx = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.9, 1.05), out.shape[1], dtype=np.float32)
            out = np.clip(out.astype(np.float32) * gx[None, :, None], 0, 255).astype(np.uint8)
            out = cv2.resize(out, (int(out.shape[1] * 0.85), int(out.shape[0] * 0.85)), interpolation=cv2.INTER_AREA)
        else:  # flatbed scan: small skew, toner noise
            capture = "scan"
            angle = rng.uniform(-4, 4)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            out = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
        if rng.random() < 0.5:
            k = rng.choice([3, 5])
            out = cv2.GaussianBlur(out, (k, k), 0)
        out = np.clip(out.astype(np.float32) * rng.uniform(0.8, 1.0) + rng.uniform(0, 30)
                      + nr.normal(0, rng.uniform(3, 12), out.shape), 0, 255).astype(np.uint8)
        if rng.random() < 0.3:  # salt-and-pepper specks
            mask = nr.random(out.shape[:2]) < 0.002
            out[mask] = 0
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(55, 90)])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR), capture

    def generate(self, doc_type: str) -> Sample:
        canvas, fields = getattr(self, doc_type)()
        clean = cv2.cvtColor(np.asarray(canvas.img), cv2.COLOR_RGB2BGR)
        degraded, capture = self.degrade(clean)
        return Sample(doc_type, fields, canvas.items, degraded, clean, capture)


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def to_pil(img: np.ndarray) -> Image.Image:
    return Image.open(io.BytesIO(encode_png(img)))
