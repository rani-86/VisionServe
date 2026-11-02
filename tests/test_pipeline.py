from __future__ import annotations

import cv2
import numpy as np
import pytest

from docserve.extraction import extract_fields, normalize_truth, normalize_value
from docserve.preprocessing import ImageDecodeError, decode_image, estimate_skew, find_page_quad, preprocess
from docserve.synth import DOC_TYPES, DocumentGenerator
from docserve.text_metrics import cer, wer


@pytest.mark.parametrize("angle", [-4.0, -1.5, 2.0, 3.5])
def test_deskew_recovers_known_rotation(angle):
    clean = DocumentGenerator("test", seed=3).generate("invoice").clean
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), borderValue=255)
    assert estimate_skew(rotated) == pytest.approx(-angle, abs=0.3)


def test_perspective_detected_on_photos_not_scans():
    g = DocumentGenerator("test", seed=11)
    photos = scans = 0
    for _ in range(12):
        s = g.generate("payslip")
        gray = cv2.cvtColor(s.image, cv2.COLOR_BGR2GRAY)
        found = find_page_quad(gray) is not None
        if s.capture == "photo":
            photos += 1
            assert found
        else:
            scans += 1
            assert not found
    assert photos and scans


def test_preprocess_outputs():
    s = DocumentGenerator("test", seed=4).generate("bank_statement")
    r = preprocess(s.image)
    assert r.gray.ndim == 2 and r.binary.shape == r.gray.shape
    assert set(np.unique(r.binary)) <= {0, 255}
    assert r.steps[0] == "grayscale" and "roi_crop" in r.steps


def test_decode_rejects_bad_input():
    with pytest.raises(ImageDecodeError):
        decode_image(b"")
    with pytest.raises(ImageDecodeError):
        decode_image(b"%PDF-1.7 not an image")
    tiny = cv2.imencode(".png", np.zeros((10, 10, 3), np.uint8))[1].tobytes()
    with pytest.raises(ImageDecodeError):
        decode_image(tiny)
    big = cv2.imencode(".png", np.zeros((3000, 3000, 3), np.uint8))[1].tobytes()
    with pytest.raises(ImageDecodeError):
        decode_image(big, max_pixels=1_000_000)


@pytest.mark.parametrize("doc_type", DOC_TYPES)
def test_extraction_on_ground_truth_text(doc_type):
    """With perfect OCR, every field of every layout/label variant must be extracted."""
    g = DocumentGenerator("val", seed=9)
    for _ in range(25):
        s = g.generate(doc_type)
        got = extract_fields(doc_type, s.lines)
        for k, v in s.fields.items():
            assert got[k] == normalize_truth(doc_type, k, v), (k, s.lines)


@pytest.mark.parametrize("kind,raw,expected", [
    ("pan", "ABCD E1234F", "ABCDE1234F"),
    ("pan", "A8CDE12O4F", "ABCDE1204F"),
    ("ifsc", "HDFC0OO1234", "HDFC0001234"),
    ("digits", "12O45 67l90", "1204567190"),
    ("code", "INV- 4O213", "INV-40213"),
    ("amount", "1,23,456.70", "123,456.70"),
    ("amount", "12 345.00", "12,345.00"),
    ("date", "5 / 3 / 2024", "05/03/2024"),
])
def test_ocr_confusion_normalisation(kind, raw, expected):
    assert normalize_value(kind, raw) == expected


def test_cer_wer():
    assert cer("hello world", "hello world") == 0
    assert cer("abcd", "abed") == pytest.approx(0.25)
    assert wer("total amount 100", "total amont 100") == pytest.approx(1 / 3)


@pytest.mark.parametrize("doc_type,rows,field,expected", [
    ("bank_statement", ["Balance as on date: 104, 784.81"], "closing_balance", "104,784.81"),
    ("bank_statement", ["Baiance as on date: 51,194.73"], "closing_balance", "51,194.73"),  # OCR typo in label
    ("bank_statement", ["07/02 UPI/X 8,582.99 64,773.73", "Closing Bal: 77,094. 70"], "closing_balance", "77,094.70"),
    ("id_card", ["Perrmanent Account Number.", "OVZQX7935R"], "id_number", "OVZQX7935R"),
    ("invoice", ["TOTAL: 1234.50"], "total_amount", "1,234.50"),
    ("invoice", ["Subtotal 9,999.00", "Grand Total: 1,23,456.70"], "total_amount", "123,456.70"),
])
def test_extraction_tolerates_ocr_noise(doc_type, rows, field, expected):
    assert extract_fields(doc_type, rows)[field] == expected


def test_fuzzy_matching_does_not_fire_on_short_labels():
    assert extract_fields("id_card", ["PAM: ABCDE1234F"])["id_number"] is None
