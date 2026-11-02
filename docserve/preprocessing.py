"""OpenCV document preprocessing: geometry correction and image clean-up before OCR.

Pipeline (each step is recorded so its effect can be ablated in evaluation):
  decode -> grayscale -> perspective correction (photo of a page) -> deskew
  -> denoise -> contrast enhancement (CLAHE) -> ROI crop
  (+ adaptive threshold + morphology, lazily, for engines that want a binary image)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


class ImageDecodeError(ValueError):
    pass


@dataclass
class PreprocessResult:
    gray: np.ndarray  # enhanced grayscale, fed to PaddleOCR and the classifier
    skew_angle: float = 0.0
    perspective_corrected: bool = False
    steps: list[str] = field(default_factory=list)
    _binary: np.ndarray | None = field(default=None, repr=False)

    @property
    def binary(self) -> np.ndarray:
        """Binarised page for Tesseract; computed on first use since PP-OCR reads grayscale."""
        if self._binary is None:
            self._binary = binarize(self.gray)
        return self._binary


def decode_image(data: bytes, max_pixels: int = 40_000_000, min_side: int = 64) -> np.ndarray:
    """Decode upload bytes to BGR, rejecting corrupt, tiny or decompression-bomb images."""
    buf = np.frombuffer(data, dtype=np.uint8)
    if buf.size == 0:
        raise ImageDecodeError("empty file")
    # Read the header first so a huge image is rejected before allocating it.
    header = cv2.imdecode(buf, cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if header is None:
        raise ImageDecodeError("file is not a decodable image (PNG/JPEG/TIFF/BMP/WebP)")
    if header.shape[0] * header.shape[1] * 64 > max_pixels:
        raise ImageDecodeError(f"image exceeds {max_pixels:,} pixels")
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError("file is not a decodable image")
    if min(img.shape[:2]) < min_side:
        raise ImageDecodeError(f"image is smaller than {min_side}px on its short side")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def find_page_quad(gray: np.ndarray, min_area_frac: float = 0.3) -> np.ndarray | None:
    """Find the four corners of a page photographed on a darker background."""
    scale = 800 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    scale = min(scale, 1.0)
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    # Page is the bright region; Otsu separates it from the background better than edges on noisy photos.
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    h, w = small.shape[:2]
    # A page that already fills the frame (a scan) needs no warp.
    if area < min_area_frac * h * w or area > 0.97 * h * w:
        return None
    peri = cv2.arcLength(c, True)
    for eps in (0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return _order_corners(approx) / scale
    return None


def correct_perspective(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = quad
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _skew_scorer(gray: np.ndarray, long_side: int):
    scale = long_side / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    inv = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = inv.shape
    center = (w / 2, h / 2)

    def score(angle: float) -> float:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(inv, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        prof = cv2.reduce(rot, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32F).ravel()
        return float(np.sum(np.diff(prof) ** 2))

    return score


def estimate_skew(gray: np.ndarray, max_angle: float = 8.0) -> float:
    """Skew angle in degrees via projection profiles: text rows are sharpest when level.

    Coarse 0.5-degree search on a 500px thumbnail, then a 0.1-degree search around the
    best angle at 1000px. Positive result = rotate by this angle to level the text.
    """
    coarse_score = _skew_scorer(gray, 500)
    coarse = np.arange(-max_angle, max_angle + 1e-6, 0.5)
    best = float(coarse[np.argmax([coarse_score(a) for a in coarse])])
    fine_score = _skew_scorer(gray, 1000)
    fine = np.arange(best - 0.5, best + 0.5 + 1e-6, 0.1)
    return float(fine[np.argmax([fine_score(a) for a in fine])])


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def denoise(gray: np.ndarray, strong: bool = False) -> np.ndarray:
    """Remove salt-and-pepper specks only when they are present.

    A 3x3 median erodes thin strokes of small text ("/", "#", "l"), which cost ~7 points of
    field accuracy on receipts in A/B tests, so it runs only when impulse noise is detected.
    Non-local means (`strong`) is gentler on strokes but ~20x slower.
    """
    if strong:
        return cv2.fastNlMeansDenoising(gray, None, h=10)
    med = cv2.medianBlur(gray, 3)
    impulse = float((cv2.absdiff(gray, med) > 100).mean())
    return med if impulse > 5e-4 else gray


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def crop_roi(gray: np.ndarray, margin: int = 20) -> np.ndarray:
    """Crop to the bounding box of ink, dropping empty borders."""
    inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    pts = cv2.findNonZero(inv)
    if pts is None:
        return gray
    x, y, w, h = cv2.boundingRect(pts)
    H, W = gray.shape
    if w * h < 0.05 * H * W:  # nearly blank page: keep everything
        return gray
    return gray[max(0, y - margin):min(H, y + h + margin), max(0, x - margin):min(W, x + w + margin)]


def binarize(gray: np.ndarray) -> np.ndarray:
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    # Opening on the inverted image removes isolated specks without eroding strokes.
    inv = cv2.morphologyEx(255 - binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return 255 - inv


def preprocess(img: np.ndarray, *, strong_denoise: bool = False, enabled: bool = True) -> PreprocessResult:
    gray = to_gray(img)
    if not enabled:
        return PreprocessResult(gray=gray, steps=["grayscale"])
    steps = ["grayscale"]
    corrected = False
    quad = find_page_quad(gray)
    if quad is not None:
        gray = correct_perspective(gray, quad)
        corrected = True
        steps.append("perspective")
    angle = estimate_skew(gray)
    if abs(angle) >= 0.2:
        gray = rotate(gray, angle)
        steps.append("deskew")
    gray = denoise(gray, strong_denoise)
    gray = enhance_contrast(gray)
    gray = crop_roi(gray)
    steps += ["denoise", "clahe", "roi_crop"]
    return PreprocessResult(gray=gray, skew_angle=angle,
                            perspective_corrected=corrected, steps=steps)


def classifier_tensor(gray: np.ndarray, size: int, mean: list[float], std: list[float]) -> np.ndarray:
    """Letterbox to size x size, replicate to 3 channels, normalise. Returns float32 CHW."""
    h, w = gray.shape[:2]
    s = size / max(h, w)
    resized = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size), 255, dtype=np.uint8)
    y0, x0 = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    x = np.repeat((canvas.astype(np.float32) / 255.0)[None], 3, axis=0)
    return (x - np.asarray(mean, np.float32)[:, None, None]) / np.asarray(std, np.float32)[:, None, None]
