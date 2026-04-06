#!/usr/bin/env python3
"""
Multi-slide Deep Zoom tile server for CAMELYON WSI Viewer.

Supports:
- Slide upload & local storage
- Slide metadata (magnification, channels, vendor)
- Color deconvolution (H&E / H-DAB separation via scikit-image)
- Per-slide DeepZoom tile serving
- Annotation & heatmap overlays
- Mock model-based analysis (HoVerNet, SAM, etc.)

Endpoints (under /api):
  /api/slides                                    -> List all slides
  /api/slides/upload                             -> Upload a new slide (POST)
  /api/slides/{slide_id}/info                    -> Slide metadata
  /api/slides/{slide_id}/dzi                     -> DZI XML for base WSI
  /api/slides/{slide_id}/tile/{l}/{c}_{r}.jpeg   -> Base WSI tile
  /api/slides/{slide_id}/deconvolve/{channel}/dzi -> DZI for deconvolved channel
  /api/slides/{slide_id}/deconvolve/{channel}/tile/{l}/{c}_{r}.png -> Deconvolved tile
  /api/slides/{slide_id}/overlay/annotations/...  -> Annotation overlay
  /api/slides/{slide_id}/overlay/heatmap/...      -> Heatmap overlay
  /api/slides/{slide_id}/overlay/tissue/...       -> Tissue mask overlay
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import random
import shutil
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import numpy as np
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image, ImageDraw, ImageFilter
from pydantic import BaseModel
from skimage.color import rgb2hed, hed2rgb

from server.inference_service import InferenceJobManager, JobStatus

# ── Configuration ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DEMO_SLIDE_PATH = ROOT / "data/camelyon17/training/center_0/patient_010_node_4.tif"
DEMO_ANNOTATION_XML = ROOT / "data/camelyon17/training/center_0/patient_010_node_4.xml"
TILE_SIZE = 256
OVERLAP = 0
LIMIT_BOUNDS = True
HEATMAP_CELL_SIZE = 2048
HEATMAP_OFFSET = (-2048, 0)

HEATMAP_PALETTES = {
    "metastasis": [
        (0.0, (40, 70, 170)),
        (0.33, (0, 180, 180)),
        (0.66, (255, 220, 80)),
        (1.0, (255, 50, 50)),
    ],
    "epithelial": [
        (0.0, (20, 100, 130)),
        (0.33, (20, 170, 150)),
        (0.66, (190, 210, 120)),
        (1.0, (240, 190, 90)),
    ],
    "normal": [
        (0.0, (40, 120, 70)),
        (0.33, (80, 180, 80)),
        (0.66, (200, 210, 120)),
        (1.0, (230, 190, 110)),
    ],
    "inflammatory": [
        (0.0, (80, 30, 120)),
        (0.33, (140, 60, 160)),
        (0.66, (200, 100, 180)),
        (1.0, (240, 150, 200)),
    ],
    "necrosis": [
        (0.0, (60, 60, 60)),
        (0.33, (120, 80, 40)),
        (0.66, (180, 120, 60)),
        (1.0, (220, 160, 80)),
    ],
}

# Model definitions
# PanNuke type mapping from type_info.json:
# 0=nolabel, 1=neoplastic, 2=inflammatory, 3=connective, 4=necrosis, 5=non-neoplastic
HOVERNET_CELL_TYPES = ["neoplastic", "inflammatory", "connective", "necrosis", "non-neoplastic"]
HOVERNET_TYPE_COLORS = {
    "nolabe": (128, 128, 128),
    "neopla": (255, 0, 0),
    "inflam": (0, 255, 0),
    "connec": (0, 0, 255),
    "necros": (255, 255, 0),
    "no-neo": (255, 165, 0),
}

MODELS = [
    {
        "id": "hovernet",
        "label": "HoVerNet (PanNuke)",
        "description": "Real nuclear segmentation & classification. Runs inference on your machine using MPS/CPU.",
        "cell_types": HOVERNET_CELL_TYPES,
        "real": True,
    },
    {
        "id": "mock-default",
        "label": "Mock Default",
        "description": "Default mock model for demonstration (random heatmap).",
        "cell_types": ["metastasis", "epithelial", "normal"],
        "real": False,
    },
    {
        "id": "mock-hi-sens",
        "label": "Mock High Sensitivity",
        "description": "Mock model calibrated for high sensitivity (random heatmap).",
        "cell_types": ["metastasis", "epithelial", "normal"],
        "real": False,
    },
    {
        "id": "mock-hi-spec",
        "label": "Mock High Specificity",
        "description": "Mock model calibrated for high specificity (random heatmap).",
        "cell_types": ["metastasis", "epithelial", "normal"],
        "real": False,
    },
]


# ── Slide Registry (in-memory) ──────────────────────────────────────────

class SlideEntry:
    """Manages one slide's OpenSlide + DeepZoomGenerator."""

    def __init__(self, path: Path, annotation_xml: Optional[Path] = None, display_name: Optional[str] = None):
        self.path = path
        self.annotation_xml = annotation_xml
        self.display_name = display_name or path.stem
        self.slide = OpenSlide(str(path))
        self.dz = DeepZoomGenerator(
            self.slide, tile_size=TILE_SIZE, overlap=OVERLAP, limit_bounds=LIMIT_BOUNDS
        )
        self.l0_downsamples = tuple(
            2 ** (self.dz.level_count - dz_level - 1)
            for dz_level in range(self.dz.level_count)
        )
        self._polygons_cache = None
        self._tissue_mask = None
        self._tissue_mask_size = None

    @property
    def slide_id(self) -> str:
        return hashlib.sha256(str(self.path).encode()).hexdigest()[:12]

    @property
    def polygons(self):
        if self._polygons_cache is None:
            if self.annotation_xml and self.annotation_xml.exists():
                self._polygons_cache = _load_polygons(self.annotation_xml)
            else:
                self._polygons_cache = []
        return self._polygons_cache

    @property
    def tissue_mask(self):
        """Low-res boolean mask: True where tissue exists (not white/black background)."""
        if self._tissue_mask is None:
            thumb_size = (512, 512)
            thumb = self.slide.get_thumbnail(thumb_size)
            arr = np.array(thumb.convert("RGB"))
            self._tissue_mask_size = (arr.shape[1], arr.shape[0])  # (w, h)
            gray = np.mean(arr, axis=2)
            saturation = np.max(arr, axis=2).astype(float) - np.min(arr, axis=2).astype(float)
            # Tissue: NOT white AND NOT black, and has some color saturation
            is_not_white = gray < 220
            is_not_black = gray > 25
            has_color = saturation > 15
            self._tissue_mask = is_not_white & is_not_black & has_color
        return self._tissue_mask

    def is_tissue(self, l0_x: float, l0_y: float) -> bool:
        """Check if level-0 coordinate falls on tissue."""
        mask = self.tissue_mask
        w, h = self.slide.dimensions
        mw, mh = self._tissue_mask_size
        mx = int(l0_x / w * mw)
        my = int(l0_y / h * mh)
        if 0 <= mx < mw and 0 <= my < mh:
            return bool(mask[my, mx])
        return False

    def get_properties(self) -> dict:
        props = dict(self.slide.properties)
        magnification = props.get("openslide.objective-power", "Unknown")
        vendor = props.get("openslide.vendor", "Unknown")
        mpp_x = props.get("openslide.mpp-x", None)
        mpp_y = props.get("openslide.mpp-y", None)

        # Detect stain channels
        channels = self._detect_channels()

        return {
            "slide_id": self.slide_id,
            "filename": self.path.name,
            "display_name": self.display_name,
            "dimensions": self.slide.dimensions,
            "magnification": magnification,
            "vendor": vendor,
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,
            "level_count": self.dz.level_count,
            "tile_size": TILE_SIZE,
            "channels": channels,
            "annotations_count": len(self.polygons),
            "file_size_mb": round(self.path.stat().st_size / (1024 * 1024), 1),
        }

    def _detect_channels(self) -> list[dict]:
        """Detect available stain channels. For H&E and H-DAB staining."""
        channels = [
            {"id": "original", "label": "Original (RGB)", "description": "Full color composite"},
            {"id": "hematoxylin", "label": "Hematoxylin (H)", "description": "Nuclear stain — blue/purple"},
            {"id": "eosin", "label": "Eosin (E)", "description": "Cytoplasm/ECM stain — pink"},
            {"id": "dab", "label": "DAB", "description": "IHC chromogen — brown"},
        ]
        return channels


# Global slide registry
_slide_registry: dict[str, SlideEntry] = {}


def _register_slide(path: Path, annotation_xml: Optional[Path] = None, display_name: Optional[str] = None) -> SlideEntry:
    entry = SlideEntry(path, annotation_xml, display_name)
    _slide_registry[entry.slide_id] = entry
    return entry


def _get_slide(slide_id: str) -> SlideEntry:
    if slide_id not in _slide_registry:
        raise HTTPException(status_code=404, detail=f"Slide '{slide_id}' not found")
    return _slide_registry[slide_id]


# ── Helper Functions ─────────────────────────────────────────────────────

def _load_polygons(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    polys = []
    for ann in root.findall(".//Annotation"):
        coords = [
            (float(c.attrib["X"]), float(c.attrib["Y"]))
            for c in ann.findall(".//Coordinate")
        ]
        if len(coords) < 3:
            continue
        label = ann.attrib.get("PartOfGroup") or ann.attrib.get("Name") or "unknown"
        color = ann.attrib.get("Color") or "#F4FA58"
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        polys.append({
            "label": label,
            "coords": coords,
            "color": color,
            "bbox": (min(xs), min(ys), max(xs), max(ys)),
        })
    return polys


def _polys_bbox(polys):
    xs, ys = [], []
    for p in polys:
        x0, y0, x1, y1 = p["bbox"]
        xs.extend([x0, x1])
        ys.extend([y0, y1])
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _dzi_xml(width: int, height: int, tile_size: int, overlap: int, fmt: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" TileSize="{tile_size}" Overlap="{overlap}" Format="{fmt}">
    <Size Width="{width}" Height="{height}"/>
</Image>
"""


def _model_seed(model: str) -> int:
    digest = hashlib.sha256(model.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _smooth_noise_score(x_l0: float, y_l0: float, spacing: float, model_seed: int) -> float:
    gx = x_l0 / spacing
    gy = y_l0 / spacing
    x0 = math.floor(gx)
    y0 = math.floor(gy)
    tx = gx - x0
    ty = gy - y0

    def val(ix, iy):
        rng = random.Random((ix * 73856093) ^ (iy * 19349663) ^ model_seed)
        return rng.random()

    v00 = val(x0, y0)
    v10 = val(x0 + 1, y0)
    v01 = val(x0, y0 + 1)
    v11 = val(x0 + 1, y0 + 1)
    v0 = v00 * (1 - tx) + v10 * tx
    v1 = v01 * (1 - tx) + v11 * tx
    return v0 * (1 - ty) + v1 * ty


def _colormap(score: float, palette_name: str = "metastasis"):
    score = max(0.0, min(1.0, score))
    stops = HEATMAP_PALETTES.get(palette_name, HEATMAP_PALETTES["metastasis"])
    for i in range(len(stops) - 1):
        s0, c0 = stops[i]
        s1, c1 = stops[i + 1]
        if score <= s1:
            t = (score - s0) / (s1 - s0 + 1e-9)
            r = int(c0[0] + t * (c1[0] - c0[0]))
            g = int(c0[1] + t * (c1[1] - c0[1]))
            b = int(c0[2] + t * (c1[2] - c0[2]))
            break
    else:
        r, g, b = stops[-1][1]
    alpha = int(70 + 90 * score)
    return (r, g, b, alpha)


def _draw_annotations_tile(polys, tile, lvl, col, row, tile_size, l0_downsamples, offset):
    draw = ImageDraw.Draw(tile, "RGBA")
    down = l0_downsamples[lvl]
    origin_x = col * tile_size
    origin_y = row * tile_size
    for poly in polys:
        x0, y0, x1, y1 = poly["bbox"]
        zx0 = (x0 - offset[0]) / down
        zy0 = (y0 - offset[1]) / down
        zx1 = (x1 - offset[0]) / down
        zy1 = (y1 - offset[1]) / down
        if zx1 < origin_x or zy1 < origin_y:
            continue
        if zx0 > origin_x + tile.width or zy0 > origin_y + tile.height:
            continue
        coords = [
            ((x - offset[0]) / down - origin_x, (y - offset[1]) / down - origin_y)
            for x, y in poly["coords"]
        ]
        color = poly.get("color") or "#F4FA58"
        draw.polygon(coords, outline=color, fill=color + "55")


def _draw_tissue_tile(tile, lvl, col, row, tile_size, l0_downsamples, offset, bbox):
    if bbox is None:
        return
    draw = ImageDraw.Draw(tile, "RGBA")
    down = l0_downsamples[lvl]
    origin_x = col * tile_size
    origin_y = row * tile_size
    x0, y0, x1, y1 = bbox
    pad = 4096
    x0 -= pad
    y0 -= pad
    x1 += pad
    y1 += pad
    zx0 = (x0 - offset[0]) / down - origin_x
    zy0 = (y0 - offset[1]) / down - origin_y
    zx1 = (x1 - offset[0]) / down - origin_x
    zy1 = (y1 - offset[1]) / down - origin_y
    draw.rectangle([zx0, zy0, zx1, zy1], fill=(180, 220, 255, 50), outline=None)


def _draw_heatmap_tile(tile, lvl, col, row, tile_size, l0_downsamples, offset, spacing,
                       model_seed, palette, bbox=None, field_offset=(0, 0), draw_vector=False,
                       tissue_check=None):
    down = l0_downsamples[lvl]
    origin_x = col * tile_size
    origin_y = row * tile_size
    pix = tile.load()

    bx0 = by0 = bx1 = by1 = None
    if bbox:
        bx0, by0, bx1, by1 = bbox
        pad = spacing * 2
        bx0 -= pad
        by0 -= pad
        bx1 += pad
        by1 += pad

    for y in range(tile.height):
        l0_y = (origin_y + y) * down + offset[1]
        if bbox and (l0_y < by0 or l0_y > by1):
            continue
        for x in range(tile.width):
            l0_x = (origin_x + x) * down + offset[0]
            if bbox and (l0_x < bx0 or l0_x > bx1):
                continue
            # Skip background pixels (only render on tissue)
            if tissue_check and not tissue_check(l0_x, l0_y):
                continue
            score = _smooth_noise_score(
                l0_x + field_offset[0], l0_y + field_offset[1], spacing, model_seed
            )
            r, g, b, a = _colormap(score, palette)
            pix[x, y] = (r, g, b, a)

    if draw_vector and tile_size >= 128:
        draw = ImageDraw.Draw(tile, "RGBA")
        step = 32
        max_arrow_len = 14
        min_arrow_len = 4
        delta = spacing * 0.1

        for y in range(step // 2, tile.height, step):
            l0_y = (origin_y + y) * down + offset[1]
            if bbox and (l0_y < by0 or l0_y > by1):
                continue
            for x in range(step // 2, tile.width, step):
                l0_x = (origin_x + x) * down + offset[0]
                if bbox and (l0_x < bx0 or l0_x > bx1):
                    continue
                if tissue_check and not tissue_check(l0_x, l0_y):
                    continue

                lx = l0_x + field_offset[0]
                ly = l0_y + field_offset[1]
                s_right = _smooth_noise_score(lx + delta, ly, spacing, model_seed)
                s_left = _smooth_noise_score(lx - delta, ly, spacing, model_seed)
                s_down = _smooth_noise_score(lx, ly + delta, spacing, model_seed)
                s_up = _smooth_noise_score(lx, ly - delta, spacing, model_seed)

                dx = s_right - s_left
                dy = s_down - s_up
                mag = math.hypot(dx, dy)
                norm_mag = min(1.0, mag / 0.05)

                if norm_mag < 0.1:
                    continue

                dx /= mag
                dy /= mag
                arrow_len = min_arrow_len + (max_arrow_len - min_arrow_len) * norm_mag
                alpha = int(100 + 155 * norm_mag)

                end_x = x + dx * arrow_len
                end_y = y + dy * arrow_len

                shadow_color = (0, 0, 0, int(alpha * 0.6))
                sw = 1
                draw.line([(x + sw, y + sw), (end_x + sw, end_y + sw)], fill=shadow_color, width=1)
                color = (255, 255, 255, alpha)
                draw.line([(x, y), (end_x, end_y)], fill=color, width=1)

                angle = math.atan2(dy, dx)
                head_len = 4 + 2 * norm_mag
                angle1 = angle + math.pi * 0.90
                angle2 = angle - math.pi * 0.90
                hx1 = end_x + math.cos(angle1) * head_len
                hy1 = end_y + math.sin(angle1) * head_len
                hx2 = end_x + math.cos(angle2) * head_len
                hy2 = end_y + math.sin(angle2) * head_len
                draw.polygon([(end_x + sw, end_y + sw), (hx1 + sw, hy1 + sw), (hx2 + sw, hy2 + sw)], fill=shadow_color)
                draw.polygon([(end_x, end_y), (hx1, hy1), (hx2, hy2)], fill=color)


def _deconvolve_tile(tile_img: Image.Image, channel: str) -> Image.Image:
    """Apply color deconvolution and return the requested channel as a false-color image."""
    arr = np.array(tile_img.convert("RGB")).astype(np.float64) / 255.0

    # Clamp to avoid log issues with pure white/black
    arr = np.clip(arr, 1e-6, 1.0)

    # rgb2hed returns Hematoxylin, Eosin, DAB in channels 0, 1, 2
    hed = rgb2hed(arr)

    if channel == "hematoxylin":
        # Reconstruct only hematoxylin channel
        out = np.zeros_like(hed)
        out[:, :, 0] = hed[:, :, 0]
        rgb = hed2rgb(out)
    elif channel == "eosin":
        out = np.zeros_like(hed)
        out[:, :, 1] = hed[:, :, 1]
        rgb = hed2rgb(out)
    elif channel == "dab":
        out = np.zeros_like(hed)
        out[:, :, 2] = hed[:, :, 2]
        rgb = hed2rgb(out)
    else:
        rgb = arr

    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


# ── FastAPI App ──────────────────────────────────────────────────────────

app = FastAPI(
    title="CAMELYON Multi-Slide Deep Zoom Server",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

web_root = ROOT / "web"
if web_root.exists():
    app.mount("/web", StaticFiles(directory=web_root, html=True), name="web")


@app.get("/")
def root():
    if not web_root.exists():
        return {"message": "web UI not found", "visit": "/api/docs"}
    landing_path = web_root / "landing.html"
    if landing_path.exists():
        return FileResponse(landing_path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    index_path = web_root / "index.html"
    if not index_path.exists():
        return {"message": "index.html not found in web/", "visit": "/api/docs"}
    return FileResponse(index_path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/app")
def app_page():
    if not web_root.exists():
        return {"message": "web UI not found", "visit": "/api/docs"}
    index_path = web_root / "index.html"
    if not index_path.exists():
        return {"message": "index.html not found in web/", "visit": "/api/docs"}
    return FileResponse(index_path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


# ── Register demo slide at startup ───────────────────────────────────────

def _init_demo():
    if DEMO_SLIDE_PATH.exists():
        ann = DEMO_ANNOTATION_XML if DEMO_ANNOTATION_XML.exists() else None
        _register_slide(DEMO_SLIDE_PATH, ann, display_name="Patient 010 Node 4 (Demo)")
    # Register any previously uploaded slides
    for f in UPLOAD_DIR.iterdir():
        if f.suffix.lower() in (".tif", ".tiff", ".svs", ".ndpi", ".mrxs"):
            xml_candidate = f.with_suffix(".xml")
            ann = xml_candidate if xml_candidate.exists() else None
            _register_slide(f, ann)


_init_demo()

# ── Inference Job Manager ────────────────────────────────────────────────
inference_manager = InferenceJobManager(ROOT)


class InferenceStartRequest(BaseModel):
    model_id: str = "hovernet"
    roi: Optional[dict] = None  # {x, y, width, height} in level-0 coords
    device: str = "auto"


# ── Slide Management ────────────────────────────────────────────────────

@app.get("/api/slides")
def list_slides():
    return [entry.get_properties() for entry in _slide_registry.values()]


@app.post("/api/slides/upload")
async def upload_slide(file: UploadFile = File(...)):
    valid_exts = {".tif", ".tiff", ".svs", ".ndpi", ".mrxs"}
    ext = Path(file.filename or "unknown.tif").suffix.lower()
    if ext not in valid_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Accepted: {', '.join(valid_exts)}",
        )

    dest = UPLOAD_DIR / (file.filename or f"slide_{uuid.uuid4().hex[:8]}{ext}")

    # If file already exists, check if it's already registered
    if dest.exists():
        existing_id = hashlib.sha256(str(dest).encode()).hexdigest()[:12]
        if existing_id in _slide_registry:
            return _slide_registry[existing_id].get_properties()

    # Write in chunks to avoid blocking for large files
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            f.write(chunk)

    try:
        entry = _register_slide(dest)
        return entry.get_properties()
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to open slide: {e}")


@app.delete("/api/slides/{slide_id}")
def delete_slide(slide_id: str):
    entry = _get_slide(slide_id)
    # Don't delete demo slide's file
    if entry.path == DEMO_SLIDE_PATH:
        raise HTTPException(status_code=403, detail="Cannot delete the demo slide")
    entry.slide.close()
    entry.path.unlink(missing_ok=True)
    del _slide_registry[slide_id]
    return {"status": "deleted", "slide_id": slide_id}


@app.get("/api/slides/{slide_id}/info")
def slide_info(slide_id: str):
    entry = _get_slide(slide_id)
    info = entry.get_properties()
    info["models"] = MODELS
    return info


# ── Per-Slide DZI & Tiles ───────────────────────────────────────────────

@app.get("/api/slides/{slide_id}/dzi")
def get_slide_dzi(slide_id: str):
    entry = _get_slide(slide_id)
    xml = entry.dz.get_dzi("jpeg")
    return Response(xml, media_type="application/xml")


@app.get("/api/slides/{slide_id}/tile/{level}/{col}_{row}.jpeg")
def get_slide_tile(slide_id: str, level: int, col: int, row: int):
    entry = _get_slide(slide_id)
    dz = entry.dz
    if level < 0 or level >= dz.level_count:
        raise HTTPException(status_code=404, detail="Invalid level")
    cols, rows = dz.level_tiles[level]
    if col < 0 or row < 0 or col >= cols or row >= rows:
        raise HTTPException(status_code=404, detail="Invalid tile address")
    tile = dz.get_tile(level, (col, row))
    buf = io.BytesIO()
    tile.save(buf, format="JPEG")
    return Response(buf.getvalue(), media_type="image/jpeg")


# ── Color Deconvolution Tiles ────────────────────────────────────────────

@app.get("/api/slides/{slide_id}/deconvolve/{channel}/dzi")
def get_deconvolve_dzi(slide_id: str, channel: str):
    entry = _get_slide(slide_id)
    valid_channels = {"original", "hematoxylin", "eosin", "dab"}
    if channel not in valid_channels:
        raise HTTPException(status_code=400, detail=f"Invalid channel '{channel}'. Valid: {valid_channels}")
    w, h = entry.slide.dimensions
    xml = _dzi_xml(w, h, TILE_SIZE, OVERLAP, "png")
    return Response(xml, media_type="application/xml")


@app.get("/api/slides/{slide_id}/deconvolve/{channel}/tile/{level}/{col}_{row}.png")
def get_deconvolve_tile(slide_id: str, channel: str, level: int, col: int, row: int):
    entry = _get_slide(slide_id)
    dz = entry.dz
    valid_channels = {"original", "hematoxylin", "eosin", "dab"}
    if channel not in valid_channels:
        raise HTTPException(status_code=400, detail=f"Invalid channel")
    if level < 0 or level >= dz.level_count:
        raise HTTPException(status_code=404, detail="Invalid level")
    cols_count, rows_count = dz.level_tiles[level]
    if col < 0 or row < 0 or col >= cols_count or row >= rows_count:
        raise HTTPException(status_code=404, detail="Invalid tile address")

    tile = dz.get_tile(level, (col, row))

    if channel != "original":
        tile = _deconvolve_tile(tile, channel)

    buf = io.BytesIO()
    tile.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


# ── Annotation Overlay ───────────────────────────────────────────────────

@app.get("/api/slides/{slide_id}/overlay/annotations/dzi")
def get_annotations_dzi(slide_id: str):
    entry = _get_slide(slide_id)
    w, h = entry.slide.dimensions
    xml = _dzi_xml(w, h, TILE_SIZE, OVERLAP, "png")
    return Response(xml, media_type="application/xml")


@app.get("/api/slides/{slide_id}/overlay/annotations/tile/{level}/{col}_{row}.png")
def get_annotations_tile(slide_id: str, level: int, col: int, row: int):
    entry = _get_slide(slide_id)
    dz = entry.dz
    if level < 0 or level >= dz.level_count:
        raise HTTPException(status_code=404, detail="Invalid level")
    cols_count, rows_count = dz.level_tiles[level]
    if col < 0 or row < 0 or col >= cols_count or row >= rows_count:
        raise HTTPException(status_code=404, detail="Invalid tile address")
    _, z_size = dz._get_tile_info(level, (col, row))
    tile = Image.new("RGBA", z_size, (0, 0, 0, 0))
    _draw_annotations_tile(entry.polygons, tile, level, col, row, TILE_SIZE, entry.l0_downsamples, dz._l0_offset)
    buf = io.BytesIO()
    tile.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


# ── Tissue Mask Overlay ──────────────────────────────────────────────────

@app.get("/api/slides/{slide_id}/overlay/tissue/dzi")
def get_tissue_dzi(slide_id: str):
    entry = _get_slide(slide_id)
    w, h = entry.slide.dimensions
    xml = _dzi_xml(w, h, TILE_SIZE, OVERLAP, "png")
    return Response(xml, media_type="application/xml")


@app.get("/api/slides/{slide_id}/overlay/tissue/tile/{level}/{col}_{row}.png")
def get_tissue_tile(slide_id: str, level: int, col: int, row: int):
    entry = _get_slide(slide_id)
    dz = entry.dz
    if level < 0 or level >= dz.level_count:
        raise HTTPException(status_code=404, detail="Invalid level")
    cols_count, rows_count = dz.level_tiles[level]
    if col < 0 or row < 0 or col >= cols_count or row >= rows_count:
        raise HTTPException(status_code=404, detail="Invalid tile address")
    _, z_size = dz._get_tile_info(level, (col, row))
    tile = Image.new("RGBA", z_size, (0, 0, 0, 0))
    bbox = _polys_bbox(entry.polygons)
    _draw_tissue_tile(tile, level, col, row, TILE_SIZE, entry.l0_downsamples, dz._l0_offset, bbox)
    buf = io.BytesIO()
    tile.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


# ── Heatmap Overlay ──────────────────────────────────────────────────────

@app.get("/api/slides/{slide_id}/overlay/heatmap/dzi")
def get_heatmap_dzi(slide_id: str):
    entry = _get_slide(slide_id)
    w, h = entry.slide.dimensions
    xml = _dzi_xml(w, h, TILE_SIZE, OVERLAP, "png")
    return Response(xml, media_type="application/xml")


@app.get("/api/slides/{slide_id}/overlay/heatmap/tile/{level}/{col}_{row}.png")
def get_heatmap_tile(
    slide_id: str, level: int, col: int, row: int,
    model: str = "mock-default", cell: str = "metastasis", vector: str = "false",
):
    entry = _get_slide(slide_id)
    dz = entry.dz
    if level < 0 or level >= dz.level_count:
        raise HTTPException(status_code=404, detail="Invalid level")
    cols_count, rows_count = dz.level_tiles[level]
    if col < 0 or row < 0 or col >= cols_count or row >= rows_count:
        raise HTTPException(status_code=404, detail="Invalid tile address")
    _, z_size = dz._get_tile_info(level, (col, row))
    tile = Image.new("RGBA", z_size, (0, 0, 0, 0))
    bbox = _polys_bbox(entry.polygons)
    seed = _model_seed(model)
    model_offset = (HEATMAP_OFFSET[0] + (seed % 1024), HEATMAP_OFFSET[1])
    draw_vector = vector.lower() == "true"

    _draw_heatmap_tile(
        tile, level, col, row, TILE_SIZE, entry.l0_downsamples, dz._l0_offset,
        HEATMAP_CELL_SIZE, seed, palette=cell, bbox=bbox,
        field_offset=model_offset, draw_vector=draw_vector,
        tissue_check=entry.is_tissue,
    )
    buf = io.BytesIO()
    tile.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


# ── Inference Endpoints ──────────────────────────────────────────────────

@app.post("/api/slides/{slide_id}/inference/start")
def start_inference(slide_id: str, req: InferenceStartRequest):
    entry = _get_slide(slide_id)
    job = inference_manager.start_inference(
        slide_entry=entry,
        model_id=req.model_id,
        roi=req.roi,
        device=req.device,
    )
    return job.to_dict()


@app.get("/api/slides/{slide_id}/inference/status/{job_id}")
def get_inference_status(slide_id: str, job_id: str):
    job = inference_manager.get_job(job_id)
    if not job or job.slide_id != slide_id:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.post("/api/slides/{slide_id}/inference/cancel/{job_id}")
def cancel_inference(slide_id: str, job_id: str):
    success = inference_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel job")
    return {"status": "cancelled", "job_id": job_id}


@app.get("/api/slides/{slide_id}/inference/results/{job_id}")
def get_inference_results(slide_id: str, job_id: str):
    job = inference_manager.get_job(job_id)
    if not job or job.slide_id != slide_id:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Job not completed (status: {job.status.value})")
    results = job.get_results()
    if not results:
        raise HTTPException(status_code=404, detail="Results not found")
    return results


@app.get("/api/slides/{slide_id}/inference/jobs")
def list_inference_jobs(slide_id: str):
    _get_slide(slide_id)  # validate slide exists
    jobs = inference_manager.get_jobs_for_slide(slide_id)
    return [j.to_dict() for j in jobs]


# ── Inference Overlay Tiles ──────────────────────────────────────────────

def _draw_inference_overlay_tile(
    tile, lvl, col, row, tile_size, l0_downsamples, offset,
    nuclei_data, filter_type=None, roi_offset=(0, 0),
):
    """Render nuclei from inference results onto a transparent tile.

    Args:
        tile: RGBA PIL Image to draw on
        lvl, col, row: tile position
        tile_size: tile size in pixels
        l0_downsamples: tuple of downsample factors per DZ level
        offset: DZ level-0 offset (l0_offset)
        nuclei_data: dict of nuclei from inference JSON
        filter_type: if set, only render nuclei of this type name
        roi_offset: (x, y) offset of the ROI in level-0 coordinates
    """
    draw = ImageDraw.Draw(tile, "RGBA")
    down = l0_downsamples[lvl]
    origin_x = col * tile_size
    origin_y = row * tile_size

    tile_l0_x0 = origin_x * down + offset[0]
    tile_l0_y0 = origin_y * down + offset[1]
    tile_l0_x1 = tile_l0_x0 + tile.width * down
    tile_l0_y1 = tile_l0_y0 + tile.height * down

    roi_x, roi_y = roi_offset

    # Adaptive sizing based on zoom level
    # At high zoom (low down), draw detailed dots
    # At low zoom (high down), smaller dots to prevent overlapping blob
    dot_radius = max(1, min(4, int(4 / max(1, down ** 0.2))))
    contour_width = max(1, int(2 / max(1, down ** 0.3)))

    count_drawn = 0

    for nuc_id, nuc in nuclei_data.items():
        # Filter by type name if specified
        if filter_type and nuc.get("type_name", "") != filter_type:
            continue

        # Get centroid in level-0 coordinates
        centroid = nuc.get("centroid")
        if not centroid:
            continue
        cx_l0 = centroid[0] + roi_x
        cy_l0 = centroid[1] + roi_y

        # Quick bounding check — skip if nucleus centroid is far from this tile
        # Use a generous margin to catch nuclei whose dots extend into the tile
        margin = dot_radius * down * 2
        if cx_l0 < tile_l0_x0 - margin or cy_l0 < tile_l0_y0 - margin:
            continue
        if cx_l0 > tile_l0_x1 + margin or cy_l0 > tile_l0_y1 + margin:
            continue

        # Get color by type
        type_name = nuc.get("type_name", "nolabe")
        base_color = HOVERNET_TYPE_COLORS.get(type_name, (255, 255, 255))

        # Convert centroid to tile-local pixel coordinates
        dot_x = (cx_l0 - offset[0]) / down - origin_x
        dot_y = (cy_l0 - offset[1]) / down - origin_y

        # Draw contour if available AND we're zoomed in enough (down <= 64)
        contour = nuc.get("contour")
        if contour and len(contour) >= 3 and down <= 64:
            coords = [
                (
                    (pt[0] + roi_x - offset[0]) / down - origin_x,
                    (pt[1] + roi_y - offset[1]) / down - origin_y,
                )
                for pt in contour
            ]
            fill_color = base_color + (50,)
            outline_color = base_color + (180,)
            try:
                draw.polygon(coords, fill=fill_color, outline=outline_color)
            except Exception:
                pass

        # Draw centroid dot (always visible)
        r = dot_radius
        dot_color = base_color + (200,)
        draw.ellipse([dot_x - r, dot_y - r, dot_x + r, dot_y + r], fill=dot_color)
        count_drawn += 1


@app.get("/api/slides/{slide_id}/overlay/inference/dzi")
def get_inference_overlay_dzi(slide_id: str):
    entry = _get_slide(slide_id)
    w, h = entry.slide.dimensions
    xml = _dzi_xml(w, h, TILE_SIZE, OVERLAP, "png")
    return Response(xml, media_type="application/xml")


@app.get("/api/slides/{slide_id}/overlay/inference/tile/{level}/{col}_{row}.png")
def get_inference_overlay_tile(
    slide_id: str, level: int, col: int, row: int,
    job_id: str = "", filter_type: str = "",
):
    entry = _get_slide(slide_id)
    dz = entry.dz
    if level < 0 or level >= dz.level_count:
        raise HTTPException(status_code=404, detail="Invalid level")
    cols_count, rows_count = dz.level_tiles[level]
    if col < 0 or row < 0 or col >= cols_count or row >= rows_count:
        raise HTTPException(status_code=404, detail="Invalid tile address")

    _, z_size = dz._get_tile_info(level, (col, row))
    tile = Image.new("RGBA", z_size, (0, 0, 0, 0))

    # Find the latest completed job for this slide, or use specified job_id
    job = None
    if job_id:
        job = inference_manager.get_job(job_id)
    else:
        job = inference_manager.get_latest_completed_job(slide_id)

    if job and job.status == JobStatus.COMPLETED:
        results = job.get_results()
        if results and "nuclei" in results:
            # Determine ROI offset
            roi_offset = (0, 0)
            if job.roi:
                roi_offset = (int(job.roi.get("x", 0)), int(job.roi.get("y", 0)))

            ft = filter_type if filter_type else None
            _draw_inference_overlay_tile(
                tile, level, col, row, TILE_SIZE, entry.l0_downsamples,
                dz._l0_offset, results["nuclei"],
                filter_type=ft, roi_offset=roi_offset,
            )

    buf = io.BytesIO()
    tile.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


# ── Density Heatmap Overlay ──────────────────────────────────────────────

# Cold-to-hot color ramp for density heatmap
DENSITY_COLORMAP = [
    (0.0, (30, 30, 120)),     # dark blue
    (0.2, (50, 100, 200)),    # blue
    (0.4, (30, 180, 180)),    # cyan
    (0.6, (100, 220, 100)),   # green
    (0.8, (255, 220, 50)),    # yellow
    (1.0, (255, 60, 30)),     # red
]


def _density_color(t):
    """Map t in [0,1] to an RGBA color using the density colormap."""
    t = max(0.0, min(1.0, t))
    for i in range(1, len(DENSITY_COLORMAP)):
        t0, c0 = DENSITY_COLORMAP[i - 1]
        t1, c1 = DENSITY_COLORMAP[i]
        if t <= t1:
            f = (t - t0) / max(0.001, t1 - t0)
            r = int(c0[0] + f * (c1[0] - c0[0]))
            g = int(c0[1] + f * (c1[1] - c0[1]))
            b = int(c0[2] + f * (c1[2] - c0[2]))
            alpha = int(40 + 140 * t)  # more opaque where denser
            return (r, g, b, alpha)
    c = DENSITY_COLORMAP[-1][1]
    return (c[0], c[1], c[2], 180)


def _build_density_grid(nuclei_data, cell_type, roi, grid_spacing=64):
    """Build a smoothed density grid counting nuclei per grid cell.

    Returns: (grid dict, max_count, grid_spacing)
    Grid keys are (gx, gy) in level-0 coordinates divided by spacing.
    """
    rx, ry = roi.get("x", 0), roi.get("y", 0)
    raw_grid = {}
    for nuc_id, nuc in nuclei_data.items():
        if cell_type and nuc.get("type_name", "") != cell_type:
            continue
        centroid = nuc.get("centroid")
        if not centroid:
            continue
        cx = centroid[0] + rx
        cy = centroid[1] + ry
        gx = int(cx // grid_spacing)
        gy = int(cy // grid_spacing)
        raw_grid[(gx, gy)] = raw_grid.get((gx, gy), 0) + 1

    if not raw_grid:
        return {}, 0, grid_spacing

    kernel = {
        (0, 0): 1.00,
        (-1, 0): 0.60, (1, 0): 0.60, (0, -1): 0.60, (0, 1): 0.60,
        (-1, -1): 0.25, (-1, 1): 0.25, (1, -1): 0.25, (1, 1): 0.25,
        (-2, 0): 0.10, (2, 0): 0.10, (0, -2): 0.10, (0, 2): 0.10,
    }
    smooth_grid = {}
    for (gx, gy), count in raw_grid.items():
        for (dx, dy), weight in kernel.items():
            key = (gx + dx, gy + dy)
            smooth_grid[key] = smooth_grid.get(key, 0.0) + count * weight

    max_count = max(smooth_grid.values()) if smooth_grid else 0
    return smooth_grid, max_count, grid_spacing


def _draw_density_tile(tile, lvl, col, row, tile_size, l0_downsamples, offset,
                       density_grid, max_count, grid_spacing, roi, draw_vector=False):
    """Render density heatmap onto a tile using pre-computed density grid."""
    if max_count == 0:
        return

    draw = ImageDraw.Draw(tile, "RGBA")
    down = l0_downsamples[lvl]
    origin_x = col * tile_size
    origin_y = row * tile_size

    rx = roi.get("x", 0)
    ry = roi.get("y", 0)
    rw = roi.get("width", 0)
    rh = roi.get("height", 0)

    # For each pixel in the tile, determine density and color it
    # We sample at a coarser resolution for performance
    sample_step = max(1, int(down ** 0.3))  # coarser stepping when zoomed out

    for py in range(0, tile.height, sample_step):
        l0_y = (origin_y + py) * down + offset[1]
        if l0_y < ry or l0_y > ry + rh:
            continue
        for px in range(0, tile.width, sample_step):
            l0_x = (origin_x + px) * down + offset[0]
            if l0_x < rx or l0_x > rx + rw:
                continue

            gx = int(l0_x // grid_spacing)
            gy = int(l0_y // grid_spacing)
            count = density_grid.get((gx, gy), 0)

            t = min(1.0, count / max(1, max_count))
            if t < 0.01:
                continue

            color = _density_color(t)
            # Fill the sample block
            for fy in range(sample_step):
                for fx in range(sample_step):
                    if py + fy < tile.height and px + fx < tile.width:
                        try:
                            tile.putpixel((px + fx, py + fy), color)
                        except Exception:
                            pass

    # Draw vector flow arrows showing density gradient direction
    if draw_vector and tile_size >= 128:
        step = 44
        max_arrow_len = 24
        min_arrow_len = 10

        for y in range(step // 2, tile.height, step):
            l0_y = (origin_y + y) * down + offset[1]
            if l0_y < ry or l0_y > ry + rh:
                continue
            for x in range(step // 2, tile.width, step):
                l0_x = (origin_x + x) * down + offset[0]
                if l0_x < rx or l0_x > rx + rw:
                    continue

                gx = int(l0_x // grid_spacing)
                gy = int(l0_y // grid_spacing)

                # Move uphill on the smoothed density field toward denser regions.
                c_right = density_grid.get((gx + 1, gy), 0.0)
                c_left = density_grid.get((gx - 1, gy), 0.0)
                c_down = density_grid.get((gx, gy + 1), 0.0)
                c_up = density_grid.get((gx, gy - 1), 0.0)
                c_here = density_grid.get((gx, gy), 0.0)

                ddx = (c_right - c_left) / max(1, max_count)
                ddy = (c_down - c_up) / max(1, max_count)
                mag = math.hypot(ddx, ddy)
                if c_here <= 0 or mag < 0.015:
                    continue

                norm_mag = min(1.0, mag / 0.14)
                ddx /= mag
                ddy /= mag
                arrow_len = min_arrow_len + (max_arrow_len - min_arrow_len) * norm_mag
                alpha = int(170 + 85 * norm_mag)

                end_x = x + ddx * arrow_len
                end_y = y + ddy * arrow_len

                shadow = (0, 0, 0, int(alpha * 0.90))
                color = (255, 255, 255, alpha)

                draw.line([(x + 2, y + 2), (end_x + 2, end_y + 2)], fill=shadow, width=7)
                draw.line([(x, y), (end_x, end_y)], fill=color, width=4)

                angle = math.atan2(ddy, ddx)
                head_len = 8 + 4 * norm_mag
                a1 = angle + math.pi * 0.84
                a2 = angle - math.pi * 0.84
                hx1 = end_x + math.cos(a1) * head_len
                hy1 = end_y + math.sin(a1) * head_len
                hx2 = end_x + math.cos(a2) * head_len
                hy2 = end_y + math.sin(a2) * head_len
                draw.polygon([(end_x + 2, end_y + 2), (hx1 + 2, hy1 + 2), (hx2 + 2, hy2 + 2)], fill=shadow)
                draw.polygon([(end_x, end_y), (hx1, hy1), (hx2, hy2)], fill=color)


# Density grid cache: keyed by (job_id, cell_type)
_density_cache = {}


@app.get("/api/slides/{slide_id}/overlay/density/dzi")
def get_density_dzi(slide_id: str):
    entry = _get_slide(slide_id)
    w, h = entry.slide.dimensions
    xml = _dzi_xml(w, h, TILE_SIZE, OVERLAP, "png")
    return Response(xml, media_type="application/xml")


@app.get("/api/slides/{slide_id}/overlay/density/tile/{level}/{col}_{row}.png")
def get_density_tile(
    slide_id: str, level: int, col: int, row: int,
    job_id: str = "", cell_type: str = "", vector: str = "false",
):
    entry = _get_slide(slide_id)
    dz = entry.dz
    if level < 0 or level >= dz.level_count:
        raise HTTPException(status_code=404, detail="Invalid level")
    cols_count, rows_count = dz.level_tiles[level]
    if col < 0 or row < 0 or col >= cols_count or row >= rows_count:
        raise HTTPException(status_code=404, detail="Invalid tile address")

    _, z_size = dz._get_tile_info(level, (col, row))
    tile = Image.new("RGBA", z_size, (0, 0, 0, 0))

    # Find the job
    job = None
    if job_id:
        job = inference_manager.get_job(job_id)
    else:
        job = inference_manager.get_latest_completed_job(slide_id)

    if job and job.status == JobStatus.COMPLETED:
        results = job.get_results()
        if results and "nuclei" in results:
            roi = job.roi or {"x": 0, "y": 0, "width": 0, "height": 0}
            cache_key = (job.job_id, cell_type)

            if cache_key not in _density_cache:
                grid, max_count, spacing = _build_density_grid(
                    results["nuclei"], cell_type if cell_type else None, roi
                )
                _density_cache[cache_key] = (grid, max_count, spacing)
            else:
                grid, max_count, spacing = _density_cache[cache_key]

            _draw_density_tile(
                tile, level, col, row, TILE_SIZE, entry.l0_downsamples,
                dz._l0_offset, grid, max_count, spacing, roi,
                draw_vector=(vector.lower() == "true"),
            )

    buf = io.BytesIO()
    tile.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


# ── Legacy Compatibility (redirect old endpoints) ────────────────────────

@app.get("/api/info")
def legacy_info():
    """Backward-compatible info endpoint — returns data for first available slide."""
    if not _slide_registry:
        raise HTTPException(status_code=404, detail="No slides available")
    first = next(iter(_slide_registry.values()))
    props = first.get_properties()
    props["slide"] = str(first.path)
    props["levels"] = first.dz.level_count
    props["overlap"] = OVERLAP
    props["limit_bounds"] = LIMIT_BOUNDS
    props["annotations"] = len(first.polygons)
    props["models"] = MODELS
    props["cell_classes"] = [
        {"id": k, "label": k.replace("_", " ").title()}
        for k in HEATMAP_PALETTES.keys()
    ]
    return props


@app.get("/api/dzi")
def legacy_dzi():
    if not _slide_registry:
        raise HTTPException(status_code=404, detail="No slides available")
    first = next(iter(_slide_registry.values()))
    xml = first.dz.get_dzi("jpeg")
    return Response(xml, media_type="application/xml")


@app.get("/api/tile/{level}/{col}_{row}.jpeg")
def legacy_tile(level: int, col: int, row: int):
    if not _slide_registry:
        raise HTTPException(status_code=404, detail="No slides available")
    first_id = next(iter(_slide_registry.keys()))
    return get_slide_tile(first_id, level, col, row)
