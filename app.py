# -*- coding: utf-8 -*-
"""
Zapfan Smart Cashier — Streamlit App
Economy Rice (Nasi Campur) AI-Powered Pricing System
Powered by Module 4: Smart Checkout Inference System
Models: YOLO | RT-DETR | Faster R-CNN
"""

import streamlit as st
import cv2
import numpy as np
import os
import time
import io
import csv
import copy
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Zapfan Smart Cashier",
    page_icon="🍚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Custom CSS — Google Material Design inspired
# ==============================================================================
st.markdown("""
<style>
/* ===== Google Fonts ===== */
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@300;400;500;700&display=swap');

/* ===== Global ===== */
html, body, [class*="css"] {
    font-family: 'Roboto', 'Google Sans', Arial, sans-serif;
    color: #202124;
}
.main .block-container { max-width: 1200px; padding-top: 1rem; }

/* ===== Sidebar — clean light Google style ===== */
section[data-testid="stSidebar"] {
    background: #f8f9fa !important;
    border-right: 1px solid #dadce0;
}
section[data-testid="stSidebar"] * {
    color: #3c4043 !important;
}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: #ffffff !important;
    border: 1px solid #dadce0 !important;
    color: #202124 !important;
}
section[data-testid="stSidebar"] .stCheckbox label span {
    color: #3c4043 !important;
}

/* ===== Force light main area ===== */
.main, .stApp, [data-testid="stAppViewContainer"] {
    background-color: #ffffff !important;
    color: #202124 !important;
}
.stApp header[data-testid="stHeader"] {
    background-color: #ffffff !important;
}

/* ===== Google-style top bar ===== */
.g-topbar {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 16px 0 12px 0;
    border-bottom: 1px solid #dadce0;
    margin-bottom: 24px;
}
.g-topbar .g-logo {
    font-family: 'Google Sans', 'Roboto', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #202124;
    display: flex;
    align-items: center;
    gap: 10px;
}
.g-topbar .g-logo .g-dot { display: inline-flex; gap: 3px; }
.g-topbar .g-logo .g-dot span {
    width: 8px; height: 8px; border-radius: 50%; display: inline-block;
}
.g-dot .d1 { background: #4285f4; }
.g-dot .d2 { background: #ea4335; }
.g-dot .d3 { background: #fbbc04; }
.g-dot .d4 { background: #34a853; }
.g-topbar .g-subtitle {
    font-size: 0.88rem;
    color: #5f6368;
    font-weight: 400;
}

/* ===== Material Card ===== */
.m-card {
    background: #ffffff;
    border: 1px solid #dadce0;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 2px rgba(60,64,67,0.1), 0 1px 3px rgba(60,64,67,0.08);
}
.m-card:hover {
    box-shadow: 0 1px 3px rgba(60,64,67,0.15), 0 4px 8px rgba(60,64,67,0.1);
}
.m-card-flat {
    background: #ffffff;
    border: 1px solid #dadce0;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

/* ===== Sidebar price items ===== */
.g-price-item {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 10px 14px;
    border-radius: 8px;
    margin-bottom: 4px;
    transition: background 0.15s;
}
.g-price-item:hover { background: #e8f0fe; }
.g-price-item .g-icon {
    width: 40px; height: 40px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.25rem;
}
.g-price-item .g-icon.meat  { background: #fce8e6; }
.g-price-item .g-icon.rice  { background: #e6f4ea; }
.g-price-item .g-icon.vege  { background: #fef7e0; }
.g-price-item .g-text .g-name {
    font-weight: 500; font-size: 0.9rem; color: #202124;
}
.g-price-item .g-text .g-val {
    font-size: 0.8rem; color: #5f6368;
}

/* ===== Stat chips (Google-style metric row) ===== */
.g-stats {
    display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0;
}
.g-chip {
    background: #e8f0fe;
    border-radius: 100px;
    padding: 10px 24px;
    display: flex; align-items: center; gap: 8px;
    flex: 1; min-width: 120px;
    justify-content: center;
}
.g-chip .g-chip-num {
    font-family: 'Google Sans', sans-serif;
    font-size: 1.2rem; font-weight: 700; color: #1a73e8;
}
.g-chip .g-chip-label {
    font-size: 0.8rem; color: #5f6368; font-weight: 400;
}

/* ===== Receipt card ===== */
.receipt-card {
    background: #fff;
    border: 1px solid #dadce0;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 2px rgba(60,64,67,0.1), 0 1px 3px rgba(60,64,67,0.08);
}
.receipt-header {
    font-family: 'Google Sans', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #202124;
    margin-bottom: 12px;
    padding-bottom: 10px;
    border-bottom: 1px solid #e8eaed;
    display: flex;
    align-items: center;
    gap: 8px;
}
.receipt-header .model-tag {
    font-weight: 400; font-size: 0.78rem;
    color: #5f6368; background: #f1f3f4;
    padding: 2px 10px; border-radius: 100px;
}
.receipt-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    font-size: 0.9rem;
    color: #3c4043;
    border-bottom: 1px solid #f1f3f4;
}
.receipt-item:last-of-type { border-bottom: none; }
.receipt-item .label {
    font-weight: 500;
    display: flex; align-items: center; gap: 6px;
}
.receipt-item .tag {
    font-size: 0.7rem;
    background: #e8f0fe;
    color: #1a73e8;
    padding: 2px 10px;
    border-radius: 100px;
    font-weight: 500;
    letter-spacing: 0.3px;
}
.receipt-item .price {
    font-family: 'Google Sans', sans-serif;
    font-weight: 500;
    color: #202124;
}
.receipt-total {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 2px solid #202124;
    display: flex;
    justify-content: space-between;
    font-family: 'Google Sans', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #202124;
}
.receipt-grand {
    margin-top: 8px;
    padding-top: 12px;
    border-top: 2px solid #1a73e8;
    display: flex;
    justify-content: space-between;
    font-family: 'Google Sans', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #1a73e8;
}

/* ===== Plate badge ===== */
.plate-badge {
    display: inline-block;
    background: #1a73e8;
    color: #fff !important;
    font-family: 'Google Sans', sans-serif;
    font-weight: 500;
    padding: 4px 16px;
    border-radius: 100px;
    font-size: 0.8rem;
    margin: 10px 0 6px 0;
    letter-spacing: 0.2px;
}

/* ===== Empty state ===== */
.g-empty {
    text-align: center;
    padding: 4rem 1rem 3rem 1rem;
}
.g-empty .g-empty-icon {
    width: 80px; height: 80px;
    margin: 0 auto 20px auto;
    background: #e8f0fe;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 2.2rem;
}
.g-empty h3 {
    font-family: 'Google Sans', sans-serif;
    font-weight: 500; color: #202124; margin: 0 0 8px 0;
}
.g-empty p { color: #5f6368; font-size: 0.95rem; margin: 0; }

/* ===== Tabs — Google style ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 0; border-bottom: 2px solid #dadce0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0;
    padding: 10px 24px;
    font-weight: 500;
    font-size: 0.9rem;
    color: #5f6368;
    border-bottom: 3px solid transparent;
    margin-bottom: -2px;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #1a73e8 !important;
    border-bottom: 3px solid #1a73e8;
    font-weight: 500;
}

/* ===== Buttons — Google Material ===== */
.stButton > button[kind="primary"] {
    background: #1a73e8 !important;
    color: white !important;
    border: none !important;
    border-radius: 100px !important;
    font-weight: 500 !important;
    padding: 8px 24px !important;
    font-size: 0.9rem !important;
    box-shadow: 0 1px 2px rgba(60,64,67,0.3) !important;
    transition: box-shadow 0.2s, background 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1765cc !important;
    box-shadow: 0 1px 3px rgba(60,64,67,0.4) !important;
}
.stButton > button:not([kind="primary"]) {
    background: #fff !important;
    color: #1a73e8 !important;
    border: 1px solid #dadce0 !important;
    border-radius: 100px !important;
    font-weight: 500 !important;
    padding: 8px 24px !important;
    font-size: 0.9rem !important;
}
.stButton > button:not([kind="primary"]):hover {
    background: #f8f9fa !important;
    border-color: #1a73e8 !important;
}

/* ===== File uploader ===== */
.stFileUploader > div {
    border: 2px dashed #dadce0 !important;
    border-radius: 12px !important;
}
.stFileUploader > div:hover {
    border-color: #1a73e8 !important;
}

/* ===== Image border ===== */
.stImage > img {
    border-radius: 12px;
    border: 1px solid #dadce0;
}

/* ===== Footer ===== */
.g-footer {
    text-align: center;
    padding: 2rem 0 1rem 0;
    font-size: 0.8rem;
    color: #9aa0a6;
    border-top: 1px solid #e8eaed;
    margin-top: 2rem;
}
.g-footer a { color: #1a73e8; text-decoration: none; }
.g-footer a:hover { text-decoration: underline; }

/* ===== Section label ===== */
.g-section {
    font-family: 'Google Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 500;
    color: #202124;
    margin: 20px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Configuration (Module 4 — Smart Checkout Engine)
# ==============================================================================
CONFIG = {
    # ── Class definitions (must match training order) ─────────────────────
    "classes": ["meat", "plate", "rice", "vege"],

    # ── Pricing (RM) ──────────────────────────────────────────────────────
    "currency": "RM",
    "base_prices": {
        "meat": 4.00,
        "rice": 1.50,
        "vege": 2.00,
        "plate": 0.00,
    },
    "size_multipliers": {
        "S": 0.7,
        "M": 1.0,
        "L": 1.5,
    },

    # ── Size thresholds (bbox area / image area) ─────────────────────────
    "size_thresholds": {
        "small_max":  0.04,   # bbox area < 4 % of image → Small
        "medium_max": 0.10,   # bbox area < 10 % → Medium, >= 10 % → Large
    },

    # ── Detection settings ────────────────────────────────────────────────
    "confidence_threshold": 0.40,
    "iou_threshold":        0.45,
    "imgsz":                640,

    # ── Size reference ─────────────────────────────────────────────────
    # "image": use full image area, "plate": use detected plate area
    "size_reference": "image",

    # ── Visualization (RGB colours for Streamlit) ─────────────────────────
    "colors": {
        "meat":  (255, 0, 0),
        "rice":  (0, 200, 0),
        "vege":  (255, 165, 0),
        "plate": (0, 255, 255),
    },
    "font_scale":    0.7,
    "box_thickness": 2,
}

# Keep an immutable copy for reset and safe defaults
DEFAULT_CONFIG = copy.deepcopy(CONFIG)

# Model file paths
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH   = os.path.join(BASE_DIR, "model_mixed_rice.pt")
RTDETR_PATH = os.path.join(BASE_DIR, "model_rtdetr_mixed_rice.pt")
FRCNN_PATH  = os.path.join(BASE_DIR, "faster_rcnn_mixed_rice.pth")

MODEL_PATHS = {
    "yolo": YOLO_PATH,
    "rtdetr": RTDETR_PATH,
    "frcnn": FRCNN_PATH,
}

# Training results directory
TRAINING_RESULTS_DIR = os.path.join(BASE_DIR, "training_results")

# Optional demo images directory
DEMO_IMAGES_DIR = os.path.join(BASE_DIR, "demo_images")

# Optional comparison images
COMPARISON_IMAGES = [
    os.path.join(BASE_DIR, "yolo vs rtdetr.png"),
    os.path.join(BASE_DIR, "rtdetr vs rcnn.png"),
]

# Map model keys → training results subfolder + display name
MODEL_RESULTS_MAP = {
    "yolo":  {"dir": os.path.join(TRAINING_RESULTS_DIR, "yolo"),  "name": "YOLOv8"},
    "rtdetr": {"dir": os.path.join(TRAINING_RESULTS_DIR, "rtdetr"), "name": "RT-DETR"},
    "frcnn": {"dir": os.path.join(TRAINING_RESULTS_DIR, "frcnn"), "name": "Faster R-CNN"},
}

# Training plot filenames we look for (in priority display order)
TRAINING_PLOTS = [
    ("results.png",            "Training Results",   "Metrics over training epochs (loss, mAP, precision, recall)"),
    ("confusion_matrix.png",   "Confusion Matrix",   "Classification confusion matrix on validation set"),
    ("confusion_matrix_normalized.png", "Confusion Matrix (Normalized)", "Normalized confusion matrix"),
    ("PR_curve.png",           "Precision-Recall Curve", "PR curve for each class"),
    ("F1_curve.png",           "F1 Curve",           "F1 score vs confidence threshold"),
    ("P_curve.png",            "Precision Curve",    "Precision vs confidence threshold"),
    ("R_curve.png",            "Recall Curve",       "Recall vs confidence threshold"),
    ("labels.jpg",             "Label Distribution", "Distribution of labels in training set"),
    ("labels_correlogram.jpg", "Labels Correlogram",  "Correlogram of label positions and sizes"),
]


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class Detection:
    """Single detected item with pricing."""
    class_name:    str
    confidence:    float
    bbox:          Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    area_fraction: float                       # bbox area / image area
    size:          str                         # S / M / L (or "—" for plate)
    price:         float                       # after size multiplier
    cropped_image: Optional[np.ndarray] = None


@dataclass
class CheckoutResult:
    """Full checkout result for one image."""
    model_name:       str
    detections:       List[Detection] = field(default_factory=list)
    total_price:      float = 0.0
    inference_time_ms: float = 0.0
    annotated_image:  Optional[np.ndarray] = None

    @property
    def item_summary(self) -> Dict[str, int]:
        """Count of each detected class."""
        return dict(Counter(d.class_name for d in self.detections))

    @property
    def billable_items(self) -> List[Detection]:
        """Items that have price > 0 (excludes plate)."""
        return [d for d in self.detections if d.price > 0]

    @property
    def plate_detection(self) -> Optional[Detection]:
        """Return the single plate detection, if any."""
        plates = [d for d in self.detections if d.class_name == "plate"]
        return plates[0] if plates else None


# ==============================================================================
# Size Estimator & Pricing
# ==============================================================================

def estimate_size(area_fraction: float) -> str:
    """Estimate portion size from bbox area as fraction of image area."""
    thresholds = CONFIG["size_thresholds"]
    if area_fraction < thresholds["small_max"]:
        return "S"
    elif area_fraction < thresholds["medium_max"]:
        return "M"
    else:
        return "L"


def calculate_price(class_name: str, size: str) -> float:
    """Calculate price for a single item based on class and size."""
    base = CONFIG["base_prices"].get(class_name, 0.0)
    mult = CONFIG["size_multipliers"].get(size, 1.0)
    return round(base * mult, 2)


# ==============================================================================
# NMS Helpers
# ==============================================================================

def _compute_iou(box1, box2) -> float:
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _nms_filter(detections: List[Dict], iou_thresh: float) -> List[Dict]:
    """Class-aware NMS to remove duplicate boxes."""
    if len(detections) <= 1:
        return detections

    by_class: Dict[str, list] = defaultdict(list)
    for d in detections:
        by_class[d["class_name"]].append(d)

    filtered: List[Dict] = []
    for cls_name, dets in by_class.items():
        dets.sort(key=lambda x: x["confidence"], reverse=True)
        keep: List[Dict] = []
        for det in dets:
            if not any(_compute_iou(det["bbox"], k["bbox"]) > iou_thresh for k in keep):
                keep.append(det)
        filtered.extend(keep)
    return filtered


# ==============================================================================
# Detector Wrapper Classes
# ==============================================================================

class YOLODetector:
    """Wrapper for YOLOv8 / RT-DETR (ultralytics API)."""

    def __init__(self, model_path: str, model_type: str = "yolo"):
        from ultralytics import YOLO, RTDETR
        self.model_type = model_type
        if model_type == "rtdetr":
            self.model = RTDETR(model_path)
            self.name = "RT-DETR"
        else:
            self.model = YOLO(model_path)
            self.name = "YOLOv8"

    def predict(self, image: np.ndarray) -> List[Dict]:
        """Return list of raw detection dicts {class_name, confidence, bbox}."""
        results = self.model.predict(
            source=image,
            conf=CONFIG["confidence_threshold"],
            iou=CONFIG["iou_threshold"],
            imgsz=CONFIG["imgsz"],
            verbose=False,
        )
        detections: List[Dict] = []
        classes = CONFIG["classes"]
        if results and len(results) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                if cls_id < len(classes):
                    detections.append({
                        "class_name": classes[cls_id],
                        "confidence": conf,
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    })
        return detections

    def warmup(self):
        if getattr(self, "_warmed", False):
            return
        dummy = np.zeros((CONFIG["imgsz"], CONFIG["imgsz"], 3), dtype=np.uint8)
        try:
            _ = self.predict(dummy)
        except Exception:
            pass
        self._warmed = True


class FasterRCNNDetector:
    """Wrapper for Faster R-CNN (torchvision)."""

    def __init__(self, model_path: str):
        self.name = "Faster R-CNN"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_classes = len(CONFIG["classes"]) + 1  # +1 for background
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> List[Dict]:
        """Return list of raw detection dicts {class_name, confidence, bbox}."""
        img_tensor = (
            torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        )
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        outputs = self.model(img_tensor)[0]

        detections: List[Dict] = []
        classes = CONFIG["classes"]
        for box, score, label in zip(
            outputs["boxes"].cpu().numpy(),
            outputs["scores"].cpu().numpy(),
            outputs["labels"].cpu().numpy(),
        ):
            if score < CONFIG["confidence_threshold"]:
                continue
            if label == 0:  # skip background
                continue
            cls_idx = label - 1
            if cls_idx < len(classes):
                x1, y1, x2, y2 = box.astype(int)
                detections.append({
                    "class_name": classes[cls_idx],
                    "confidence": float(score),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                })

        return _nms_filter(detections, CONFIG["iou_threshold"])

    @torch.no_grad()
    def warmup(self):
        if getattr(self, "_warmed", False):
            return
        dummy = torch.zeros((1, 3, 512, 512), device=self.device)
        try:
            _ = self.model(dummy)
        except Exception:
            pass
        self._warmed = True


def load_detector(model_path: str):
    """Auto-detect model type from filename and return the appropriate detector."""
    if not os.path.exists(model_path):
        return None
    ext = Path(model_path).suffix.lower()
    name_lower = Path(model_path).stem.lower()
    if ext == ".pth":
        return FasterRCNNDetector(model_path)
    elif "rtdetr" in name_lower:
        return YOLODetector(model_path, model_type="rtdetr")
    else:
        return YOLODetector(model_path, model_type="yolo")


# ==============================================================================
# Model Loading (cached — loads once per Streamlit session)
# ==============================================================================

@st.cache_resource
def get_detector(model_key: str, warmup: bool = True):
    """Return a cached detector instance for the given model key."""
    path = MODEL_PATHS.get(model_key)
    if path is None or not os.path.exists(path):
        return None
    detector = load_detector(path)
    if detector is not None and warmup:
        detector.warmup()
    return detector


def _resolve_model_key(model_option: str) -> str:
    """Map UI display name → internal key."""
    if "YOLO" in model_option:
        return "yolo"
    elif "RT-DETR" in model_option:
        return "rtdetr"
    else:
        return "frcnn"


# ==============================================================================
# Checkout Engine (Single-Plate)
# ==============================================================================

def checkout(img_rgb: np.ndarray, detector) -> CheckoutResult:
    """
    Run detection on a single image and compute the checkout.

    Enforces single-plate detection (highest-confidence plate only).
    Size is estimated from bbox area as a fraction of total image area.
    """
    img_h, img_w = img_rgb.shape[:2]
    img_area = img_h * img_w

    # ── Run inference with timing ─────────────────────────────────────────
    t0 = time.time()
    raw_detections = detector.predict(img_rgb)
    inference_ms = (time.time() - t0) * 1000

    # ── Separate plates and food items ────────────────────────────────────
    raw_plates = [d for d in raw_detections if d["class_name"] == "plate"]
    raw_foods  = [d for d in raw_detections if d["class_name"] != "plate"]

    # Keep only the highest-confidence plate
    best_plate = max(raw_plates, key=lambda p: p["confidence"]) if raw_plates else None

    # ── Build Detection objects ───────────────────────────────────────────
    detections: List[Detection] = []

    plate_area = None
    if best_plate:
        x1, y1, x2, y2 = best_plate["bbox"]
        bbox_area = (x2 - x1) * (y2 - y1)
        plate_area = bbox_area if bbox_area > 0 else None
        detections.append(Detection(
            class_name="plate",
            confidence=best_plate["confidence"],
            bbox=best_plate["bbox"],
            area_fraction=bbox_area / img_area,
            size="—",
            price=0.0,
            cropped_image=None,
        ))

    for raw in raw_foods:
        x1, y1, x2, y2 = raw["bbox"]
        bbox_area = (x2 - x1) * (y2 - y1)
        ref_area = img_area
        if CONFIG.get("size_reference") == "plate" and plate_area:
            ref_area = plate_area
        area_frac = bbox_area / ref_area if ref_area > 0 else 0.0

        size  = estimate_size(area_frac)
        price = calculate_price(raw["class_name"], size)

        # Crop with 10 % padding
        pad_x = int((x2 - x1) * 0.10)
        pad_y = int((y2 - y1) * 0.10)
        c_y1, c_y2 = max(0, y1 - pad_y), min(img_h, y2 + pad_y)
        c_x1, c_x2 = max(0, x1 - pad_x), min(img_w, x2 + pad_x)
        crop = (
            img_rgb[c_y1:c_y2, c_x1:c_x2].copy()
            if c_y2 > c_y1 and c_x2 > c_x1
            else None
        )

        detections.append(Detection(
            class_name=raw["class_name"],
            confidence=raw["confidence"],
            bbox=raw["bbox"],
            area_fraction=area_frac,
            size=size,
            price=price,
            cropped_image=crop,
        ))

    # Sort: plate first, then alphabetically, highest confidence first
    detections.sort(key=lambda d: (d.class_name != "plate", d.class_name, -d.confidence))

    subtotal = sum(d.price for d in detections)

    # ── Annotate image ────────────────────────────────────────────────────
    annotated = _draw_detections(img_rgb.copy(), detections)

    return CheckoutResult(
        model_name=detector.name,
        detections=detections,
        total_price=subtotal,
        inference_time_ms=inference_ms,
        annotated_image=annotated,
    )


# ==============================================================================
# Visualization
# ==============================================================================

def _draw_detections(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Draw bounding boxes and labels on the image (RGB)."""
    colors   = CONFIG["colors"]
    currency = CONFIG["currency"]
    font     = cv2.FONT_HERSHEY_SIMPLEX
    scale    = CONFIG["font_scale"]
    thick    = CONFIG["box_thickness"]

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = colors.get(det.class_name, (200, 200, 200))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thick)

        if det.class_name == "plate":
            label = f"Plate {det.confidence:.0%}"
        else:
            label = (
                f"{det.class_name.capitalize()} ({det.size}) "
                f"{det.confidence:.0%} {currency}{det.price:.2f}"
            )

        (tw, th), _ = cv2.getTextSize(label, font, scale, 2)
        cv2.rectangle(
            image,
            (x1, max(y1 - th - 10, 0)),
            (x1 + tw + 6, max(y1, 10)),
            color, -1,
        )
        cv2.putText(
            image, label, (x1 + 3, max(y1 - 5, 15)),
            font, scale, (255, 255, 255), 2,
        )

    return image


def _image_to_png_bytes(image_rgb: np.ndarray) -> bytes:
    """Convert an RGB image array to PNG bytes."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", image_bgr)
    return buffer.tobytes() if success else b""


def _result_to_receipt_text(result: CheckoutResult) -> str:
    """Build a plain-text receipt for download."""
    currency = CONFIG["currency"]
    lines = []
    lines.append("Zapfan Smart Cashier")
    lines.append(f"Model: {result.model_name}")
    lines.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("-")

    grouped: Dict[Tuple, Dict] = defaultdict(lambda: {"count": 0, "unit_price": 0.0})
    for d in result.billable_items:
        key = (d.class_name, d.size)
        grouped[key]["count"] += 1
        grouped[key]["unit_price"] = d.price

    for (cls, size), info in sorted(grouped.items()):
        count = info["count"]
        unit = info["unit_price"]
        line_total = unit * count
        lines.append(f"{cls.capitalize()} ({size}) x{count}: {currency}{line_total:.2f}")

    lines.append("-")
    lines.append(f"Subtotal: {currency}{result.total_price:.2f}")
    return "\n".join(lines)


def _result_to_csv(result: CheckoutResult) -> str:
    """Build a CSV string for receipt export."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["class", "size", "confidence", "unit_price"])
    for d in result.billable_items:
        writer.writerow([
            d.class_name,
            d.size,
            f"{d.confidence:.4f}",
            f"{d.price:.2f}",
        ])
    return output.getvalue()


def _list_demo_images() -> List[str]:
    """Return a sorted list of demo image filenames in demo_images/."""
    if not os.path.isdir(DEMO_IMAGES_DIR):
        return []
    exts = {".jpg", ".jpeg", ".png"}
    files = [
        f for f in os.listdir(DEMO_IMAGES_DIR)
        if Path(f).suffix.lower() in exts
    ]
    return sorted(files)


def _load_demo_image(filename: str) -> Optional[np.ndarray]:
    """Load a demo image by filename from demo_images/."""
    path = os.path.join(DEMO_IMAGES_DIR, filename)
    if not os.path.isfile(path):
        return None
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ==============================================================================
# Streamlit Display — analyse & render results
# ==============================================================================

def analyse_and_display(img_rgb, model_option, container=None, key_prefix=""):
    """Load the chosen model, run checkout engine, and render results."""
    if container is None:
        container = st.container()

    with container:
        # ── Load detector ─────────────────────────────────────────────────
        model_key = _resolve_model_key(model_option)
        with st.spinner(f"Loading {model_option} model..."):
            detector = get_detector(
                model_key,
                warmup=st.session_state.get("warmup_models", True),
            )

        if detector is None:
            st.error(
                f"Model file not found for **{model_option}**! "
                "Make sure the model weights are in the project folder."
            )
            return

        # ── Run checkout ──────────────────────────────────────────────────
        with st.spinner(f"Analyzing your plate with {model_option}..."):
            result = checkout(img_rgb, detector)

        if not result.billable_items:
            st.warning(
                f"[{model_option}] No food items detected. "
                "Try a different model or image."
            )
            return

        currency    = CONFIG["currency"]
        total_items = len(result.billable_items)
        model_label = model_option.split(" (")[0]

        # ── Summary stat chips (Google style) ─────────────────────────────
        stat_html = (
            '<div class="g-stats">'
            '<div class="g-chip">'
            '<span class="g-chip-num">1</span>'
            '<span class="g-chip-label">Plate</span></div>'
            f'<div class="g-chip">'
            f'<span class="g-chip-num">{total_items}</span>'
            f'<span class="g-chip-label">Items</span></div>'
            f'<div class="g-chip">'
            f'<span class="g-chip-num">{currency}{result.total_price:.2f}</span>'
            f'<span class="g-chip-label">Subtotal</span></div>'
            f'<div class="g-chip">'
            f'<span class="g-chip-num">{result.inference_time_ms:.0f}ms</span>'
            f'<span class="g-chip-label">Speed</span></div>'
            '</div>'
        )
        st.markdown(stat_html, unsafe_allow_html=True)

        # ── Result columns ────────────────────────────────────────────────
        col_img, col_receipt = st.columns([3, 2])

        with col_img:
            st.markdown(
                '<div class="g-section">Detection result</div>',
                unsafe_allow_html=True,
            )
            st.image(result.annotated_image, use_container_width=True)

        with col_receipt:
            st.markdown(
                f'<div class="receipt-card"><div class="receipt-header">'
                f'Receipt <span class="model-tag">{model_label}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            # Group billable items by (class, size) for a cleaner receipt
            grouped: Dict[Tuple, Dict] = defaultdict(
                lambda: {"count": 0, "unit_price": 0.0}
            )
            for d in result.billable_items:
                key = (d.class_name, d.size)
                grouped[key]["count"] += 1
                grouped[key]["unit_price"] = d.price

            for (cls, size), info in sorted(grouped.items()):
                item_emoji = {"meat": "🥩", "rice": "🍚", "vege": "🥬"}.get(cls, "🍽️")
                count = info["count"]
                unit  = info["unit_price"]
                line_total = unit * count
                qty_label = f" ×{count}" if count > 1 else ""

                st.markdown(
                    f'<div class="receipt-item">'
                    f'<span class="label">{item_emoji} {cls.capitalize()}{qty_label}'
                    f'<span class="tag">{size}</span></span>'
                    f'<span class="price">{currency}{line_total:.2f}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Plate row
            if result.plate_detection:
                st.markdown(
                    '<div class="receipt-item">'
                    '<span class="label">🍽️ Plate detected</span>'
                    '<span class="price">—</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="receipt-total">'
                f'<span>Subtotal</span><span>{currency}{result.total_price:.2f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Item count summary
            food_summary = {
                k: v for k, v in result.item_summary.items() if k != "plate"
            }
            if food_summary:
                summary_str = " · ".join(
                    f"{v}× {k}" for k, v in sorted(food_summary.items())
                )
                st.caption(f"Items: {summary_str}")

            # ── Downloads ───────────────────────────────────────────────
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            with dl_col1:
                st.download_button(
                    "Download annotated image",
                    data=_image_to_png_bytes(result.annotated_image),
                    file_name="zapfan_annotated.png",
                    mime="image/png",
                    key=f"dl_img_{key_prefix}",
                    use_container_width=True,
                )
            with dl_col2:
                st.download_button(
                    "Download receipt (txt)",
                    data=_result_to_receipt_text(result),
                    file_name="zapfan_receipt.txt",
                    mime="text/plain",
                    key=f"dl_txt_{key_prefix}",
                    use_container_width=True,
                )
            with dl_col3:
                st.download_button(
                    "Download receipt (csv)",
                    data=_result_to_csv(result),
                    file_name="zapfan_receipt.csv",
                    mime="text/csv",
                    key=f"dl_csv_{key_prefix}",
                    use_container_width=True,
                )

        # ── Cropped items ─────────────────────────────────────────────────
        cropped = [
            (d.class_name, d.confidence, d.cropped_image)
            for d in result.detections
            if d.cropped_image is not None
        ]
        if cropped:
            with st.expander(f"Detected items ({len(cropped)})", expanded=False):
                crop_cols = st.columns(min(len(cropped), 4))
                for idx, (c_name, c_score, c_img) in enumerate(cropped):
                    with crop_cols[idx % len(crop_cols)]:
                        st.image(
                            c_img,
                            caption=f"{c_name.capitalize()} ({c_score:.0%})",
                            use_container_width=True,
                        )


# ==============================================================================
# Runtime Configuration Helpers
# ==============================================================================

def _reset_sidebar_settings():
    """Reset sidebar controls to default config values."""
    st.session_state["conf_slider"] = DEFAULT_CONFIG["confidence_threshold"]
    st.session_state["iou_slider"] = DEFAULT_CONFIG["iou_threshold"]
    st.session_state["size_small_max"] = DEFAULT_CONFIG["size_thresholds"]["small_max"]
    st.session_state["size_medium_max"] = DEFAULT_CONFIG["size_thresholds"]["medium_max"]
    st.session_state["price_meat"] = DEFAULT_CONFIG["base_prices"]["meat"]
    st.session_state["price_rice"] = DEFAULT_CONFIG["base_prices"]["rice"]
    st.session_state["price_vege"] = DEFAULT_CONFIG["base_prices"]["vege"]
    st.session_state["mult_s"] = DEFAULT_CONFIG["size_multipliers"]["S"]
    st.session_state["mult_m"] = DEFAULT_CONFIG["size_multipliers"]["M"]
    st.session_state["mult_l"] = DEFAULT_CONFIG["size_multipliers"]["L"]
    st.session_state["size_reference"] = DEFAULT_CONFIG["size_reference"]


def _apply_runtime_config(enabled: bool):
    """Apply sidebar settings into CONFIG when customization is enabled."""
    if not enabled:
        CONFIG.update(copy.deepcopy(DEFAULT_CONFIG))
        return

    CONFIG["confidence_threshold"] = st.session_state.get(
        "conf_slider", DEFAULT_CONFIG["confidence_threshold"]
    )
    CONFIG["iou_threshold"] = st.session_state.get(
        "iou_slider", DEFAULT_CONFIG["iou_threshold"]
    )
    CONFIG["size_thresholds"]["small_max"] = st.session_state.get(
        "size_small_max", DEFAULT_CONFIG["size_thresholds"]["small_max"]
    )
    CONFIG["size_thresholds"]["medium_max"] = st.session_state.get(
        "size_medium_max", DEFAULT_CONFIG["size_thresholds"]["medium_max"]
    )
    CONFIG["base_prices"]["meat"] = st.session_state.get(
        "price_meat", DEFAULT_CONFIG["base_prices"]["meat"]
    )
    CONFIG["base_prices"]["rice"] = st.session_state.get(
        "price_rice", DEFAULT_CONFIG["base_prices"]["rice"]
    )
    CONFIG["base_prices"]["vege"] = st.session_state.get(
        "price_vege", DEFAULT_CONFIG["base_prices"]["vege"]
    )
    CONFIG["size_multipliers"]["S"] = st.session_state.get(
        "mult_s", DEFAULT_CONFIG["size_multipliers"]["S"]
    )
    CONFIG["size_multipliers"]["M"] = st.session_state.get(
        "mult_m", DEFAULT_CONFIG["size_multipliers"]["M"]
    )
    CONFIG["size_multipliers"]["L"] = st.session_state.get(
        "mult_l", DEFAULT_CONFIG["size_multipliers"]["L"]
    )
    CONFIG["size_reference"] = st.session_state.get(
        "size_reference", DEFAULT_CONFIG["size_reference"]
    )


# ==============================================================================
# Session State Initialisation
# ==============================================================================
if "image_history" not in st.session_state:
    st.session_state.image_history = []   # list of (filename, img_rgb) tuples
if "active_index" not in st.session_state:
    st.session_state.active_index = 0
if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None
if "new_upload_pending" not in st.session_state:
    st.session_state.new_upload_pending = False
if "auto_analyse" not in st.session_state:
    st.session_state.auto_analyse = False
if "upload_counter" not in st.session_state:
    st.session_state.upload_counter = 0
if "last_camera_id" not in st.session_state:
    st.session_state.last_camera_id = None
if "camera_counter" not in st.session_state:
    st.session_state.camera_counter = 0
if "warmup_models" not in st.session_state:
    st.session_state.warmup_models = True

# ==============================================================================
# Streamlit UI
# ==============================================================================

# --- Google-style top bar ---
st.markdown("""
<div class="g-topbar">
    <div class="g-logo">
        <span class="g-dot"><span class="d1"></span><span class="d2"></span><span class="d3"></span><span class="d4"></span></span>
        Zapfan Smart Cashier
    </div>
    <div class="g-subtitle">Economy Rice &middot; AI-Powered Detection &amp; Pricing</div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
MODEL_OPTIONS = ["YOLO (Fast)", "RT-DETR (Transformer)", "Faster R-CNN (Accurate)"]

with st.sidebar:
    st.markdown('<p style="font-family:Google Sans,sans-serif;font-size:1.1rem;font-weight:700;color:#202124;margin-bottom:4px;">Settings</p>', unsafe_allow_html=True)

    model_status = {k: os.path.exists(v) for k, v in MODEL_PATHS.items()}
    st.markdown(
        """
        <div class="m-card-flat" style="padding:12px 16px;">
            <p style="margin:0 0 6px 0;font-size:0.75rem;color:#9aa0a6;font-weight:600;letter-spacing:0.4px;">MODEL STATUS</p>
        """,
        unsafe_allow_html=True,
    )
    for key, label in [("yolo", "YOLOv8"), ("rtdetr", "RT-DETR"), ("frcnn", "Faster R-CNN")]:
        status = "Available" if model_status.get(key) else "Missing"
        color = "#188038" if model_status.get(key) else "#d93025"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;font-size:0.85rem;'>"
            f"<span>{label}</span><span style='color:{color};font-weight:600'>{status}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    model_option = st.selectbox("AI Model", options=MODEL_OPTIONS, index=0)
    compare_all = st.checkbox("Compare all models side-by-side")

    st.markdown("---")
    st.markdown(
        '<p style="font-family:Google Sans,sans-serif;font-size:0.95rem;font-weight:500;color:#202124;">Performance</p>',
        unsafe_allow_html=True,
    )
    st.toggle("Warm up models on load", key="warmup_models")
    st.caption("Improves first-run speed by pre-loading model kernels.")

    customize_settings = st.toggle("Customize pricing and detection", value=False, key="customize_settings")
    if customize_settings:
        with st.expander("Quick settings", expanded=True):
            if st.button("Reset settings", use_container_width=True):
                _reset_sidebar_settings()
                st.rerun()

            st.markdown("<div class='g-section'>Detection</div>", unsafe_allow_html=True)
            st.slider(
                "Confidence threshold",
                min_value=0.10,
                max_value=0.90,
                value=DEFAULT_CONFIG["confidence_threshold"],
                step=0.01,
                key="conf_slider",
            )
            st.slider(
                "IoU threshold",
                min_value=0.10,
                max_value=0.90,
                value=DEFAULT_CONFIG["iou_threshold"],
                step=0.01,
                key="iou_slider",
            )
            st.selectbox(
                "Size reference",
                options=["image", "plate"],
                index=0 if DEFAULT_CONFIG["size_reference"] == "image" else 1,
                format_func=lambda x: "Full image" if x == "image" else "Detected plate",
                key="size_reference",
            )

            st.markdown("<div class='g-section'>Portion sizing</div>", unsafe_allow_html=True)
            small_max = st.slider(
                "Small max (area ratio)",
                min_value=0.01,
                max_value=0.20,
                value=DEFAULT_CONFIG["size_thresholds"]["small_max"],
                step=0.01,
                key="size_small_max",
            )
            medium_max = st.slider(
                "Medium max (area ratio)",
                min_value=0.02,
                max_value=0.30,
                value=DEFAULT_CONFIG["size_thresholds"]["medium_max"],
                step=0.01,
                key="size_medium_max",
            )
            if small_max >= medium_max:
                st.warning("Medium threshold must be larger than Small. Auto-adjusted.")
                st.session_state["size_medium_max"] = min(small_max + 0.01, 0.30)

            st.markdown("<div class='g-section'>Pricing</div>", unsafe_allow_html=True)
            st.number_input(
                "Meat base price (RM)",
                min_value=0.0,
                max_value=20.0,
                value=DEFAULT_CONFIG["base_prices"]["meat"],
                step=0.1,
                key="price_meat",
            )
            st.number_input(
                "Rice base price (RM)",
                min_value=0.0,
                max_value=10.0,
                value=DEFAULT_CONFIG["base_prices"]["rice"],
                step=0.1,
                key="price_rice",
            )
            st.number_input(
                "Vegetable base price (RM)",
                min_value=0.0,
                max_value=10.0,
                value=DEFAULT_CONFIG["base_prices"]["vege"],
                step=0.1,
                key="price_vege",
            )
            st.number_input(
                "Small multiplier",
                min_value=0.1,
                max_value=2.0,
                value=DEFAULT_CONFIG["size_multipliers"]["S"],
                step=0.05,
                key="mult_s",
            )
            st.number_input(
                "Medium multiplier",
                min_value=0.1,
                max_value=2.0,
                value=DEFAULT_CONFIG["size_multipliers"]["M"],
                step=0.05,
                key="mult_m",
            )
            st.number_input(
                "Large multiplier",
                min_value=0.1,
                max_value=2.5,
                value=DEFAULT_CONFIG["size_multipliers"]["L"],
                step=0.05,
                key="mult_l",
            )

    _apply_runtime_config(customize_settings)

    st.markdown("---")
    
    # Project info card
    st.markdown("""
    <div class="m-card-flat">
        <p style="margin:0;font-size:0.75rem;color:#9aa0a6;font-weight:500;letter-spacing:0.5px;">ACADEMIC PROJECT</p>
        <p style="margin:4px 0 0 0;font-size:0.9rem;color:#202124;font-weight:500;">Mixed Rice Price Prediction</p>
        <p style="margin:2px 0 0 0;font-size:0.75rem;color:#5f6368;">BMCS2203 · Tutorial Group 5</p>
    </div>
    """, unsafe_allow_html=True)

    # Google-style price menu
    st.markdown('<p style="font-family:Google Sans,sans-serif;font-size:0.95rem;font-weight:500;color:#202124;">Price Menu</p>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="g-price-item"><div class="g-icon meat">🥩</div><div class="g-text"><div class="g-name">Meat</div><div class="g-val">RM {CONFIG['base_prices']['meat']:.2f}</div></div></div>
        <div class="g-price-item"><div class="g-icon rice">🍚</div><div class="g-text"><div class="g-name">Rice</div><div class="g-val">RM {CONFIG['base_prices']['rice']:.2f}</div></div></div>
        <div class="g-price-item"><div class="g-icon vege">🥬</div><div class="g-text"><div class="g-name">Vegetable</div><div class="g-val">RM {CONFIG['base_prices']['vege']:.2f}</div></div></div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Portion: **S** ×{CONFIG['size_multipliers']['S']:.2f} &nbsp; "
        f"**M** ×{CONFIG['size_multipliers']['M']:.2f} &nbsp; "
        f"**L** ×{CONFIG['size_multipliers']['L']:.2f}"
    )

    # --- Image History ---
    if st.session_state.image_history:
        st.markdown("---")
        st.markdown('<p style="font-family:Google Sans,sans-serif;font-size:0.95rem;font-weight:500;color:#202124;">Recent Images</p>', unsafe_allow_html=True)
        history_labels = [f"{i+1}. {name}" for i, (name, _) in enumerate(st.session_state.image_history)]
        default_idx = min(st.session_state.active_index, len(history_labels) - 1)
        selected_hist = st.radio(
            "Switch image:",
            options=range(len(history_labels)),
            format_func=lambda i: history_labels[i],
            index=default_idx,
            key=f"history_radio_{st.session_state.upload_counter}",
            label_visibility="collapsed",
        )
        if not st.session_state.new_upload_pending:
            st.session_state.active_index = selected_hist
        else:
            st.session_state.new_upload_pending = False

        if st.button("Clear history", use_container_width=True):
            st.session_state.image_history = []
            st.session_state.active_index = 0
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <p style="font-size:0.7rem;color:#9aa0a6;text-align:center;">
    <strong>Zapfan Smart Cashier</strong> © 2026<br/>
    <strong>TARUMT</strong> BMCS2203 Artificial Intelligence<br/>
    <strong>Team:</strong> Leong · Ooi · Chaw<br/>
    <a href="https://streamlit.io" target="_blank" style="color:#1a73e8;text-decoration:none;">Powered by Streamlit</a>
    </p>
    """, unsafe_allow_html=True)

# --- Main Area: Top-level page tabs ---
page_about, page_checkout, page_training = st.tabs(["ℹ️ About", "🍽️ Smart Checkout", "📊 Training Results"])

# ==============================================================================
# Page 0: About Project
# ==============================================================================
with page_about:
    about_col1, about_col2 = st.columns([1.5, 1])
    
    with about_col1:
        st.markdown("""
        ### 📚 Project Overview
        
        **Mixed Rice Price Prediction by using Object Classification using Deep Learning**
        
        #### Problem Background
        "Economy rice" (known locally as *Zapfan*) is a staple dining option in university cafeterias. 
        However, manual checkout and pricing remains subjective and error-prone, leading to:
        
        - 🔴 **Inconsistent pricing** — Different costs for identical portions based on cashier judgment
        - ⏱️ **Operational bottlenecks** — Long queues during peak lunch hours
        - 😞 **Student dissatisfaction** — Lack of transparent, standardized pricing
        
        Our solution? **AI-powered automated visual recognition and pricing.**
        
        #### Project Objectives
        1. **Develop** a functional computer vision prototype using Python
        2. **Implement & evaluate** three distinct AI object detection architectures
        3. **Design** a programmatic pricing algorithm based on portion size estimation
        4. **Deploy** a practical web interface for real-world cafeteria usage
        
        #### Motivation
        This project combines practical problem-solving with advanced machine learning techniques,
        addressing real operational challenges in campus dining while demonstrating cutting-edge
        computer vision capabilities.
        """)
    
    with about_col2:
        st.markdown("""
        ### 👥 Team Members
        
        **BMCS2203 AI Assignment**  
        Tutorial Group 5  
        Tutor: Ms Yan Yen Wei
        
        ---
        
        | # | Name | Role | 
        |---|------|------|
        | 1 | **Leong Kai Sheng** | Faster R-CNN |
        | 2 | **Ooi Jun Kang** | YOLOv8 |
        | 3 | **Chaw Chun Jia** | RT-DETR |
        
        **Programme:** BSc Information Technology (Hons) in Software Systems Development  
        **Year:** Y2S3  
        **Institution:** TARUMT
        """)

    st.markdown(
        """
        <div class="m-card-flat">
            <p style="margin:0 0 6px 0;font-size:0.85rem;color:#5f6368;">Quick Start</p>
            <ul style="margin:0 0 0 16px;">
                <li>Upload a plate photo or use the Camera tab</li>
                <li>Optionally enable Quick Settings to tune pricing</li>
                <li>Add demo images under <code>demo_images/</code> for testing</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.divider()
    
    st.markdown("""
    ### 🏆 Model Performance Comparison
    
    **Final Results:** Rigorous multi-architecture benchmarking on the food_dataset
    """)
    
    # Performance metrics table
    performance_data = {
        "Model": ["**YOLOv8**", "**RT-DETR**", "**Faster R-CNN**"],
        "mAP@50": ["0.8649", "0.9596", "0.9950"],
        "Inference Speed": ["5.26 ms/img", "38.15 ms/img", "92.12 ms/img"],
        "Best For": ["⚡ Speed & Edge", "⚖️ Balanced", "🎯 Accuracy"],
    }
    
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        st.metric(
            label="**YOLOv8** (Single-Stage CNN)",
            value="0.8649 mAP",
            delta="5.26 ms/img 📍",
            delta_color="inverse",
        )
        st.caption("**Strength:** Extremely fast inference, ideal for real-time edge deployment")
        st.caption("**Weakness:** Slightly lower accuracy on complex occlusions")
    
    with perf_col2:
        st.metric(
            label="**RT-DETR** (Transformer)",
            value="0.9596 mAP",
            delta="38.15 ms/img ⏱️",
            delta_color="normal",
        )
        st.caption("**Strength:** Excellent balance of speed & accuracy; handles overlapping items well")
        st.caption("**Weakness:** Moderate inference time")
    
    with perf_col3:
        st.metric(
            label="**Faster R-CNN** (Two-Stage CNN)",
            value="0.9950 mAP",
            delta="92.12 ms/img 🎯",
            delta_color="normal",
        )
        st.caption("**Strength:** Near-perfect accuracy; highest spatial localization precision")
        st.caption("**Weakness:** Slow inference; requires GPU support")
    
    st.divider()
    
    # Technical details
    col_tech1, col_tech2 = st.columns(2)
    
    with col_tech1:
        st.markdown("""
        ### 🔧 Technologies Used
        
        - **Ultralytics** — YOLOv8 & RT-DETR model training
        - **PyTorch** — Faster R-CNN custom implementation
        - **OpenCV (cv2)** — Image processing & visualization
        - **Streamlit** — Interactive web deployment
        - **Python** — Core development language
        """)
    
    with col_tech2:
        st.markdown("""
        ### 📊 Dataset
        
        **food_dataset** with mixed annotation formats:
        - Standard bounding boxes (YOLO format)
        - Segmentation polygons (precise outlines)
        
        **Classes (4):**
        - 🥩 Meat (ID: 0)
        - 🍚 Rice (ID: 2)
        - 🥬 Vegetable (ID: 3)
        - 🍽️ Plate (ID: 1)
        
        **Split:** 80% training, 20% validation
        """)

    st.divider()

    st.markdown("### 📈 Model Comparison (Uploaded Results)")
    comparison_found = False
    for img_path in COMPARISON_IMAGES:
        if os.path.isfile(img_path):
            comparison_found = True
            st.image(img_path, use_container_width=True)

    if not comparison_found:
        st.info("No comparison images found. Add images like 'yolo vs rtdetr.png' to the project root.")

    st.markdown(
        """
        ### ✅ Conclusion: Optimal Model
        Based on the uploaded comparison results, **RT-DETR** provides the best balance between accuracy
        and speed for real cafeteria use. It handles overlapping food items better than YOLO and runs
        significantly faster than Faster R-CNN while maintaining high mAP.
        """
    )
    
    st.markdown("""
    ### 💡 Key Achievements
    
    ✅ **Rigorous Multi-Architecture Benchmarking** — Successfully implemented and compared three distinct paradigms of object detection
    
    ✅ **Custom Data Engineering** — Seamless translation of YOLO polygon formats to PyTorch tensor requirements
    
    ✅ **Spatial Pricing Algorithm** — Automated portion size estimation based on bbox area ratios
    
    ✅ **End-to-End Deployment** — Full prototype implementation with interactive Streamlit web interface
    
    ### 🚀 Future Enhancements
    
    - Instance segmentation (exact pixel-level masks) for precise sizing
    - 3D depth sensing (RGB-D cameras) for volumetric accuracy
    - Edge AI optimization (quantization, TensorRT) for low-cost deployment
    - Active learning pipeline for continuous model improvement
    """)

# ==============================================================================
# Page 1: Smart Checkout
# ==============================================================================
with page_checkout:
    st.markdown('<div class="g-section">Add an image</div>', unsafe_allow_html=True)
    input_tab_upload, input_tab_camera, input_tab_demo = st.tabs([
        "Upload file",
        "Camera",
        "Demo images",
    ])

    with input_tab_upload:
        uploaded_file = st.file_uploader(
            "Drag & drop or browse a photo of your plate",
            type=["jpg", "jpeg", "png"],
            key="uploader",
        )

        # Process a new upload
        if uploaded_file is not None:
            current_file_id = uploaded_file.file_id
            if current_file_id != st.session_state.last_file_id:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if img_bgr is None:
                    st.error("Could not read the uploaded image. Please try a different file.")
                else:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    base_name = uploaded_file.name
                    existing_names = [n for n, _ in st.session_state.image_history]
                    name = base_name
                    counter = 2
                    while name in existing_names:
                        dot = base_name.rfind('.')
                        if dot != -1:
                            name = f"{base_name[:dot]} ({counter}){base_name[dot:]}"
                        else:
                            name = f"{base_name} ({counter})"
                        counter += 1
                    st.session_state.image_history.append((name, img_rgb))
                    st.session_state.active_index = len(st.session_state.image_history) - 1
                    st.session_state.last_file_id = current_file_id
                    st.session_state.new_upload_pending = True
                    st.session_state.auto_analyse = True
                    st.session_state.upload_counter += 1
                    st.rerun()

    with input_tab_camera:
        st.caption("Use your device camera. The image will be analysed automatically.")
        camera_photo = st.camera_input(
            "Tap to capture",
            key="camera_input",
        )

        if camera_photo is not None:
            current_camera_id = camera_photo.file_id
            if current_camera_id != st.session_state.last_camera_id:
                cam_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
                cam_bgr = cv2.imdecode(cam_bytes, cv2.IMREAD_COLOR)

                if cam_bgr is None:
                    st.error("Could not read the camera image. Please try again.")
                else:
                    cam_rgb = cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2RGB)
                    st.session_state.camera_counter += 1
                    base_name = f"Camera Shot {st.session_state.camera_counter}"
                    existing_names = [n for n, _ in st.session_state.image_history]
                    name = base_name
                    counter = 2
                    while name in existing_names:
                        name = f"{base_name} ({counter})"
                        counter += 1
                    st.session_state.image_history.append((name, cam_rgb))
                    st.session_state.active_index = len(st.session_state.image_history) - 1
                    st.session_state.last_camera_id = current_camera_id
                    st.session_state.new_upload_pending = True
                    st.session_state.auto_analyse = True
                    st.session_state.upload_counter += 1
                    st.rerun()

    with input_tab_demo:
        demo_files = _list_demo_images()
        if not demo_files:
            st.info(
                "No demo images found. Add JPG/PNG files to the demo_images/ folder."
            )
        else:
            st.caption("Select a bundled demo image for quick testing.")
            demo_choice = st.selectbox("Demo image", demo_files)
            if st.button("Use demo image", type="primary", use_container_width=True):
                demo_img = _load_demo_image(demo_choice)
                if demo_img is None:
                    st.error("Could not load the selected demo image.")
                else:
                    existing_names = [n for n, _ in st.session_state.image_history]
                    name = demo_choice
                    counter = 2
                    while name in existing_names:
                        dot = demo_choice.rfind('.')
                        if dot != -1:
                            name = f"{demo_choice[:dot]} ({counter}){demo_choice[dot:]}"
                        else:
                            name = f"{demo_choice} ({counter})"
                        counter += 1
                    st.session_state.image_history.append((name, demo_img))
                    st.session_state.active_index = len(st.session_state.image_history) - 1
                    st.session_state.new_upload_pending = True
                    st.session_state.auto_analyse = True
                    st.session_state.upload_counter += 1
                    st.rerun()

    # --- Display active image & run analysis ---
    if st.session_state.image_history:
        active_name, img_rgb = st.session_state.image_history[st.session_state.active_index]

        st.markdown('<div class="g-section">Current image</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:#5f6368;font-size:0.85rem;margin:0 0 8px 0;">{active_name}</p>', unsafe_allow_html=True)
        st.image(img_rgb, use_container_width=True)

        # Action buttons
        btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])
        with btn_col1:
            run_analysis = st.button("Analyse", type="primary", use_container_width=True)
        with btn_col2:
            remove_current = st.button("Remove", use_container_width=True)
        with btn_col3:
            upload_new = st.button("New image", use_container_width=True)

        # Handle remove
        if remove_current:
            st.session_state.image_history.pop(st.session_state.active_index)
            if st.session_state.image_history:
                st.session_state.active_index = max(0, st.session_state.active_index - 1)
            else:
                st.session_state.active_index = 0
            st.rerun()

        if upload_new:
            st.info("Use the **Upload file** or **Camera** tab above to add a new image.")

        # Run analysis (manual button or automatic on new upload)
        should_analyse = run_analysis or st.session_state.auto_analyse
        if st.session_state.auto_analyse:
            st.session_state.auto_analyse = False
        if should_analyse:
            st.markdown("---")
            if compare_all:
                st.markdown('<div class="g-section">Model comparison</div>', unsafe_allow_html=True)
                available_options = [
                    m for m in MODEL_OPTIONS
                    if get_detector(
                        _resolve_model_key(m),
                        warmup=st.session_state.get("warmup_models", True),
                    ) is not None
                ]
                if not available_options:
                    st.error("No model weights found. Please add model files to the project.")
                else:
                    tabs = st.tabs([m.split(" (")[0] for m in available_options])
                    for tab, m_opt in zip(tabs, available_options):
                        key_prefix = _resolve_model_key(m_opt)
                        analyse_and_display(img_rgb, m_opt, container=tab, key_prefix=key_prefix)
            else:
                analyse_and_display(
                    img_rgb,
                    model_option,
                    key_prefix=_resolve_model_key(model_option),
                )

        # Batch analysis across history
        if len(st.session_state.image_history) > 1:
            with st.expander("Batch analysis (history)", expanded=False):
                st.caption("Run analysis on all images in history using the selected model.")
                run_batch = st.button("Run batch analysis", use_container_width=True)
                if run_batch:
                    model_key = _resolve_model_key(model_option)
                    detector = get_detector(
                        model_key,
                        warmup=st.session_state.get("warmup_models", True),
                    )
                    if detector is None:
                        st.error("Selected model file is missing. Please choose another model.")
                    else:
                        rows = []
                        with st.spinner("Running batch analysis..."):
                            for name, img in st.session_state.image_history:
                                result = checkout(img, detector)
                                rows.append({
                                    "image": name,
                                    "items": len(result.billable_items),
                                    "total": f"{CONFIG['currency']}{result.total_price:.2f}",
                                    "speed_ms": f"{result.inference_time_ms:.0f}",
                                })
                        st.dataframe(rows, use_container_width=True)

                        output = io.StringIO()
                        writer = csv.writer(output)
                        writer.writerow(["image", "items", "total", "speed_ms"])
                        for row in rows:
                            writer.writerow([
                                row["image"],
                                row["items"],
                                row["total"],
                                row["speed_ms"],
                            ])
                        st.download_button(
                            "Download batch summary (csv)",
                            data=output.getvalue(),
                            file_name="zapfan_batch_summary.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

        # Footer
        st.markdown('<div class="g-footer">🏆 Zapfan Smart Cashier © 2026 &middot; TARUMT BMCS2203 AI Project &middot; <a href="https://streamlit.io" target="_blank">Streamlit</a></div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="g-empty">
            <div class="g-empty-icon">📷</div>
            <h3>No image yet</h3>
            <p>Upload a photo or take a picture of your economy rice plate to get started.</p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# Page 2: Training Results
# ==============================================================================
with page_training:
    st.markdown(
        '<div class="g-section" style="font-size:1.1rem;">Training Results &amp; Model Evaluation</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "View training metrics, confusion matrices, and evaluation curves for all three models. "
        "Place image files in the `training_results/<model>/` folders."
    )
    st.markdown(
        """
        <div class="m-card-flat">
            <p style="margin:0 0 6px 0;font-size:0.85rem;color:#5f6368;">Tip</p>
            <p style="margin:0;font-size:0.85rem;">If plots do not show up, verify file names like <code>results.png</code> and <code>confusion_matrix.png</code> exist in each model folder.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Discover available plots per model ---
    IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
    any_plots_found = False

    # Build tabs for all three models
    tr_tab_yolo, tr_tab_rtdetr, tr_tab_frcnn = st.tabs(["YOLOv8", "RT-DETR", "Faster R-CNN"])

    for tr_tab, model_key in zip(
        [tr_tab_yolo, tr_tab_rtdetr, tr_tab_frcnn],
        ["yolo", "rtdetr", "frcnn"],
    ):
        with tr_tab:
            model_info = MODEL_RESULTS_MAP[model_key]
            results_dir = model_info["dir"]
            model_display = model_info["name"]

            if not os.path.isdir(results_dir):
                st.info(f"No training results folder found for **{model_display}**.")
                continue

            # Collect known plots that exist
            found_plots = []
            for filename, title, description in TRAINING_PLOTS:
                filepath = os.path.join(results_dir, filename)
                if os.path.isfile(filepath):
                    found_plots.append((filepath, title, description))

            # Also discover any extra images not in the predefined list
            known_filenames = {fn for fn, _, _ in TRAINING_PLOTS}
            extra_files = sorted([
                f for f in os.listdir(results_dir)
                if Path(f).suffix.lower() in IMAGE_EXTS and f not in known_filenames
            ])
            for extra in extra_files:
                filepath = os.path.join(results_dir, extra)
                title = Path(extra).stem.replace("_", " ").title()
                found_plots.append((filepath, title, ""))

            if not found_plots:
                st.markdown(
                    f'<div class="g-empty" style="padding:2rem 1rem;">'
                    f'<div class="g-empty-icon" style="width:60px;height:60px;font-size:1.6rem;">📊</div>'
                    f'<h3 style="font-size:1rem;">No training plots for {model_display}</h3>'
                    f'<p style="font-size:0.85rem;">'
                    f'Place <code>results.png</code>, <code>confusion_matrix.png</code>, etc. in '
                    f'<code>training_results/{model_key}/</code></p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                continue

            any_plots_found = True

            # ── Display Results & Confusion Matrix side-by-side if both exist ──
            results_path = os.path.join(results_dir, "results.png")
            confusion_path = os.path.join(results_dir, "confusion_matrix.png")
            has_results = os.path.isfile(results_path)
            has_confusion = os.path.isfile(confusion_path)

            if has_results and has_confusion:
                st.markdown(
                    f'<div class="g-section">Key Metrics — {model_display}</div>',
                    unsafe_allow_html=True,
                )
                col_res, col_conf = st.columns(2)
                with col_res:
                    st.image(
                        results_path,
                        caption="Training Results — Metrics over epochs",
                        use_container_width=True,
                    )
                with col_conf:
                    st.image(
                        confusion_path,
                        caption="Confusion Matrix",
                        use_container_width=True,
                    )
            elif has_results:
                st.markdown(
                    f'<div class="g-section">Training Results — {model_display}</div>',
                    unsafe_allow_html=True,
                )
                st.image(results_path, caption="Training Results", use_container_width=True)
            elif has_confusion:
                st.markdown(
                    f'<div class="g-section">Confusion Matrix — {model_display}</div>',
                    unsafe_allow_html=True,
                )
                st.image(confusion_path, caption="Confusion Matrix", use_container_width=True)

            # ── Additional plots in expandable section ──
            extra_plots = [
                (fp, t, d) for fp, t, d in found_plots
                if os.path.basename(fp) not in ("results.png", "confusion_matrix.png")
            ]
            if extra_plots:
                with st.expander(
                    f"Additional evaluation plots ({len(extra_plots)})",
                    expanded=False,
                ):
                    # Display in rows of 2
                    for i in range(0, len(extra_plots), 2):
                        cols = st.columns(2)
                        for j, col in enumerate(cols):
                            idx = i + j
                            if idx < len(extra_plots):
                                fp, title, desc = extra_plots[idx]
                                with col:
                                    caption = f"{title} — {desc}" if desc else title
                                    st.image(fp, caption=caption, use_container_width=True)
