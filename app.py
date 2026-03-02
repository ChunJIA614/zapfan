# -*- coding: utf-8 -*-
"""
Zapfan Smart Cashier — Streamlit App
Economy Rice (Nasi Campur) AI-Powered Pricing System
Models: YOLO | RT-DETR | Faster R-CNN
"""

import streamlit as st
import cv2
import numpy as np
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

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
# Constants
# ==============================================================================
CLASSES = ['meat', 'plate', 'rice', 'vege']
BASE_PRICES = {'meat': 4.00, 'rice': 1.50, 'vege': 2.00}
CURRENCY = "RM"
SIZE_MULTIPLIERS = {'S': 0.7, 'M': 1.0, 'L': 1.5}
COLORS = {
    'rice': (0, 255, 0),
    'vege': (255, 0, 0),
    'meat': (0, 0, 255),
    'plate': (0, 255, 255),
}

# Model file paths (relative to this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(BASE_DIR, "model_mixed_rice.pt")
RTDETR_PATH = os.path.join(BASE_DIR, "model_rtdetr_mixed_rice.pt")
FRCNN_PATH = os.path.join(BASE_DIR, "faster_rcnn_mixed_rice.pth")


# ==============================================================================
# Helper Functions
# ==============================================================================
def get_portion_size(box_area: float, plate_area: float) -> str:
    """Determine portion size based on box-to-plate area ratio."""
    if plate_area <= 0:
        return 'M'
    ratio = box_area / plate_area
    if ratio < (2 / 9):
        return 'S'
    elif ratio > (4 / 9):
        return 'L'
    else:
        return 'M'


# ==============================================================================
# Model Loading (cached so models load only once)
# ==============================================================================
@st.cache_resource
def load_yolo():
    """Load YOLO model."""
    if not os.path.exists(YOLO_PATH):
        return None
    from ultralytics import YOLO
    return YOLO(YOLO_PATH)


@st.cache_resource
def load_rtdetr():
    """Load RT-DETR model."""
    if not os.path.exists(RTDETR_PATH):
        return None
    from ultralytics import RTDETR
    return RTDETR(RTDETR_PATH)


@st.cache_resource
def load_frcnn():
    """Load Faster R-CNN model."""
    if not os.path.exists(FRCNN_PATH):
        return None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASSES) + 1)
    model.load_state_dict(torch.load(FRCNN_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


# ==============================================================================
# Inference Functions
# ==============================================================================
def run_ultralytics(model, img_rgb):
    """Run inference using an Ultralytics model (YOLO / RT-DETR).

    Returns
    -------
    valid_foods : list[dict]
        Detected food items (name, box, score).
    plates : list[dict]
        All detected plates (box, score), sorted by x-coordinate.
    """
    results = model.predict(img_rgb, conf=0.25, verbose=False)
    valid_foods = []
    plates = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            task_name = CLASSES[cls_id]

            if task_name == 'plate':
                plates.append({'box': (x1, y1, x2, y2), 'score': conf})
            else:
                valid_foods.append({
                    'name': task_name,
                    'box': (x1, y1, x2, y2),
                    'score': conf,
                })

    # Sort plates left-to-right so numbering is consistent
    plates.sort(key=lambda p: p['box'][0])
    return valid_foods, plates


def run_frcnn(model, img_rgb):
    """Run inference using Faster R-CNN (PyTorch).

    Returns
    -------
    valid_foods : list[dict]
        Detected food items (name, box, score).
    plates : list[dict]
        All detected plates (box, score), sorted by x-coordinate.
    """
    device = next(model.parameters()).device
    img_tensor = torch.as_tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    CONF_THRESH = 0.5
    mask = predictions['scores'] > CONF_THRESH
    boxes = predictions['boxes'][mask].cpu().numpy()
    labels = predictions['labels'][mask].cpu().numpy()
    scores = predictions['scores'][mask].cpu().numpy()

    valid_foods = []
    plates = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cls_id = labels[i] - 1  # Shift for background class
        task_name = CLASSES[cls_id]
        conf = scores[i]

        if task_name == 'plate':
            plates.append({'box': (x1, y1, x2, y2), 'score': conf})
        else:
            valid_foods.append({
                'name': task_name,
                'box': (x1, y1, x2, y2),
                'score': conf,
            })

    # Sort plates left-to-right so numbering is consistent
    plates.sort(key=lambda p: p['box'][0])
    return valid_foods, plates


def _intersection_area(box_a, box_b):
    """Compute intersection area between two (x1, y1, x2, y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 > x1 and y2 > y1:
        return (x2 - x1) * (y2 - y1)
    return 0


def assign_foods_to_plates(valid_foods, plates):
    """Assign each food item to its most-overlapping plate.

    Returns a dict  plate_index -> [food_items].
    Index -1 is used when no plates were detected at all.
    """
    if not plates:
        return {-1: list(valid_foods)}

    assignments = {i: [] for i in range(len(plates))}

    for food in valid_foods:
        fx1, fy1, fx2, fy2 = food['box']
        food_cx, food_cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
        food_area = max(1, (fx2 - fx1) * (fy2 - fy1))

        best_plate = -1
        best_overlap = 0

        for pi, plate in enumerate(plates):
            overlap = _intersection_area(food['box'], plate['box'])
            if overlap > best_overlap:
                best_overlap = overlap
                best_plate = pi

        # If overlap covers > 30 % of the food area, assign directly
        if best_plate >= 0 and best_overlap > 0.3 * food_area:
            assignments[best_plate].append(food)
        else:
            # Fall back to nearest plate by center distance
            min_dist = float('inf')
            nearest = 0
            for pi, plate in enumerate(plates):
                px1, py1, px2, py2 = plate['box']
                plate_cx, plate_cy = (px1 + px2) / 2, (py1 + py2) / 2
                dist = ((food_cx - plate_cx) ** 2 + (food_cy - plate_cy) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest = pi
            assignments[nearest].append(food)

    return assignments


# Distinct colours for plate bounding boxes when multiple plates are present
_PLATE_COLORS = [
    (0, 255, 255), (255, 165, 0), (255, 0, 255),
    (128, 255, 0), (0, 128, 255), (255, 128, 128),
]


def draw_results(img_rgb, valid_foods, plates, plate_assignments):
    """Draw bounding boxes, labels, and per-plate prices on the image.

    Returns
    -------
    img_display : ndarray  – annotated image
    plate_results : list[dict] – per-plate receipt data
    grand_total : float
    all_cropped : list[tuple] – (name, score, cropped_img)
    """
    img_display = img_rgb.copy()
    img_clean = img_rgb.copy()
    grand_total = 0.0
    plate_results = []
    all_cropped = []
    num_plates = len(plates) if plates else 1

    # --- Draw every detected plate box ---
    for pi, plate in enumerate(plates):
        px1, py1, px2, py2 = plate['box']
        p_color = _PLATE_COLORS[pi % len(_PLATE_COLORS)]
        cv2.rectangle(img_display, (px1, py1), (px2, py2), p_color, 3)
        p_label = (f"Plate {pi + 1} ({plate['score']:.2f})"
                   if num_plates > 1
                   else f"Plate ({plate['score']:.2f})")
        cv2.putText(img_display, p_label,
                    (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, p_color, 2)

    # --- Process each plate group ---
    for plate_idx in sorted(plate_assignments.keys()):
        foods = plate_assignments[plate_idx]
        if not foods:
            continue

        plate_num = plate_idx + 1 if plate_idx >= 0 else 1

        # Determine plate area for portion sizing
        if 0 <= plate_idx < len(plates):
            px1, py1, px2, py2 = plates[plate_idx]['box']
            plate_area = max(1, (px2 - px1) * (py2 - py1))
        else:
            # No plate detected — estimate from food boundaries
            min_x = min(d['box'][0] for d in foods)
            min_y = min(d['box'][1] for d in foods)
            max_x = max(d['box'][2] for d in foods)
            max_y = max(d['box'][3] for d in foods)
            plate_area = max(1, (max_x - min_x) * (max_y - min_y))
            cv2.rectangle(img_display, (min_x, min_y), (max_x, max_y),
                          (255, 255, 255), 2, cv2.LINE_AA)

        plate_total = 0.0
        receipt_lines = []

        for item in foods:
            task_name = item['name']
            conf = item['score']
            x1, y1, x2, y2 = item['box']
            box_area = (x2 - x1) * (y2 - y1)
            size_label = get_portion_size(box_area, plate_area)

            multiplier = SIZE_MULTIPLIERS[size_label]
            final_price = BASE_PRICES[task_name] * multiplier
            plate_total += final_price

            ratio_percentage = (box_area / plate_area) * 100
            receipt_lines.append({
                'item': task_name.capitalize(),
                'size': size_label,
                'ratio': ratio_percentage,
                'price': final_price,
            })

            # Crop
            crop_y1, crop_y2 = max(0, y1), min(img_clean.shape[0], y2)
            crop_x1, crop_x2 = max(0, x1), min(img_clean.shape[1], x2)
            if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                cropped_part = img_clean[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                all_cropped.append((task_name, conf, cropped_part))

            # Draw bounding box
            color = COLORS[task_name]
            rgb = (color[2], color[1], color[0])
            cv2.rectangle(img_display, (x1, y1), (x2, y2), rgb, 3)

            label = f"{task_name.capitalize()} ({size_label}) {CURRENCY}{final_price:.2f}"
            if num_plates > 1:
                label = f"P{plate_num} {label}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img_display, (x1, max(y1 - t_size[1] - 10, 0)),
                          (x1 + t_size[0], max(y1, 10)), rgb, -1)
            cv2.putText(img_display, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        grand_total += plate_total
        plate_results.append({
            'plate_num': plate_num,
            'receipt_lines': receipt_lines,
            'total': plate_total,
        })

    return img_display, plate_results, grand_total, all_cropped


# ==============================================================================
# Helper: run a single model and display results in a container
# ==============================================================================
def analyse_and_display(img_rgb, model_option, container=None):
    """Load the chosen model, run inference, and render results inside *container*."""
    if container is None:
        container = st.container()
    with container:
        # Load model
        with st.spinner(f"Loading {model_option} model..."):
            if "YOLO" in model_option:
                model = load_yolo()
            elif "RT-DETR" in model_option:
                model = load_rtdetr()
            else:
                model = load_frcnn()

        if model is None:
            st.error(f"Model file not found for **{model_option}**! "
                     "Make sure the model weights are in the project folder.")
            return

        # Run inference
        with st.spinner(f"Analyzing your plate(s) with {model_option}..."):
            if "Faster R-CNN" in model_option:
                valid_foods, plates = run_frcnn(model, img_rgb)
            else:
                valid_foods, plates = run_ultralytics(model, img_rgb)

        if not valid_foods:
            st.warning(f"[{model_option}] No food items detected. Try a different model or image.")
            return

        # Assign foods to plates & draw
        plate_assignments = assign_foods_to_plates(valid_foods, plates)
        img_display, plate_results, grand_total, all_cropped = draw_results(
            img_rgb, valid_foods, plates, plate_assignments
        )

        num_plates = len(plates) if plates else 1
        total_items = sum(len(pr['receipt_lines']) for pr in plate_results)
        model_label = model_option.split(" (")[0]

        # ---- Summary stat chips (Google style) ----
        plates_word = "Plates" if num_plates != 1 else "Plate"
        stat_html = (
            '<div class="g-stats">'
            f'<div class="g-chip"><span class="g-chip-num">{num_plates}</span><span class="g-chip-label">{plates_word}</span></div>'
            f'<div class="g-chip"><span class="g-chip-num">{total_items}</span><span class="g-chip-label">Items</span></div>'
            f'<div class="g-chip"><span class="g-chip-num">{CURRENCY}{grand_total:.2f}</span><span class="g-chip-label">Total</span></div>'
            '</div>'
        )
        st.markdown(stat_html, unsafe_allow_html=True)

        # ---- Result columns ----
        col_img, col_receipt = st.columns([3, 2])

        with col_img:
            st.markdown('<div class="g-section">Detection result</div>', unsafe_allow_html=True)
            st.image(img_display, use_container_width=True)
            if num_plates > 1:
                st.info(f"**{num_plates} plates** detected — prices calculated per plate.")

        with col_receipt:
            # Build receipt HTML
            receipt_html = '<div class="receipt-card">'
            receipt_html += (
                '<div class="receipt-header">'
                f'Receipt <span class="model-tag">{model_label}</span>'
                '</div>'
            )

            for pr in plate_results:
                if num_plates > 1:
                    pnum = pr["plate_num"]
                    receipt_html += f'<div class="plate-badge">Plate {pnum}</div>'

                for line in pr['receipt_lines']:
                    item_emoji = {"Meat": "\ud83e\udd69", "Rice": "\ud83c\udf5a", "Vege": "\ud83e\udd6c"}.get(line['item'], "\ud83c\udf5e")
                    i_name = line["item"]
                    i_size = line["size"]
                    i_price = line["price"]
                    receipt_html += (
                        f'<div class="receipt-item">'
                        f'<span class="label">{item_emoji} {i_name}'
                        f'<span class="tag">{i_size}</span></span>'
                        f'<span class="price">{CURRENCY}{i_price:.2f}</span>'
                        f'</div>'
                    )

                if num_plates > 1:
                    p_total = pr["total"]
                    receipt_html += (
                        f'<div class="receipt-total">'
                        f'<span>Subtotal</span><span>{CURRENCY}{p_total:.2f}</span>'
                        f'</div>'
                    )

            if num_plates > 1:
                receipt_html += (
                    f'<div class="receipt-grand">'
                    f'<span>Grand Total</span><span>{CURRENCY}{grand_total:.2f}</span>'
                    f'</div>'
                )
            else:
                receipt_html += (
                    f'<div class="receipt-total">'
                    f'<span>Total</span><span>{CURRENCY}{grand_total:.2f}</span>'
                    f'</div>'
                )

            receipt_html += '</div>'
            st.markdown(receipt_html, unsafe_allow_html=True)

        # ---- Cropped items ----
        if all_cropped:
            with st.expander(f"Detected items ({len(all_cropped)})", expanded=False):
                crop_cols = st.columns(min(len(all_cropped), 4))
                for idx, (c_name, c_score, c_img) in enumerate(all_cropped):
                    with crop_cols[idx % len(crop_cols)]:
                        st.image(c_img, caption=f"{c_name.capitalize()} ({c_score:.0%})",
                                 use_container_width=True)


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

    model_option = st.selectbox("AI Model", options=MODEL_OPTIONS, index=0)
    compare_all = st.checkbox("Compare all models side-by-side")

    st.markdown("---")

    # Google-style price menu
    st.markdown('<p style="font-family:Google Sans,sans-serif;font-size:0.95rem;font-weight:500;color:#202124;">Price Menu</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="g-price-item"><div class="g-icon meat">🥩</div><div class="g-text"><div class="g-name">Meat</div><div class="g-val">RM 4.00</div></div></div>
    <div class="g-price-item"><div class="g-icon rice">🍚</div><div class="g-text"><div class="g-name">Rice</div><div class="g-val">RM 1.50</div></div></div>
    <div class="g-price-item"><div class="g-icon vege">🥬</div><div class="g-text"><div class="g-name">Vegetable</div><div class="g-val">RM 2.00</div></div></div>
    """, unsafe_allow_html=True)
    st.caption("Portion: **S** ×0.7 &nbsp; **M** ×1.0 &nbsp; **L** ×1.5")

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
    st.markdown('<p style="font-size:0.75rem;color:#9aa0a6;text-align:center;">Zapfan © 2026 &middot; Powered by Streamlit</p>', unsafe_allow_html=True)

# --- Main Area: Image Input ---
st.markdown('<div class="g-section">Add an image</div>', unsafe_allow_html=True)
input_tab_upload, input_tab_camera = st.tabs(["Upload file", "Camera"])

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
            tabs = st.tabs([m.split(" (")[0] for m in MODEL_OPTIONS])
            for tab, m_opt in zip(tabs, MODEL_OPTIONS):
                analyse_and_display(img_rgb, m_opt, container=tab)
        else:
            analyse_and_display(img_rgb, model_option)

    # Footer
    st.markdown('<div class="g-footer">Zapfan Smart Cashier &copy; 2026 &middot; <a href="https://streamlit.io" target="_blank">Streamlit</a></div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="g-empty">
        <div class="g-empty-icon">📷</div>
        <h3>No image yet</h3>
        <p>Upload a photo or take a picture of your economy rice plate to get started.</p>
    </div>
    """, unsafe_allow_html=True)
