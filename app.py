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
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# State & Theme Management
# ==============================================================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True # Default to Next-Gen Dark Mode

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# ==============================================================================
# Custom CSS — Google Antigravity UI & Tailwind Injection
# ==============================================================================
# Generate CSS variables based on toggle state
if st.session_state.dark_mode:
    theme_vars = """
    :root {
        --ag-bg: #090e17;
        --ag-card: rgba(16, 24, 39, 0.55);
        --ag-border: rgba(255, 255, 255, 0.08);
        --ag-shadow: 0 12px 40px rgba(0, 0, 0, 0.6);
        --ag-text: #f8fafc;
        --ag-text-muted: #94a3b8;
        --ag-accent: #0ea5e9; /* Sky Blue */
        --ag-accent-glow: rgba(14, 165, 233, 0.25);
        --ag-success: #10b981; /* Emerald */
        --ag-warning: #f59e0b; /* Amber */
        
        /* Streamlit Native Overrides */
        --background-color: #090e17;
        --secondary-background-color: #0f172a;
        --text-color: #f8fafc;
        --primary-color: #0ea5e9;
    }
    """
else:
    theme_vars = """
    :root {
        --ag-bg: #f4f7fb;
        --ag-card: rgba(255, 255, 255, 0.65);
        --ag-border: rgba(255, 255, 255, 0.8);
        --ag-shadow: 0 10px 40px rgba(14, 165, 233, 0.08);
        --ag-text: #0f172a;
        --ag-text-muted: #64748b;
        --ag-accent: #0284c7; /* Darker Sky Blue */
        --ag-accent-glow: rgba(2, 132, 199, 0.15);
        --ag-success: #059669; /* Emerald */
        --ag-warning: #d97706; /* Amber */

        /* Streamlit Native Overrides */
        --background-color: #f4f7fb;
        --secondary-background-color: #ffffff;
        --text-color: #0f172a;
        --primary-color: #0284c7;
    }
    """

st.markdown(f"""
<style>
/* Inject Tailwind CSS via CDN */
@import url('https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

{theme_vars}

/* Global App Styling */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
}}
.stApp, [data-testid="stAppViewContainer"] {{
    background-color: var(--ag-bg) !important;
    background-image: 
        radial-gradient(circle at 15% 50%, var(--ag-accent-glow), transparent 25%),
        radial-gradient(circle at 85% 30%, rgba(16, 185, 129, 0.05), transparent 25%) !important;
    background-attachment: fixed;
    color: var(--ag-text) !important;
}}

/* Hide default streamlit headers/footers to make it look native */
header[data-testid="stHeader"] {{ background: transparent !important; }}
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}

/* Antigravity Custom Classes */
.ag-card {{
    background: var(--ag-card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--ag-border);
    border-radius: 24px;
    box-shadow: var(--ag-shadow);
    transition: transform 0.3s cubic-bezier(0.25, 0.8, 0.25, 1), box-shadow 0.3s ease;
}}
.ag-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 20px 40px var(--ag-accent-glow);
}}

/* Streamlit Sidebar overrides */
section[data-testid="stSidebar"] {{
    background-color: var(--ag-card) !important;
    backdrop-filter: blur(24px) !important;
    border-right: 1px solid var(--ag-border) !important;
}}

/* Streamlit Button Overrides (Pill shaped, floating) */
.stButton > button {{
    background: var(--ag-card) !important;
    border: 1px solid var(--ag-border) !important;
    border-radius: 9999px !important;
    backdrop-filter: blur(10px);
    box-shadow: var(--ag-shadow) !important;
    color: var(--ag-text) !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    padding: 0.5rem 1.5rem !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px);
    border-color: var(--ag-accent) !important;
    color: var(--ag-accent) !important;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, var(--ag-accent), #2563eb) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 8px 20px var(--ag-accent-glow) !important;
}}
.stButton > button[kind="primary"]:hover {{
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 12px 25px var(--ag-accent-glow) !important;
    color: #ffffff !important;
}}

/* Streamlit Image Wrapper styling */
[data-testid="stImage"] img {{
    border-radius: 16px;
    border: 1px solid var(--ag-border);
    box-shadow: var(--ag-shadow);
}}

/* Uploader & Camera container */
[data-testid="stFileUploader"] > div {{
    background: var(--ag-card) !important;
    border: 2px dashed var(--ag-border) !important;
    border-radius: 20px !important;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
}}
[data-testid="stFileUploader"] > div:hover {{
    border-color: var(--ag-accent) !important;
    background: var(--ag-accent-glow) !important;
}}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px; background: transparent; border: none;
}}
.stTabs[data-baseweb="tab"] {{
    background: var(--ag-card);
    border: 1px solid var(--ag-border);
    border-radius: 9999px;
    padding: 8px 20px;
    font-weight: 500;
    color: var(--ag-text-muted);
    transition: all 0.2s;
}}
.stTabs[data-baseweb="tab"][aria-selected="true"] {{
    background: var(--ag-accent) !important;
    color: #ffffff !important;
    border-color: var(--ag-accent) !important;
    box-shadow: 0 4px 15px var(--ag-accent-glow);
}}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Constants
# ==============================================================================
CLASSES = ['meat', 'plate', 'rice', 'vege']
BASE_PRICES = {'meat': 4.00, 'rice': 1.50, 'vege': 2.00}
CURRENCY = "RM"
SIZE_MULTIPLIERS = {'S': 0.7, 'M': 1.0, 'L': 1.5}

# Next-Gen Bounding Box Colors (RGB) - No Purple!
COLORS = {
    'meat': (239, 68, 68),    # Rose/Red-500
    'vege': (16, 185, 129),   # Emerald-500
    'rice': (226, 232, 240),  # Slate-200
    'plate': (14, 165, 233),  # Sky-500
}

# Distinct colours for plate bounding boxes when multiple plates are present
_PLATE_COLORS =[
    (14, 165, 233),   # Sky
    (245, 158, 11),   # Amber
    (16, 185, 129),   # Emerald
    (239, 68, 68),    # Red
    (99, 102, 241),   # Indigo
    (20, 184, 166),   # Teal
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(BASE_DIR, "model_mixed_rice.pt")
RTDETR_PATH = os.path.join(BASE_DIR, "model_rtdetr_mixed_rice.pt")
FRCNN_PATH = os.path.join(BASE_DIR, "faster_rcnn_mixed_rice.pth")

# ==============================================================================
# Helper Functions
# ==============================================================================
def get_portion_size(box_area: float, plate_area: float) -> str:
    if plate_area <= 0: return 'M'
    ratio = box_area / plate_area
    if ratio < (2 / 9): return 'S'
    elif ratio > (4 / 9): return 'L'
    else: return 'M'

@st.cache_resource
def load_yolo():
    if not os.path.exists(YOLO_PATH): return None
    from ultralytics import YOLO
    return YOLO(YOLO_PATH)

@st.cache_resource
def load_rtdetr():
    if not os.path.exists(RTDETR_PATH): return None
    from ultralytics import RTDETR
    return RTDETR(RTDETR_PATH)

@st.cache_resource
def load_frcnn():
    if not os.path.exists(FRCNN_PATH): return None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASSES) + 1)
    model.load_state_dict(torch.load(FRCNN_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

# ==============================================================================
# Inference Logic
# ==============================================================================
def run_ultralytics(model, img_rgb):
    results = model.predict(img_rgb, conf=0.25, verbose=False)
    valid_foods, plates = [],[]
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            task_name = CLASSES[cls_id]
            if task_name == 'plate':
                plates.append({'box': (x1, y1, x2, y2), 'score': conf})
            else:
                valid_foods.append({'name': task_name, 'box': (x1, y1, x2, y2), 'score': conf})
    plates.sort(key=lambda p: p['box'][0])
    return valid_foods, plates

def run_frcnn(model, img_rgb):
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
    valid_foods, plates = [],[]
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cls_id = labels[i] - 1
        task_name = CLASSES[cls_id]
        conf = scores[i]
        if task_name == 'plate':
            plates.append({'box': (x1, y1, x2, y2), 'score': conf})
        else:
            valid_foods.append({'name': task_name, 'box': (x1, y1, x2, y2), 'score': conf})
    plates.sort(key=lambda p: p['box'][0])
    return valid_foods, plates

def _intersection_area(box_a, box_b):
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    if x2 > x1 and y2 > y1: return (x2 - x1) * (y2 - y1)
    return 0

def assign_foods_to_plates(valid_foods, plates):
    if not plates: return {-1: list(valid_foods)}
    assignments = {i:[] for i in range(len(plates))}
    for food in valid_foods:
        fx1, fy1, fx2, fy2 = food['box']
        food_cx, food_cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
        food_area = max(1, (fx2 - fx1) * (fy2 - fy1))
        best_plate, best_overlap = -1, 0
        for pi, plate in enumerate(plates):
            overlap = _intersection_area(food['box'], plate['box'])
            if overlap > best_overlap:
                best_overlap = overlap
                best_plate = pi
        if best_plate >= 0 and best_overlap > 0.3 * food_area:
            assignments[best_plate].append(food)
        else:
            min_dist, nearest = float('inf'), 0
            for pi, plate in enumerate(plates):
                px1, py1, px2, py2 = plate['box']
                plate_cx, plate_cy = (px1 + px2) / 2, (py1 + py2) / 2
                dist = ((food_cx - plate_cx) ** 2 + (food_cy - plate_cy) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist, nearest = dist, pi
            assignments[nearest].append(food)
    return assignments

def draw_results(img_rgb, valid_foods, plates, plate_assignments):
    img_display = img_rgb.copy()
    img_clean = img_rgb.copy()
    grand_total = 0.0
    plate_results =[]
    all_cropped =[]
    num_plates = len(plates) if plates else 1

    for pi, plate in enumerate(plates):
        px1, py1, px2, py2 = plate['box']
        p_color = _PLATE_COLORS[pi % len(_PLATE_COLORS)]
        cv2.rectangle(img_display, (px1, py1), (px2, py2), p_color, 3)
        p_label = f"Plate {pi + 1} ({plate['score']:.2f})" if num_plates > 1 else f"Plate ({plate['score']:.2f})"
        cv2.putText(img_display, p_label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, p_color, 2)

    for plate_idx in sorted(plate_assignments.keys()):
        foods = plate_assignments[plate_idx]
        if not foods: continue
        plate_num = plate_idx + 1 if plate_idx >= 0 else 1

        if 0 <= plate_idx < len(plates):
            px1, py1, px2, py2 = plates[plate_idx]['box']
            plate_area = max(1, (px2 - px1) * (py2 - py1))
        else:
            min_x, min_y = min(d['box'][0] for d in foods), min(d['box'][1] for d in foods)
            max_x, max_y = max(d['box'][2] for d in foods), max(d['box'][3] for d in foods)
            plate_area = max(1, (max_x - min_x) * (max_y - min_y))
            cv2.rectangle(img_display, (min_x, min_y), (max_x, max_y), (255, 255, 255), 2, cv2.LINE_AA)

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
                'item': task_name.capitalize(), 'size': size_label, 'ratio': ratio_percentage, 'price': final_price
            })

            crop_y1, crop_y2 = max(0, y1), min(img_clean.shape[0], y2)
            crop_x1, crop_x2 = max(0, x1), min(img_clean.shape[1], x2)
            if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                cropped_part = img_clean[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                all_cropped.append((task_name, conf, cropped_part))

            color = COLORS[task_name]
            # OpenCV expects BGR. Our constants are RGB, so reverse it
            rgb = (color[2], color[1], color[0])
            cv2.rectangle(img_display, (x1, y1), (x2, y2), rgb, 3)

            label = f"{task_name.capitalize()} ({size_label}) {CURRENCY}{final_price:.2f}"
            if num_plates > 1: label = f"P{plate_num} {label}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img_display, (x1, max(y1 - t_size[1] - 10, 0)), (x1 + t_size[0], max(y1, 10)), rgb, -1)
            cv2.putText(img_display, label, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        grand_total += plate_total
        plate_results.append({'plate_num': plate_num, 'receipt_lines': receipt_lines, 'total': plate_total})

    return img_display, plate_results, grand_total, all_cropped

# ==============================================================================
# UI Component Rendering
# ==============================================================================
def analyse_and_display(img_rgb, model_option, container=None):
    if container is None:
        container = st.container()
    with container:
        with st.spinner(f"Initiating {model_option} neural engine..."):
            if "YOLO" in model_option: model = load_yolo()
            elif "RT-DETR" in model_option: model = load_rtdetr()
            else: model = load_frcnn()

        if model is None:
            st.error(f"System Offline: Model weights for **{model_option}** missing.")
            return

        with st.spinner(f"Analyzing semantic map with {model_option}..."):
            if "Faster R-CNN" in model_option: valid_foods, plates = run_frcnn(model, img_rgb)
            else: valid_foods, plates = run_ultralytics(model, img_rgb)

        if not valid_foods:
            st.warning("No edible constructs detected. Reposition and try again.")
            return

        plate_assignments = assign_foods_to_plates(valid_foods, plates)
        img_display, plate_results, grand_total, all_cropped = draw_results(img_rgb, valid_foods, plates, plate_assignments)

        num_plates = len(plates) if plates else 1
        total_items = sum(len(pr['receipt_lines']) for pr in plate_results)
        model_label = model_option.split(" (")[0]

        # Antigravity Stat Chips
        st.markdown(f"""
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 my-6">
            <div class="ag-card flex flex-col items-center justify-center p-4">
                <span class="text-3xl font-extrabold text-[var(--ag-accent)]">{num_plates}</span>
                <span class="text-xs font-semibold uppercase tracking-wider text-[var(--ag-text-muted)] mt-1">Plates Detected</span>
            </div>
            <div class="ag-card flex flex-col items-center justify-center p-4">
                <span class="text-3xl font-extrabold text-[var(--ag-success)]">{total_items}</span>
                <span class="text-xs font-semibold uppercase tracking-wider text-[var(--ag-text-muted)] mt-1">Items Categorized</span>
            </div>
            <div class="ag-card flex flex-col items-center justify-center p-4">
                <span class="text-3xl font-extrabold text-[var(--ag-accent)]">{CURRENCY} {grand_total:.2f}</span>
                <span class="text-xs font-semibold uppercase tracking-wider text-[var(--ag-text-muted)] mt-1">Total Valuation</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_img, col_receipt = st.columns([3, 2])

        with col_img:
            st.markdown('<h3 class="text-lg font-semibold text-[var(--ag-text)] mb-3 tracking-tight">Vision Output</h3>', unsafe_allow_html=True)
            st.image(img_display, use_container_width=True)

        with col_receipt:
            receipt_html = f"""
            <div class="ag-card p-6">
                <div class="flex justify-between items-center border-b border-[var(--ag-border)] pb-4 mb-4">
                    <h3 class="text-xl font-bold tracking-tight text-[var(--ag-text)]">Transaction</h3>
                    <span class="px-3 py-1 text-[10px] font-bold uppercase tracking-widest rounded-full bg-[var(--ag-accent-glow)] text-[var(--ag-accent)] border border-[var(--ag-border)]">{model_label}</span>
                </div>
            """
            for pr in plate_results:
                if num_plates > 1:
                    receipt_html += f'<div class="text-xs font-bold uppercase tracking-widest text-[var(--ag-text-muted)] mt-4 mb-2">Plate {pr["plate_num"]}</div>'
                
                for line in pr['receipt_lines']:
                    emoji = {"Meat": "🥩", "Rice": "🍚", "Vege": "🥬"}.get(line['item'], "🍽️")
                    receipt_html += f"""
                    <div class="flex justify-between items-center py-2">
                        <div class="flex items-center gap-3">
                            <span class="text-lg">{emoji}</span>
                            <div>
                                <div class="text-sm font-semibold text-[var(--ag-text)]">{line["item"]}</div>
                                <div class="text-[10px] text-[var(--ag-text-muted)] uppercase tracking-wide">Size: {line["size"]}</div>
                            </div>
                        </div>
                        <span class="font-bold text-[var(--ag-text)]">{CURRENCY} {line["price"]:.2f}</span>
                    </div>
                    """
                if num_plates > 1:
                    receipt_html += f"""
                    <div class="flex justify-between items-center pt-3 mt-2 border-t border-[var(--ag-border)] border-dashed">
                        <span class="text-sm font-medium text-[var(--ag-text-muted)]">Subtotal</span>
                        <span class="text-sm font-bold text-[var(--ag-text)]">{CURRENCY} {pr["total"]:.2f}</span>
                    </div>
                    """

            receipt_html += f"""
                <div class="flex justify-between items-center pt-4 mt-4 border-t-2 border-[var(--ag-border)]">
                    <span class="text-lg font-extrabold text-[var(--ag-text)] tracking-tight">Total</span>
                    <span class="text-2xl font-extrabold text-[var(--ag-accent)]">{CURRENCY} {grand_total:.2f}</span>
                </div>
            </div>
            """
            st.markdown(receipt_html, unsafe_allow_html=True)

        if all_cropped:
            with st.expander("Explore Detected Constructs"):
                crop_cols = st.columns(min(len(all_cropped), 4))
                for idx, (c_name, c_score, c_img) in enumerate(all_cropped):
                    with crop_cols[idx % len(crop_cols)]:
                        st.image(c_img, caption=f"{c_name.capitalize()} ({c_score:.0%})", use_container_width=True)

# ==============================================================================
# Initialization
# ==============================================================================
if "image_history" not in st.session_state: st.session_state.image_history =[]
if "active_index" not in st.session_state: st.session_state.active_index = 0
if "last_file_id" not in st.session_state: st.session_state.last_file_id = None
if "new_upload_pending" not in st.session_state: st.session_state.new_upload_pending = False
if "auto_analyse" not in st.session_state: st.session_state.auto_analyse = False
if "upload_counter" not in st.session_state: st.session_state.upload_counter = 0
if "last_camera_id" not in st.session_state: st.session_state.last_camera_id = None
if "camera_counter" not in st.session_state: st.session_state.camera_counter = 0

# ==============================================================================
# Top Navigation Bar (Antigravity Style)
# ==============================================================================
st.markdown("""
<div class="ag-card !p-4 mb-6 flex items-center justify-between">
    <div class="flex items-center gap-4">
        <div class="flex gap-1.5 p-2 rounded-full bg-[var(--ag-border)] shadow-inner">
            <div class="w-2.5 h-2.5 rounded-full bg-blue-500 animate-pulse"></div>
            <div class="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" style="animation-delay: 150ms;"></div>
            <div class="w-2.5 h-2.5 rounded-full bg-amber-500 animate-pulse" style="animation-delay: 300ms;"></div>
        </div>
        <h1 class="text-xl md:text-2xl font-extrabold tracking-tight text-[var(--ag-text)]">Zapfan <span class="text-[var(--ag-accent)] font-light">Smart Cashier</span></h1>
    </div>
    <div class="hidden md:block px-4 py-1.5 rounded-full border border-[var(--ag-border)] bg-[var(--ag-border)] text-xs font-semibold uppercase tracking-widest text-[var(--ag-text-muted)]">
        AI Detection Engine
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# Sidebar
# ==============================================================================
with st.sidebar:
    st.markdown('<div class="text-xs font-bold uppercase tracking-widest text-[var(--ag-text-muted)] mb-3">System Settings</div>', unsafe_allow_html=True)
    
    st.toggle("Next-Gen Dark Mode", value=st.session_state.dark_mode, on_change=toggle_theme, key="theme_toggle")

    MODEL_OPTIONS =["YOLO (Fast)", "RT-DETR (Transformer)", "Faster R-CNN (Accurate)"]
    model_option = st.selectbox("Neural Model", options=MODEL_OPTIONS, index=0)
    compare_all = st.checkbox("Multi-Model Matrix")

    st.markdown('<hr style="border-color: var(--ag-border); margin: 2rem 0;">', unsafe_allow_html=True)

    st.markdown('<div class="text-xs font-bold uppercase tracking-widest text-[var(--ag-text-muted)] mb-3">Current Pricing</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ag-card !p-3 mb-3 flex items-center gap-4">
        <div class="w-10 h-10 rounded-full flex items-center justify-center bg-red-100 text-xl shadow-inner border border-red-200">🥩</div>
        <div class="flex-1">
            <div class="text-sm font-bold text-[var(--ag-text)] tracking-tight">Meat Construct</div>
            <div class="text-xs font-medium text-[var(--ag-accent)]">RM 4.00</div>
        </div>
    </div>
    <div class="ag-card !p-3 mb-3 flex items-center gap-4">
        <div class="w-10 h-10 rounded-full flex items-center justify-center bg-slate-100 text-xl shadow-inner border border-slate-200">🍚</div>
        <div class="flex-1">
            <div class="text-sm font-bold text-[var(--ag-text)] tracking-tight">Carbohydrate Base</div>
            <div class="text-xs font-medium text-[var(--ag-accent)]">RM 1.50</div>
        </div>
    </div>
    <div class="ag-card !p-3 mb-3 flex items-center gap-4">
        <div class="w-10 h-10 rounded-full flex items-center justify-center bg-emerald-100 text-xl shadow-inner border border-emerald-200">🥬</div>
        <div class="flex-1">
            <div class="text-sm font-bold text-[var(--ag-text)] tracking-tight">Fibrous Matter</div>
            <div class="text-xs font-medium text-[var(--ag-accent)]">RM 2.00</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Volume scaling: **S** ×0.7 | **M** ×1.0 | **L** ×1.5")

    if st.session_state.image_history:
        st.markdown('<hr style="border-color: var(--ag-border); margin: 2rem 0;">', unsafe_allow_html=True)
        st.markdown('<div class="text-xs font-bold uppercase tracking-widest text-[var(--ag-text-muted)] mb-3">Data Stream</div>', unsafe_allow_html=True)
        history_labels =[f"Capture {i+1}: {name}" for i, (name, _) in enumerate(st.session_state.image_history)]
        default_idx = min(st.session_state.active_index, len(history_labels) - 1)
        selected_hist = st.radio(
            "Select sequence:", options=range(len(history_labels)), format_func=lambda i: history_labels[i],
            index=default_idx, key=f"history_radio_{st.session_state.upload_counter}", label_visibility="collapsed"
        )
        if not st.session_state.new_upload_pending: st.session_state.active_index = selected_hist
        else: st.session_state.new_upload_pending = False

        if st.button("Purge Memory", use_container_width=True):
            st.session_state.image_history =[]
            st.session_state.active_index = 0
            st.rerun()

# ==============================================================================
# Main Work Area
# ==============================================================================
input_tab_upload, input_tab_camera = st.tabs(["Link Data Source", "Live Optics"])

with input_tab_upload:
    uploaded_file = st.file_uploader("Establish uplink via file upload", type=["jpg", "jpeg", "png"], key="uploader")
    if uploaded_file is not None:
        current_file_id = uploaded_file.file_id
        if current_file_id != st.session_state.last_file_id:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                base_name = uploaded_file.name
                existing_names =[n for n, _ in st.session_state.image_history]
                name = base_name
                counter = 2
                while name in existing_names:
                    dot = base_name.rfind('.')
                    name = f"{base_name[:dot]} ({counter}){base_name[dot:]}" if dot != -1 else f"{base_name} ({counter})"
                    counter += 1
                st.session_state.image_history.append((name, img_rgb))
                st.session_state.active_index = len(st.session_state.image_history) - 1
                st.session_state.last_file_id = current_file_id
                st.session_state.new_upload_pending = True
                st.session_state.auto_analyse = True
                st.session_state.upload_counter += 1
                st.rerun()

with input_tab_camera:
    st.markdown('<p class="text-sm text-[var(--ag-text-muted)] mb-2">Initialize optical sensor array.</p>', unsafe_allow_html=True)
    camera_photo = st.camera_input("Optical Uplink", key="camera_input", label_visibility="collapsed")
    if camera_photo is not None:
        current_camera_id = camera_photo.file_id
        if current_camera_id != st.session_state.last_camera_id:
            cam_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
            cam_bgr = cv2.imdecode(cam_bytes, cv2.IMREAD_COLOR)
            if cam_bgr is not None:
                cam_rgb = cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2RGB)
                st.session_state.camera_counter += 1
                base_name = f"Optical Node {st.session_state.camera_counter}"
                existing_names =[n for n, _ in st.session_state.image_history]
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

# ==============================================================================
# Active Interface & Analysis Processing
# ==============================================================================
if st.session_state.image_history:
    st.markdown('<hr style="border-color: var(--ag-border); margin: 2rem 0;">', unsafe_allow_html=True)
    active_name, img_rgb = st.session_state.image_history[st.session_state.active_index]

    st.markdown(f'<div class="text-xs font-bold uppercase tracking-widest text-[var(--ag-text-muted)] mb-3">Active Node: {active_name}</div>', unsafe_allow_html=True)
    st.image(img_rgb, use_container_width=True)

    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])
    with btn_col1:
        run_analysis = st.button("Execute Deep Scan", type="primary", use_container_width=True)
    with btn_col2:
        remove_current = st.button("Eject", use_container_width=True)
    with btn_col3:
        upload_new = st.button("Standby", use_container_width=True)

    if remove_current:
        st.session_state.image_history.pop(st.session_state.active_index)
        st.session_state.active_index = max(0, st.session_state.active_index - 1) if st.session_state.image_history else 0
        st.rerun()

    should_analyse = run_analysis or st.session_state.auto_analyse
    if st.session_state.auto_analyse:
        st.session_state.auto_analyse = False

    if should_analyse:
        st.markdown('<hr style="border-color: var(--ag-border); margin: 2rem 0;">', unsafe_allow_html=True)
        if compare_all:
            st.markdown('<h3 class="text-xl font-bold text-[var(--ag-text)] tracking-tight mb-4">Neural Matrix Comparison</h3>', unsafe_allow_html=True)
            tabs = st.tabs([m.split(" (")[0] for m in MODEL_OPTIONS])
            for tab, m_opt in zip(tabs, MODEL_OPTIONS):
                analyse_and_display(img_rgb, m_opt, container=tab)
        else:
            analyse_and_display(img_rgb, model_option)

else:
    st.markdown("""
    <div class="ag-card flex flex-col items-center justify-center py-16 px-4 mt-8">
        <div class="w-20 h-20 rounded-full bg-[var(--ag-border)] shadow-inner flex items-center justify-center text-4xl mb-6 animate-pulse">📡</div>
        <h3 class="text-2xl font-bold text-[var(--ag-text)] tracking-tight mb-2">Awaiting Input Signal</h3>
        <p class="text-[var(--ag-text-muted)] text-center max-w-md">Provide visual telemetry via uplink (File) or optical sensor (Camera) to initialize the AI analysis module.</p>
    </div>
    """, unsafe_allow_html=True)