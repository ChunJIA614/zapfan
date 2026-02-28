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
)

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
    """Run inference using an Ultralytics model (YOLO / RT-DETR)."""
    results = model.predict(img_rgb, conf=0.25, verbose=False)
    valid_foods = []
    plate_box = None
    highest_plate_conf = 0

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            task_name = CLASSES[cls_id]

            if task_name == 'plate':
                if conf > highest_plate_conf:
                    highest_plate_conf = conf
                    plate_box = (x1, y1, x2, y2)
            else:
                valid_foods.append({
                    'name': task_name,
                    'box': (x1, y1, x2, y2),
                    'score': conf,
                })

    return valid_foods, plate_box, highest_plate_conf


def run_frcnn(model, img_rgb):
    """Run inference using Faster R-CNN (PyTorch)."""
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
    plate_box = None
    highest_plate_conf = 0

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cls_id = labels[i] - 1  # Shift for background class
        task_name = CLASSES[cls_id]
        conf = scores[i]

        if task_name == 'plate':
            if conf > highest_plate_conf:
                highest_plate_conf = conf
                plate_box = (x1, y1, x2, y2)
        else:
            valid_foods.append({
                'name': task_name,
                'box': (x1, y1, x2, y2),
                'score': conf,
            })

    return valid_foods, plate_box, highest_plate_conf


def draw_results(img_rgb, valid_foods, plate_box, highest_plate_conf, plate_area):
    """Draw bounding boxes, labels, and prices on the image. Returns annotated image, receipt lines, total bill, and cropped images."""
    img_display = img_rgb.copy()
    img_clean = img_rgb.copy()
    total_bill = 0.0
    receipt_lines = []
    cropped_images = []

    # Draw plate
    if plate_box is not None:
        px1, py1, px2, py2 = plate_box
        cv2.rectangle(img_display, (px1, py1), (px2, py2), COLORS['plate'], 3)
        cv2.putText(img_display, f"Plate ({highest_plate_conf:.2f})",
                    (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['plate'], 2)
    else:
        # Draw estimated plate boundary
        min_x = min(d['box'][0] for d in valid_foods)
        min_y = min(d['box'][1] for d in valid_foods)
        max_x = max(d['box'][2] for d in valid_foods)
        max_y = max(d['box'][3] for d in valid_foods)
        cv2.rectangle(img_display, (min_x, min_y), (max_x, max_y), (255, 255, 255), 2, cv2.LINE_AA)

    # Draw food items
    for item in valid_foods:
        task_name = item['name']
        conf = item['score']
        x1, y1, x2, y2 = item['box']
        box_area = (x2 - x1) * (y2 - y1)
        size_label = get_portion_size(box_area, plate_area)

        multiplier = SIZE_MULTIPLIERS[size_label]
        final_price = BASE_PRICES[task_name] * multiplier
        total_bill += final_price

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
            cropped_images.append((task_name, conf, cropped_part))

        # Draw bounding box
        color = COLORS[task_name]
        rgb = (color[2], color[1], color[0])
        cv2.rectangle(img_display, (x1, y1), (x2, y2), rgb, 3)

        label = f"{task_name.capitalize()} ({size_label}) {CURRENCY}{final_price:.2f}"
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img_display, (x1, max(y1 - t_size[1] - 10, 0)),
                      (x1 + t_size[0], max(y1, 10)), rgb, -1)
        cv2.putText(img_display, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img_display, receipt_lines, total_bill, cropped_images


# ==============================================================================
# Helper: run a single model and display results in a container
# ==============================================================================
def analyse_and_display(img_rgb, model_option, container=st):
    """Load the chosen model, run inference, and render results inside *container*."""
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
        with st.spinner(f"🔍 [{model_option}] Analyzing your plate..."):
            if "Faster R-CNN" in model_option:
                valid_foods, plate_box, highest_plate_conf = run_frcnn(model, img_rgb)
            else:
                valid_foods, plate_box, highest_plate_conf = run_ultralytics(model, img_rgb)

        if not valid_foods:
            st.warning(f"[{model_option}] No food items detected. Try a different model or image.")
            return

        # Calculate plate area
        if plate_box is not None:
            px1, py1, px2, py2 = plate_box
            plate_area = max(1, (px2 - px1) * (py2 - py1))
        else:
            st.info("Plate not detected — estimating from food boundaries.")
            min_x = min(d['box'][0] for d in valid_foods)
            min_y = min(d['box'][1] for d in valid_foods)
            max_x = max(d['box'][2] for d in valid_foods)
            max_y = max(d['box'][3] for d in valid_foods)
            plate_area = max(1, (max_x - min_x) * (max_y - min_y))

        # Draw results
        img_display, receipt_lines, total_bill, cropped_images = draw_results(
            img_rgb, valid_foods, plate_box, highest_plate_conf, plate_area
        )

        # ---- Result columns ----
        col_img, col_receipt = st.columns([2, 1])

        with col_img:
            st.subheader("🔍 Detection Result")
            st.image(img_display, use_container_width=True)

        with col_receipt:
            st.subheader("🧾 Smart Receipt")
            model_label = model_option.split(" (")[0]
            st.caption(f"Model: **{model_label}**")

            for line in receipt_lines:
                st.markdown(
                    f"- **{line['item']}** ({line['size']}, {line['ratio']:.1f}%) "
                    f"— {CURRENCY}{line['price']:.2f}"
                )

            st.divider()
            st.markdown(f"### Total: {CURRENCY}{total_bill:.2f}")

        # ---- Cropped items ----
        if cropped_images:
            st.subheader("✂️ Detected Items")
            crop_cols = st.columns(min(len(cropped_images), 4))
            for idx, (c_name, c_score, c_img) in enumerate(cropped_images):
                with crop_cols[idx % len(crop_cols)]:
                    st.image(c_img, caption=f"{c_name.upper()} ({c_score:.0%})",
                             use_container_width=True)


# ==============================================================================
# Session State Initialisation
# ==============================================================================
if "image_history" not in st.session_state:
    st.session_state.image_history = []   # list of (filename, img_rgb) tuples
if "active_index" not in st.session_state:
    st.session_state.active_index = 0

# ==============================================================================
# Streamlit UI
# ==============================================================================
st.title("🍚 Zapfan Smart Cashier")
st.markdown("**AI-Powered Economy Rice (Nasi Campur) Pricing System**")
st.divider()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")

    MODEL_OPTIONS = ["YOLO (Fast)", "RT-DETR (Transformer)", "Faster R-CNN (Accurate)"]
    model_option = st.selectbox("Select AI Model", options=MODEL_OPTIONS, index=0)

    compare_all = st.checkbox("🔀 Compare all 3 models side-by-side")

    st.divider()
    st.markdown("### 💰 Price List")
    st.markdown(f"- **Meat**: {CURRENCY}4.00")
    st.markdown(f"- **Rice**: {CURRENCY}1.50")
    st.markdown(f"- **Vege**: {CURRENCY}2.00")
    st.caption("Portion multipliers: S×0.7, M×1.0, L×1.5")

    # --- Image History (quick-switch between previous uploads) ---
    if st.session_state.image_history:
        st.divider()
        st.markdown("### 🖼️ Image History")
        history_labels = [f"{i+1}. {name}" for i, (name, _) in enumerate(st.session_state.image_history)]
        selected_hist = st.radio(
            "Switch to a previous upload:",
            options=range(len(history_labels)),
            format_func=lambda i: history_labels[i],
            index=st.session_state.active_index,
            key="history_radio",
        )
        st.session_state.active_index = selected_hist

        if st.button("🗑️ Clear all history"):
            st.session_state.image_history = []
            st.session_state.active_index = 0
            st.rerun()

# --- Main Area: Image Upload ---
uploaded_file = st.file_uploader(
    "📸 Upload a picture of your mixed rice plate",
    type=["jpg", "jpeg", "png"],
    key="uploader",
)

# Process a new upload
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not read the uploaded image. Please try a different file.")
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Avoid duplicate entries for the same filename
        existing_names = [n for n, _ in st.session_state.image_history]
        if uploaded_file.name not in existing_names:
            st.session_state.image_history.append((uploaded_file.name, img_rgb))
            st.session_state.active_index = len(st.session_state.image_history) - 1

# --- Display active image & run analysis ---
if st.session_state.image_history:
    active_name, img_rgb = st.session_state.image_history[st.session_state.active_index]

    st.image(img_rgb, caption=f"Current image: {active_name}", use_container_width=True)

    # Action buttons
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        run_analysis = st.button("▶️ Analyse", type="primary", use_container_width=True)
    with btn_col2:
        remove_current = st.button("🗑️ Remove this image", use_container_width=True)
    with btn_col3:
        upload_new = st.button("📸 Upload another", use_container_width=True)

    # Handle remove
    if remove_current:
        st.session_state.image_history.pop(st.session_state.active_index)
        if st.session_state.image_history:
            st.session_state.active_index = max(0, st.session_state.active_index - 1)
        else:
            st.session_state.active_index = 0
        st.rerun()

    # Handle upload-another (just scroll to uploader)
    if upload_new:
        st.info("👆 Use the uploader above to add a new image. It will be added to your history.")

    # Run analysis
    if run_analysis:
        st.divider()
        if compare_all:
            st.subheader("🔀 Side-by-Side Model Comparison")
            tabs = st.tabs([m.split(" (")[0] for m in MODEL_OPTIONS])
            for tab, m_opt in zip(tabs, MODEL_OPTIONS):
                analyse_and_display(img_rgb, m_opt, container=tab)
        else:
            analyse_and_display(img_rgb, model_option)
else:
    st.info("👆 Upload a photo of your economy rice plate to get started!")
