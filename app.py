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
        with st.spinner(f"🔍 [{model_option}] Analyzing your plate(s)..."):
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

        # ---- Result columns ----
        col_img, col_receipt = st.columns([2, 1])

        with col_img:
            st.subheader("🔍 Detection Result")
            st.image(img_display, use_container_width=True)
            if num_plates > 1:
                st.info(f"🍽️ **{num_plates} plates** detected — prices calculated separately.")

        with col_receipt:
            st.subheader("🧾 Smart Receipt")
            model_label = model_option.split(" (")[0]
            st.caption(f"Model: **{model_label}**")

            for pr in plate_results:
                if num_plates > 1:
                    st.markdown(f"#### 🍽️ Plate {pr['plate_num']}")

                for line in pr['receipt_lines']:
                    st.markdown(
                        f"- **{line['item']}** ({line['size']}, {line['ratio']:.1f}%) "
                        f"— {CURRENCY}{line['price']:.2f}"
                    )

                if num_plates > 1:
                    st.markdown(f"**Subtotal: {CURRENCY}{pr['total']:.2f}**")
                    st.divider()

            if num_plates > 1:
                st.markdown(f"### 🧮 Grand Total: {CURRENCY}{grand_total:.2f}")
            else:
                st.divider()
                st.markdown(f"### Total: {CURRENCY}{grand_total:.2f}")

        # ---- Cropped items ----
        if all_cropped:
            st.subheader("✂️ Detected Items")
            crop_cols = st.columns(min(len(all_cropped), 4))
            for idx, (c_name, c_score, c_img) in enumerate(all_cropped):
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
        # Clamp active_index to valid range
        default_idx = min(st.session_state.active_index, len(history_labels) - 1)
        selected_hist = st.radio(
            "Switch to a previous upload:",
            options=range(len(history_labels)),
            format_func=lambda i: history_labels[i],
            index=default_idx,
            key=f"history_radio_{st.session_state.upload_counter}",
        )
        # Only update active_index from the radio when there is no pending new upload
        if not st.session_state.new_upload_pending:
            st.session_state.active_index = selected_hist
        else:
            st.session_state.new_upload_pending = False

        if st.button("🗑️ Clear all history"):
            st.session_state.image_history = []
            st.session_state.active_index = 0
            st.rerun()

# --- Main Area: Image Input ---
input_tab_upload, input_tab_camera = st.tabs(["📁 Upload Image", "📷 Take Photo"])

with input_tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a picture of your mixed rice plate",
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
    st.markdown("Take a photo using your device camera. The image will be processed immediately.")
    camera_photo = st.camera_input(
        "Point your camera at the plate and click to capture",
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

    # Run analysis (manual button or automatic on new upload)
    should_analyse = run_analysis or st.session_state.auto_analyse
    if st.session_state.auto_analyse:
        st.session_state.auto_analyse = False
    if should_analyse:
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
