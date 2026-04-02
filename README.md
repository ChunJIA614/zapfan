# 🍚 Zapfan Smart Cashier
## Mixed Rice Price Prediction using Deep Learning

**Academic Project** | BMCS2203 Artificial Intelligence | TARUMT | Tutorial Group 5

---

## 📋 Project Overview

Zapfan (Economy Rice) is a staple cafeteria meal, but manual checkout and pricing is subjective and error-prone. This project automates the entire process using **AI-powered object detection** and intelligent pricing algorithms.

### Problem Statement
- 🔴 **Inconsistent Pricing** — Identical portions charged differently based on cashier judgment
- ⏱️  **Operational Bottlenecks** — Long queues during peak hours
- 😞 **Student Dissatisfaction** — Lack of transparent, standardized pricing

### Solution
A complete **end-to-end computer vision pipeline** that:
1. **Detects** food items (meat, rice, vegetables) on a plate using AI
2. **Estimates** portion sizes based on visual area ratios
3. **Calculates** fair, standardized pricing automatically
4. **Deploys** as an interactive web application

---

## 👥 Team Members

| # | Name | Student ID | Role | Signature |
|---|------|-----------|------|-----------|
| 1 | **Leong Kai Sheng** | 25WMR09840 | Faster R-CNN | ✓ |
| 2 | **Ooi Jun Kang** | 25WMR09855 | YOLOv8 | ✓ |
| 3 | **Chaw Chun Jia** | 25WMR09815 | RT-DETR | ✓ |

**Tutor:** Ms Yan Yen Wei  
**Date:** 04 March 2026  

---

## 🎯 Project Objectives

### Primary Aim
To analyse and employ appropriate AI techniques to design an intelligent smart cashier system capable of solving manual checkout bottlenecks and pricing inconsistencies in university cafeterias.

### Specific Objectives
1. **Develop** a functional image processing and computer vision prototype using Python
2. **Implement & evaluate** three distinct AI object detection architectures:
   - YOLOv8 (Single-Stage CNN)
   - RT-DETR (Vision Transformer)
   - Faster R-CNN (Two-Stage CNN)
3. **Design** a programmatic pricing algorithm using bounding box dimensions to estimate portion sizes
4. **Deploy** as an interactive web application

---

## 🏆 Model Performance Comparison

### Final Results

| Model | mAP@50 | Inference Speed | Best For |
|-------|--------|-----------------|----------|
| **YOLOv8** | 0.8649 | 5.26 ms/img | ⚡ Speed & Edge Deployment |
| **RT-DETR** | **0.9596** | 38.15 ms/img | ⚖️ **Balanced (RECOMMENDED)** |
| **Faster R-CNN** | **0.9950** | 92.12 ms/img | 🎯 Maximum Accuracy |

#### Model Analysis

**YOLOv8 (Single-Stage CNN)**
- ✅ Extremely fast inference (~5 ms)
- ✅ Ideal for real-time, low-compute environments
- ⚠️ Slightly lower accuracy on complex occlusions

**RT-DETR (Vision Transformer)** ⭐ **SELECTED FOR DEPLOYMENT**
- ✅ Excellent balance of speed & accuracy
- ✅ Handles overlapping food items well (global self-attention)
- ✅ Reasonable inference time for cafeteria use
- ⚠️ Moderate computational requirement

**Faster R-CNN (Two-Stage CNN)**
- ✅ Near-perfect accuracy (99.50% mAP)
- ✅ Highest spatial localization precision
- ⚠️ Slow inference (92 ms per image)
- ⚠️ Requires GPU support

---

## 🔧 Technologies & Tools

### Core Libraries
- **Ultralytics** — YOLOv8 & RT-DETR training and inference
- **PyTorch & Torchvision** — Faster R-CNN custom implementation
- **OpenCV (cv2)** — Image processing, visualization, and annotation
- **Streamlit** — Interactive web application framework
- **NumPy & Pandas** — Data processing

### Hardware
- **GPU:** NVIDIA T4 (for training)
- **Framework:** Python 3.9+

### Development Environment
- Google Colab (training)
- VS Code (development)
- Streamlit Cloud (deployment)

---

## 📊 Dataset

**Dataset Name:** `food_dataset`

### Classes (4)
1. 🥩 **Meat** (Class ID: 0) — Various meat dishes
2. 🍚 **Rice** (Class ID: 2) — Staple base
3. 🥬 **Vegetable** (Class ID: 3) — Assorted vegetables
4. 🍽️ **Plate** (Class ID: 1) — Physical plate (for size reference)

### Annotation Formats
- **Standard Bounding Boxes:** YOLO format (x_center, y_center, width, height)
- **Segmentation Polygons:** Precise pixel outlines (x1, y1, x2, y2, ...)

### Data Split
- **Training:** 80% of images
- **Validation:** 20% of images

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd zapfan

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### 3. Navigate the App

- **About Tab:** Project information, team members, and performance metrics
- **Smart Checkout Tab:** Upload food images, select model, get instant pricing
- **Training Results Tab:** View training metrics and evaluation curves

### 4. Upload an Image

- Use the file uploader or camera to capture a plate
- Select your preferred AI model
- Click "Analyse" to get instant pricing
- View detailed receipt and cropped item previews

---

## 💰 Pricing Configuration

Base prices (configurable in `CONFIG` dictionary):
- 🥩 **Meat:** RM 4.00 per serving
- 🍚 **Rice:** RM 1.50 per serving
- 🥬 **Vegetable:** RM 2.00 per serving
- 🍽️ **Plate:** RM 0.00 (no charge)

### Portion Size Multipliers
- **S (Small):** 0.7× (< 4% of plate area)
- **M (Medium):** 1.0× (4%-10% of plate area)
- **L (Large):** 1.5× (> 10% of plate area)

---

## 📁 Project Structure

```
zapfan/
├── app.py                              # Streamlit web application
├── findingrice.py                      # Training scripts (YOLOv8, RT-DETR, Faster R-CNN)
├── model_mixed_rice.pt                 # YOLOv8 trained weights
├── model_rtdetr_mixed_rice.pt          # RT-DETR trained weights
├── faster_rcnn_mixed_rice.pth          # Faster R-CNN trained weights
├── requirements.txt                    # Python dependencies
├── packages.txt                        # System packages
├── README.md                           # This file
├── training_results/                   # Training metrics and curves
│   ├── yolo/                           # YOLOv8 training plots
│   ├── rtdetr/                         # RT-DETR training plots
│   └── frcnn/                          # Faster R-CNN training plots
└── food_dataset.zip                    # Training dataset (compressed)
```

---

## 🎓 Key Achievements

✅ **Rigorous Multi-Architecture Benchmarking**
- Successfully implemented three distinct paradigms of object detection
- Empirically validated theoretical trade-offs between architectures
- Provided practical guidance for deployment scenarios

✅ **Custom Data Engineering**
- Seamless translation of YOLO polygon format to PyTorch tensor requirements
- Robust handling of mixed annotation formats
- Efficient data pipeline for training

✅ **Spatial Pricing Algorithm**
- Intelligent portion size estimation based on bbox area ratios
- Automated price calculation with multiplier logic
- Fair, transparent pricing without human bias

✅ **End-to-End Prototype Deployment**
- Functional Streamlit web application
- Real-time inference with model selection
- Interactive image upload and analysis
- Receipt generation and item visualization

---

## 📈 Results & Discussion

### Phase 1: YOLOv8 vs RT-DETR
- RT-DETR achieved 11% higher accuracy (0.9596 vs 0.8649)
- YOLOv8 was 7× faster (5.26 ms vs 38.15 ms)
- **Winner:** RT-DETR selected for its superior handling of overlapping items

### Phase 2: RT-DETR vs Faster R-CNN
- Faster R-CNN achieved near-perfect accuracy (0.9950)
- RT-DETR was 2.4× faster (38.15 ms vs 92.12 ms)
- **Winner:** RT-DETR selected for balanced performance and practical deployment

### Performance Trade-off Analysis

| Deployment Scenario | Recommended Model | Reason |
|-------|---|---|
| **High-Volume Cafeteria** | YOLOv8 | Maximum throughput, acceptable accuracy |
| **Standard Cafeteria** | RT-DETR | Best balance of speed, accuracy, and cost |
| **Premium/Audit-Critical** | Faster R-CNN | Maximum accuracy for compliance |

---

## 🔮 Future Enhancements

### 1. Instance Segmentation
- Replace bounding boxes with pixel-level masks (YOLOv8-Seg, Mask R-CNN)
- Eliminate "empty space" calculation error
- More precise portion sizing

### 2. 3D Depth Sensing
- Upgrade to RGB-D camera (Intel RealSense)
- Calculate actual physical volume of food
- Price based on mass rather than 2D area

### 3. Edge AI Optimization
- Model quantization (32-bit to 8-bit)
- NVIDIA TensorRT optimization
- Deployment on Raspberry Pi / Jetson Nano
- Affordable, low-cost hardware support

### 4. Active Learning Pipeline
- Human-in-the-loop correction system
- Automated model retraining on corrected data
- Continuous adaptation to new dishes and recipes

---

## 📝 Limitations

### Current Constraints
1. **Limited Dataset Scope** — Only 4 basic food categories; real restaurants have dozens of dishes
2. **2D Bounding Box Inaccuracy** — Food is rarely rectangular; diagonal items cause overestimation
3. **No Volumetric Awareness** — Cannot detect depth; stacked vs. flat portions identical
4. **Hardware Requirements** — GPU necessary for optimal performance; CPU deployment too slow

### Being Addressed
- Expansion to instance segmentation for precise shapes
- Integration of depth sensing for volumetric calculation
- Model optimization for low-cost edge devices

---

## 📚 References & Sources

1. **Jocher, G., Chaurasia, A., & Qiu, J. (2023).** YOLO by Ultralytics (Version 8.0.0)  
   https://github.com/ultralytics/ultralytics

2. **Lv, W., et al. (2023).** DETRs Beat YOLOs on Real-time Object Detection  
   arXiv preprint arXiv:2304.08069

3. **Ren, S., He, K., Girshick, R., & Sun, J. (2015).** Faster R-CNN: Towards Real-Time Object Detection  
   Advances in Neural Information Processing Systems, 28

4. **Lin, T. Y., et al. (2014).** Microsoft COCO: Common Objects in Context  
   European Conference on Computer Vision, 740-755

5. **Streamlit. (2024).** Streamlit Documentation  
   https://docs.streamlit.io/

---

## 🔗 Links

- **Live Demo:** https://zapfan-p8jmmy9i5anpwakoxdmku2.streamlit.app/
- **GitHub (Code):** [Project Repository]
- **Academic Documentation:** Assignment Report (PDF)

---

## 📞 Contact & Questions

For questions about this project, contact the team members or refer to the **About** tab within the Streamlit application.

---

<div align="center">

**🍚 Zapfan Smart Cashier © 2026**

TARUMT BMCS2203 Artificial Intelligence  
Tutorial Group 5 | Year 2, Semester 3

*Powered by Streamlit, PyTorch, and OpenCV*

</div>
