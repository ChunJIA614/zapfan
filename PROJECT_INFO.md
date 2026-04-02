# 📚 ZAPFAN PROJECT DOCUMENTATION
## Mixed Rice Price Prediction using Deep Learning

---

## 📋 Academic Information

**Institution:** Tunku Abdul Rahman University of Management and Technology (TARUMT)  
**Module:** BMCS2203 - Artificial Intelligence  
**Session:** 202601 (Academic Year 2025/26)  
**Programme:** Bachelor of Information Technology (Honours) in Software Systems Development  
**Year/Semester:** Year 2, Semester 3  
**Tutorial Group:** 5  
**Module Tutor:** Ms Yan Yen Wei  
**Submission Date:** 04 March 2026  

---

## 👥 Team Composition

### Team Members

| S/N | Student Name | Student ID | Module In Charge | Signature | Date |
|-----|------|------|------|------|------|
| 1 | Leong Kai Sheng | 25WMR09840 | Faster R-CNN | ✓ Leong | 04 Mar 2026 |
| 2 | Ooi Jun Kang | 25WMR09855 | YOLOv8 | ✓ Ooi | 04 Mar 2026 |
| 3 | Chaw Chun Jia | 25WMR09815 | RT-DETR | ✓ Chaw | 04 Mar 2026 |

### Academic Integrity
All team members acknowledge:
- ✅ Submitted work is original and their own
- ✅ Work is in their own words
- ✅ Use of AI generative technology has been disclosed

---

## 🎯 Project Title & Problem Statement

**Title:** Mixed Rice Price Prediction by using Object Classification using Deep Learning

### Problem Background

"Economy rice," commonly known as *Zapfan* in Malaysia, is a staple dining option in university cafeterias due to its affordability and wide variety of choices. Customers typically select multiple dishes—such as assorted meats and vegetables—to accompany a serving of steamed rice.

#### Current Issues

1. **Manual Checkout Process Limitations**
   - Cashiers must quickly scan unstructured plates visually
   - Identify overlapping components instantly
   - Estimate portion sizes from memory
   - All within seconds during high-volume periods

2. **Pricing Inconsistency**
   - Human judgment is subjective and error-prone
   - Identical portions charged differently
   - Fatigue and bias affect fair assessment
   - Creates confusion and dissatisfaction among students

3. **Operational Bottlenecks**
   - Heavy cognitive load on cashiers
   - Significant delays during peak lunch hours
   - Reduces overall cafeteria throughput
   - Frustrates students rushing between classes

4. **Lack of Localized Solutions**
   - Unlike standardized, barcoded products
   - Economy rice features highly unstructured layouts
   - Overlapping ingredients with shared color profiles
   - No established computer vision solutions for this domain

---

## 📝 Project Objectives & Aims

### Primary Aim
To analyse and employ appropriate Artificial Intelligence (AI) techniques to design an intelligent smart cashier system capable of solving manual checkout bottlenecks and pricing inconsistencies currently experienced in university cafeterias.

### Specific Objectives

**Objective 1: Develop Computer Vision Prototype**
- Create functional image processing system using Python
- Automatically detect and isolate key meal components
- Classify items into: meat, vegetables, rice, and plate
- Enable real-time food recognition

**Objective 2: Implement & Evaluate AI Architectures**
- Deploy three distinct object detection paradigms:
  - **YOLOv8** (Single-stage CNN) — Speed optimized
  - **RT-DETR** (Vision Transformer) — Context-aware
  - **Faster R-CNN** (Two-stage CNN) — Accuracy optimized
- Rigorously compare performance metrics
- Determine most effective algorithm for unstructured visual environment

**Objective 3: Design Pricing Algorithm**
- Utilize bounding box coordinates from AI models
- Mathematically estimate portion sizes (S/M/L)
- Calculate ratios of detected food area to detected plate area
- Generate automatic, objective bills

**Objective 4: Prototype Deployment**
- Transition theoretical models to practical application
- Create interactive web interface
- Enable seamless end-user interaction

---

## 💡 Project Motivation

### Academic Motivation
The highly unstructured visual nature of mixed rice presents a uniquely challenging computer vision problem:
- Unlike retail items with barcodes
- Ingredients frequently overlap
- Similar color profiles across dishes
- Lack consistent shapes or boundaries

This complexity provides an ideal scenario to apply advanced machine learning techniques.

### Practical Motivation
- **Operational Efficiency:** Drastically reduce cognitive load on staff
- **Queue Reduction:** Minimize waiting times during peak hours
- **Fair Pricing:** Guarantee transparent, standardized pricing
- **Student Satisfaction:** Enable confidence in billing accuracy
- **Technical Excellence:** Benchmark cutting-edge AI architectures

### Business Motivation
- Addressable market: All university cafeterias, food courts, hawker stalls
- Scalable solution: Can be adapted to any food type or restaurant
- Competitive advantage: First automated solution for unstructured food pricing

---

## 🔬 Methodology

### 1. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│          Input: Food Plate Image (Upload or Camera)          │
└────────────────────────┬────────────────────────────────────┘
                         │
                    ┌────▼─────┐
                    │ AI Model  │  (YOLO / RT-DETR / Faster R-CNN)
                    └────┬─────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
      ┌───▼──┐      ┌───▼──┐      ┌───▼──┐
      │Detections │  │ Class. │   │BBoxes│
      └────┬──┘   └───┬───┘   └───┬──┘
          │            │            │
          └────────────┼────────────┘
                       │
            ┌──────────▼──────────┐
            │ Pricing Algorithm   │
            │ (Size × Base × Mult)│
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │ Receipt Generation  │
            │ (Itemized + Total)  │
            └─────────────────────┘
```

### 2. Dataset Description

**Dataset Name:** `food_dataset`

#### Classes (4)
1. **Meat** (Class ID: 0)
   - Various types of meat dishes
   - Highly variable in color, texture, shape
   
2. **Plate** (Class ID: 1)
   - Physical plate containing food
   - Critical for pricing algorithm (area denominator)
   
3. **Rice** (Class ID: 2)
   - Staple base of meal
   - Typically large, contiguous white mass
   - Can be partially occluded by other items
   
4. **Vegetable** (Class ID: 3)
   - Assorted vegetable dishes
   - Visually scattered, lack definitive boundaries

#### Annotation Formats
- **Bounding Box Format:** Standard YOLO (class_id, x_center, y_center, width, height)
- **Polygon Format:** Segmentation polygons (class_id, x1, y1, x2, y2, x3, y3, ...)

#### Data Split
- **Training:** 80% of images
- **Validation/Testing:** 20% of images
- Mixed format handling via custom data pipeline

### 3. Object Detection Architectures

#### Architecture 1: YOLOv8 (Single-Stage CNN)

**Process:**
1. **Input Image Grid Division** — Divide input into fine grid (13×13 cells)
2. **Anchor-Free Prediction** — Each cell directly predicts center (x,y), width, height, confidence
3. **Multi-Scale Output** — Process across multiple feature pyramid scales
4. **Non-Maximum Suppression** — Filter redundant overlapping boxes

**Characteristics:**
- Single forward pass through network
- Optimized for real-time inference
- Trade-off: Speed over extreme accuracy

#### Architecture 2: RT-DETR (Vision Transformer)

**Process:**
1. **CNN Backbone** — Extract initial visual features
2. **Hybrid Encoder** — Multi-scale feature fusion
3. **Transformer Encoder** — Global self-attention across entire image
4. **Object Queries** — Fixed set of learnable queries for object detection
5. **Direct Set Prediction** — Directly output refined predictions

**Characteristics:**
- Global context understanding (handles occlusions better)
- No need for NMS (native duplicate suppression)
- Better for complex, overlapping scenes
- Moderate computational cost

#### Architecture 3: Faster R-CNN (Two-Stage CNN)

**Process:**
1. **CNN Backbone** — Extract feature maps (ResNet50)
2. **Region Proposal Network (RPN)** — Generate ~300 region proposals
3. **RoI Pooling** — Extract features from each proposed region
4. **Classification Head** — Classify and refine box coordinates

**Characteristics:**
- Two-stage refinement process
- Prioritizes accuracy over speed
- High computational cost
- Excellent spatial localization

---

## 📊 Performance Results

### Phase 1: YOLOv8 vs RT-DETR

#### YOLOv8 Metrics
- **mAP@50:** 0.8649 (86.49%)
- **Inference Speed:** 5.26 ms/image
- **Key Insight:** Fastest model; slight accuracy trade-off

#### RT-DETR Metrics
- **mAP@50:** 0.9596 (95.96%)
- **Inference Speed:** 38.15 ms/image
- **Key Insight:** Significant accuracy improvement; handles overlapping food well

#### Phase 1 Winner
**RT-DETR** selected due to:
- 11% higher accuracy (0.9596 vs 0.8649)
- Superior handling of complex, overlapping scenarios
- More suitable for varied real-world cafeteria environments

### Phase 2: RT-DETR vs Faster R-CNN

#### RT-DETR Metrics (Phase 1 Winner)
- **mAP@50:** 0.9596 (95.96%)
- **Inference Speed:** 38.15 ms/image

#### Faster R-CNN Metrics
- **mAP@50:** 0.9950 (99.50%)
- **Inference Speed:** 92.12 ms/image
- **Key Insight:** Near-perfect accuracy; 2.4× slower

#### Phase 2 Analysis
- Faster R-CNN achieves highest accuracy (+3.54% absolute)
- But 2.4× slower inference time (38.15 ms → 92.12 ms)
- For cafeteria use case, speed matters alongside accuracy

### Final Model Selection: RT-DETR

**Rationale:**
✅ Excellent accuracy (95.96%) for fair pricing  
✅ Reasonable speed (38 ms) for cafeteria throughput  
✅ Global self-attention handles overlapping items well  
✅ No GPU requirement for deployment (can run on modern CPUs)  
✅ Best practical balance for real-world deployment  

---

## 💰 Pricing Algorithm

### Algorithm Overview
```
FOR each detected food item:
    1. Calculate bbox area (x2-x1) × (y2-y1)
    2. Calculate image area
    3. Compute area fraction = bbox_area / image_area
    4. Estimate portion size:
       - If area_fraction < 0.04 → Size = "S" (Small)
       - If 0.04 ≤ area_fraction < 0.10 → Size = "M" (Medium)
       - If area_fraction ≥ 0.10 → Size = "L" (Large)
    5. Calculate price = base_price × multiplier[size]
    
TOTAL = SUM of all food item prices
```

### Configuration

#### Base Prices (RM - Malaysian Ringgit)
- **Meat:** RM 4.00
- **Rice:** RM 1.50
- **Vegetable:** RM 2.00
- **Plate:** RM 0.00 (no charge)

#### Size Multipliers
- **Small (S):** 0.7× (< 4% of image)
- **Medium (M):** 1.0× (4-10% of image)
- **Large (L):** 1.5× (> 10% of image)

#### Example Calculation
```
Scenario: 1 meat (M), 1 rice (L), 1 vegetable (S)

Meat (Medium):     RM 4.00 × 1.0 = RM 4.00
Rice (Large):      RM 1.50 × 1.5 = RM 2.25
Vegetable (Small): RM 2.00 × 0.7 = RM 1.40
                                    ─────────
SUBTOTAL:                           RM 7.65
```

---

## 🎯 Evaluation Metrics

### Precision
- **Definition:** Out of all boxes labeled as "Meat", how many actually are Meat?
- **Formula:** TP / (TP + FP)
- **Purpose:** Measure false positive rate

### Recall
- **Definition:** Out of all actual Meat on plate, how many did model detect?
- **Formula:** TP / (TP + FN)
- **Purpose:** Measure missed detections

### Mean Average Precision (mAP@50)
- **Definition:** Average precision across all classes at IoU threshold of 50%
- **Formula:** Average of per-class AP values
- **Purpose:** Standard metric for object detection
- **Range:** 0 - 1 (higher is better)

### Inference Speed (FPS / ms per image)
- **Definition:** How fast can the model process one image?
- **Metric:** Milliseconds per image or Frames Per Second (FPS)
- **Purpose:** Critical for real-time cafeteria deployment

---

## 🔧 Technologies & Tools Used

### Software Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| **Ultralytics** | Latest | YOLO & RT-DETR training/inference |
| **PyTorch** | 1.12+ | Deep learning framework (Faster R-CNN) |
| **Torchvision** | 0.13+ | Pre-trained models and utilities |
| **OpenCV (cv2)** | 4.5+ | Image processing and visualization |
| **Streamlit** | 1.20+ | Web application framework |
| **NumPy** | 1.22+ | Numerical computations |
| **Pandas** | 1.4+ | Data manipulation |

### Hardware
- **GPU:** NVIDIA T4 (for training)
- **CPU:** Compatible with modern processors
- **Memory:** 8GB+ recommended

### Development Environment
- **IDE:** Google Colab (training), VS Code (development)
- **Version Control:** Git
- **Deployment:** Streamlit Cloud

---

## 📂 Deliverables

### Code Deliverables
✅ `app.py` — Complete Streamlit web application  
✅ `findingrice.py` — Training scripts for all three models  
✅ Trained model weights (.pt, .pth files)  

### Documentation Deliverables
✅ `README.md` — Comprehensive project guide  
✅ `PROJECT_INFO.md` — This file (detailed specifications)  
✅ Assignment Report (PDF) — Full academic documentation  

### Training Results
✅ Training metrics plots (per model)  
✅ Confusion matrices  
✅ Precision-Recall curves  
✅ F1 score curves  

---

## 🚀 Deployment

### Local Deployment
```bash
cd zapfan
pip install -r requirements.txt
streamlit run app.py
```

### Cloud Deployment
Deployed on Streamlit Cloud at:  
https://zapfan-p8jmmy9i5anpwakoxdmku2.streamlit.app/

### System Requirements
- **OS:** Windows, macOS, Linux
- **Python:** 3.9 or higher
- **Disk Space:** 2GB (including models)
- **RAM:** 4GB minimum (8GB recommended)
- **Internet:** Required for first-time setup and cloud deployment

---

## 📈 Key Achievements

### ✅ Research & Algorithm Development
- Rigorous comparison of three cutting-edge object detection paradigms
- Empirical validation of theoretical trade-offs
- Custom data engineering for mixed annotation formats

### ✅ Implementation Excellence
- Unified training pipeline supporting multiple architectures
- Spatial pricing algorithm with intelligent sizing
- Efficient inference with caching mechanisms

### ✅ Practical Deployment
- Production-ready Streamlit web application
- Interactive model selection and comparison
- Real-time image processing with visual feedback

### ✅ Academic Rigor
- Comprehensive testing methodology
- Detailed performance evaluation
- Clear documentation of results and limitations

---

## 🔮 Future Enhancement Roadmap

### Short Term (Months 1-2)
- [ ] Expand dataset to include more food types
- [ ] Fine-tune models with additional training data
- [ ] Optimize inference time for faster processing

### Medium Term (Months 3-6)
- [ ] Implement instance segmentation (pixel-level masks)
- [ ] Add 3D depth sensing capabilities
- [ ] Deploy model quantization for edge devices

### Long Term (Months 6-12)
- [ ] Integration with actual cafeteria POS systems
- [ ] Active learning pipeline for continuous improvement
- [ ] Multi-language support for international deployment
- [ ] Payment integration (e-wallet, card systems)

---

## ⚠️ Limitations & Constraints

### Current Limitations

**1. Dataset Scope**
- Limited to 4 basic food categories
- Real-world restaurants have dozens of distinct dishes
- Many dishes share similar visual textures and colors

**2. 2D Bounding Box Inaccuracy**
- Food is rarely perfectly rectangular
- Diagonal items generate large empty-space bounding boxes
- Results in portion size overestimation

**3. No Volumetric Awareness**
- Cannot detect depth or height of food
- Stacked vs. flat portions appear identical
- Fairness issue for customers

**4. Hardware Constraints**
- Slower models require GPU for cafeteria-speed performance
- CPU-only environments experience unacceptable delays
- Not affordable for all restaurants

### Mitigation Strategies
- Future instance segmentation for precise shapes
- RGB-D cameras for volumetric calculation
- Model optimization (quantization) for low-cost deployment
- Active learning for continuous dataset expansion

---

## 📚 References

1. Jocher, G., Chaurasia, A., & Qiu, J. (2023). YOLO by Ultralytics (Version 8.0.0). GitHub.
2. Lv, W., Xu, S., Zhao, Y., et al. (2023). DETRs Beat YOLOs on Real-time Object Detection. arXiv preprint.
3. Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN. Advances in Neural Information Processing Systems, 28.
4. Lin, T. Y., Maire, M., Belongie, S., et al. (2014). Microsoft COCO: Common Objects in Context. ECCV 2014.
5. Streamlit. (2024). Streamlit Documentation. Retrieved from https://docs.streamlit.io/

---

<div align="center">

**🍚 Zapfan Smart Cashier**

TARUMT BMCS2203 Artificial Intelligence  
Academic Year 2025/26, Session 202601

*Transforming Cafeteria Checkout with AI*

</div>
