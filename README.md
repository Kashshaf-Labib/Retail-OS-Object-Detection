<p align="center">
  <h1 align="center">ShelfVision AI</h1>
  <p align="center">
    <strong>End-to-End Retail Shelf Detection with MLOps</strong><br>
    YOLOv8m | YOLOv11m | 76 SKU Classes | FastAPI | Docker | Airflow | CI/CD
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8m-Ultralytics-purple?logo=pytorch" alt="YOLO">
  <img src="https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-Compose-blue?logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/Airflow-2.8-red?logo=apacheairflow" alt="Airflow">
  <img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?logo=githubactions" alt="CI/CD">
</p>

<p align="center">
  <a href="https://drive.google.com/drive/folders/1kBSzd2xSj-QzwDbvjVIPaYZ6VpFTt6lI"><strong>Google Drive: All Training Runs, Model Weights & Outputs</strong></a>
</p>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
  - [With Docker (Recommended)](#option-a-docker-recommended)
  - [Without Docker](#option-b-without-docker-manual-setup)
- [Dataset Analysis](#dataset-analysis)
- [Model Training Strategy and Evolution](#model-training-strategy-and-evolution)
  - [Phase 1: Architecture and Hyperparameter Exploration](#phase-1-architecture-and-hyperparameter-exploration)
  - [Phase 2: Data-Centric Improvements](#phase-2-data-centric-improvements-stratified-split)
- [Data Preprocessing and Rebalancing](#data-preprocessing-and-rebalancing)
- [API Reference](#api-reference)
- [Frontend Dashboard](#frontend-dashboard)
- [MLOps Pipeline](#mlops-pipeline)
- [Data Drift Monitoring and Auto-Retraining](#data-drift-monitoring-and-auto-retraining)
- [CI/CD Pipeline](#cicd-pipeline)
- [Project Structure](#project-structure)
- [Full Experiment Log](#full-experiment-log)

---

## Project Overview

**ShelfVision AI** is a production-grade retail shelf product detection system that identifies and counts 76 unique product SKUs from store shelf images. The project delivers:

1. **Object Detection** -- YOLOv8m/YOLOv11m models trained for detecting fine-grained retail products
2. **Share of Shelf Analytics** -- Computes percentage distribution of each SKU for shelf space analysis
3. **Data Drift Monitoring** -- PSI-based drift detection with automatic retraining triggers
4. **Full MLOps Pipeline** -- Docker, CI/CD, Airflow orchestration, experiment tracking, and model versioning

### Key Results (Best Model: YOLOv8m Stratified + Oversample)

| Metric | Value |
|--------|-------|
| **Precision** | 81.15% |
| **Recall** | 92.75% |
| **mAP@50** | 96.00% |
| **mAP@50-95** | 69.57% |
| **Classes** | 76 retail SKUs |
| **Model** | YOLOv8m (Stratified Split + Oversample) |

---

## Architecture

```
+-------------------------------------------------------------------------+
|                          ShelfVision AI                                  |
|                                                                         |
|   +----------+    +------------------+    +-----------------------+     |
|   | Frontend |---->  FastAPI Server  |---->  YOLO Detector        |     |
|   | (Web UI) |    |  (routes.py)     |    |  (detector.py)        |     |
|   +----------+    +--------+---------+    +----------+------------+     |
|                            |                          |                  |
|                            |         +----------------v-----------+     |
|                            |         |  Share of Shelf Analytics  |     |
|                            |         |  (share_of_shelf.py)       |     |
|                            |         +----------------------------+     |
|                            |                                            |
|                            v                                            |
|                   +--------------+    +--------------------------+      |
|                   | Drift Monitor|--->|  Auto-Retrain Trigger    |      |
|                   | (PSI Engine) |    |  (Airflow DAG)           |      |
|                   +--------------+    +--------------------------+      |
|                                                                         |
|   +----------------------------------------------------------+         |
|   |  GitHub Actions CI/CD: Lint > Test > Docker Build/Push   |         |
|   +----------------------------------------------------------+         |
+-------------------------------------------------------------------------+
```

---

## Quick Start

### Prerequisites

| Requirement | For Docker | For Manual |
|-------------|:----------:|:----------:|
| Docker Desktop | Required | -- |
| Python 3.10+ | -- | Required |
| Git | Required | Required |
| GPU | Optional | Optional |

### Option A: Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/your-username/retail-shelf-detection.git
cd retail-shelf-detection

# 2. Place your trained model weights
#    Download best.pt from the GitHub Releases page
cp ~/Downloads/best.pt models/best.pt

# 3. Build and run (one command)
docker compose up --build

# 4. Open your browser
#    Web UI:      http://localhost:8000
#    API Docs:    http://localhost:8000/docs
#    Health:      http://localhost:8000/api/health
```

#### Running with Airflow (Full MLOps Stack)

```bash
docker compose --profile full up --build
# API:     http://localhost:8000
# Airflow: http://localhost:8080  (login: admin / admin)
```

#### Running Tests Inside Docker

```bash
docker exec shelf-detection-api pytest tests/ -v
```

### Option B: Without Docker (Manual Setup)

```bash
# 1. Clone the repository
git clone https://github.com/your-username/retail-shelf-detection.git
cd retail-shelf-detection

# 2. Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your trained model weights
#    Download best.pt and place in models/ folder
#    models/best.pt

# 5. (Optional) Create .env from template
copy .env.example .env
# Edit .env if needed (defaults work out of the box)

# 6. Start the server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 7. Open your browser
#    Web UI:      http://localhost:8000
#    API Docs:    http://localhost:8000/docs
```

#### Running Tests Locally

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## Dataset Analysis

### Overview

The dataset consists of retail shelf images with 76 unique product SKU classes, sourced from Roboflow.

| Split | Images | Annotations | Avg Objects/Image |
|-------|-------:|------------:|------------------:|
| Train | 924 | 3,979 | 4.3 |
| Valid | 40 | 199 | 5.0 |
| Test | 35 | 145 | 4.1 |
| **Total** | **999** | **4,323** | **4.3** |

### Class Imbalance Problem

The dataset exhibits **severe class imbalance**:

- **Most common class**: q280 -- 443 instances (10.25%)
- **Least common class**: q178 -- 2 instances (0.05%)
- **Imbalance ratio**: **221.5x** between most and least common classes
- **4 classes with zero training samples** (q178, q199, q212, q299)
- **23 classes with fewer than 10 training samples**

```
CLASS IMBALANCE STATISTICS
===========================================================
  Most common:    q280 (443 total, 10.25%)
  Least common:   q178 (2 total, 0.05%)
  Imbalance ratio: 221.5x
  Classes < 5 training samples:  13 / 76
  Classes < 10 training samples: 23 / 76
  Classes < 20 training samples: 38 / 76
  Median instances: 36  |  Mean: 56.9  |  Std: 74.5
===========================================================
```

> The class distribution analysis is generated by `notebooks/01_preprocessing.ipynb`, which produces a 4-panel visualization (bar chart, split comparison, box plot, and Pareto curve) showing the full imbalance picture.

### Top 10 Classes by Instance Count

| Rank | Class | Train | Valid | Test | Total | Share% |
|------|-------|------:|------:|-----:|------:|-------:|
| 1 | q280 | 393 | 32 | 18 | 443 | 10.25% |
| 2 | q13 | 384 | 44 | 8 | 436 | 10.09% |
| 3 | q145 | 357 | 12 | 0 | 369 | 8.54% |
| 4 | q64 | 225 | 0 | 0 | 225 | 5.20% |
| 5 | q91 | 221 | 0 | 4 | 225 | 5.20% |
| 6 | q262 | 162 | 4 | 1 | 167 | 3.86% |
| 7 | q289 | 114 | 0 | 8 | 122 | 2.82% |
| 8 | q8 / q40 / q66 | 99 | -- | -- | ~99 | ~2.3% |

---

## Model Training Strategy and Evolution

The model selection process was iterative, conducted in two phases. Phase 1 explored different architectures and hyperparameters on the original dataset. Phase 2 shifted to a data-centric approach after identifying class imbalance as the core bottleneck.

### Phase 1: Architecture and Hyperparameter Exploration

#### Experiment 1: Baseline YOLOv11n (640x640)

| Metric | Value |
|--------|-------|
| Precision | 74.87% |
| Recall | **66.47%** |
| mAP@50 | 76.87% |
| mAP@50-95 | 46.38% |

**Strategy**: Start with the smallest YOLO variant to establish a performance baseline with default settings.

**Observation**: Decent precision but recall was too low at 66.47% -- the model was missing one-third of products on the shelf. For retail analytics, missing products is worse than false positives, since inventory tracking requires finding every item.

---

#### Experiment 2: YOLOv11n with Lower Confidence (0.15)

| Metric | Value | Delta from Baseline |
|--------|-------|:-------------------:|
| Precision | 67.84% | -7.03% |
| Recall | 66.90% | +0.43% |
| mAP@50 | 69.84% | -7.03% |
| mAP@50-95 | 44.86% | -1.52% |

**Strategy**: Lower the confidence threshold from 0.25 to 0.15 to capture more detections.

**Observation**: Did not work. Marginal recall improvement (+0.43%) came at a heavy precision cost (-7.03%). The model was already uncertain on its predictions -- lowering the threshold just added more wrong guesses rather than recovering truly missed objects. This confirmed the problem was in the model's feature extraction, not in the threshold.

---

#### Experiment 3: YOLOv11n with Enhanced Augmentation

| Metric | Value | Delta from Baseline |
|--------|-------|:-------------------:|
| Precision | 76.88% | +2.01% |
| Recall | 68.78% | +2.31% |
| mAP@50 | 79.21% | +2.34% |
| mAP@50-95 | 33.54% | -12.84% |

**Strategy**: Keep YOLOv11n but add aggressive augmentation -- mosaic (1.0), mixup (0.15), copy-paste (0.1), rotation (+/-15 deg), scale (0.7), color jitter (HSV).

**Observation**: Moderate improvement. Recall improved by +2.3% and precision also improved. However, mAP@50-95 dropped significantly (-12.84%). This suggests that while the model found more objects, its bounding box localization became less precise at stricter IoU thresholds. The nano model simply did not have enough capacity to handle both augmentation diversity and precise localization for 76 classes.

---

#### Experiment 4: YOLOv11m (640x640) with Enhanced Augmentation

| Metric | Value | Delta from Baseline |
|--------|-------|:-------------------:|
| Precision | **80.99%** | +6.12% |
| Recall | **76.58%** | +10.11% |
| mAP@50 | **84.52%** | +7.65% |
| mAP@50-95 | **50.89%** | +4.51% |

**Strategy**: Scale up to the medium-sized YOLOv11m (20M params vs 3M for nano) while keeping the enhanced augmentation.

**Observation**: Major breakthrough. Every metric improved substantially -- recall jumped from 66.47% to 76.58% (+10%) and precision from 74.87% to 80.99% (+6%). The larger model had enough capacity to learn discriminative features for 76 fine-grained product classes. This confirmed that the nano model was fundamentally too small for this task complexity.

---

#### Experiment 5: YOLOv11m at 1280x1280

| Metric | Value | Delta from Experiment 4 |
|--------|-------|:-----------------------:|
| Precision | 72.68% | -8.31% |
| Recall | **85.62%** | +9.04% |
| mAP@50 | -- | -- |
| mAP@50-95 | -- | -- |

**Strategy**: Double the input resolution from 640 to 1280 pixels to capture small products that were being missed.

**Observation**: Strong recall improvement. Recall reached 85.62% -- the model now finds 85% of all products on the shelf. Precision dropped to 72.68% as the higher resolution occasionally detected background textures as products. This confirmed that resolution scaling helps recall substantially, but the original imbalanced dataset was limiting both precision and overall mAP.

---

#### Experiment 6: RT-DETR-L (Transformer Architecture)

| Metric | Value |
|--------|-------|
| Precision | 65.55% |
| Recall | **85.76%** |
| mAP@50 | **88.36%** |
| mAP@50-95 | **54.09%** |

**Strategy**: Try a completely different architecture -- Real-Time DEtection TRansformer with global attention.

**Observation**: Highest mAP but lowest precision. The transformer's global attention mechanism excelled at finding objects (recall: 85.76%, mAP@50: 88.36%) but struggled with fine-grained SKU classification (precision: 65.55%). Transformers need significantly more data to learn discriminative features for 76 similar-looking product classes. The model was effective at locating products but poor at distinguishing which product it was.

---

#### Experiment 7: RT-DETR-X (Extra-Large Transformer)

| Metric | Value | Delta from RT-DETR-L |
|--------|-------|:--------------------:|
| Precision | 63.67% | -1.88% |
| Recall | 85.56% | -0.20% |
| mAP@50 | 86.93% | -1.43% |
| mAP@50-95 | 47.75% | -6.34% |

**Strategy**: Scale up the transformer to the extra-large variant, hoping more parameters would improve classification.

**Observation**: No improvement -- actually worse. The larger transformer overfitted on the small dataset (924 training images). This confirmed that the bottleneck is the dataset size and class imbalance, not the model architecture. More parameters without more data simply leads to overfitting, especially for fine-grained classification.

---

### Phase 2: Data-Centric Improvements (Stratified Split)

After Phase 1 established that the class imbalance and dataset split were the core bottleneck, Phase 2 focused on fixing the data.

**Approach**: Pool all 999 images from train/valid/test, then re-split using stratified sampling (80/10/10) based on the dominant class per image. This ensures every class is proportionally represented in each split, eliminating the 4 zero-train classes.

#### Experiment 8: YOLOv8m Stratified + Oversample (640x640) -- BEST MODEL

| Metric | Value | Delta from Experiment 5 |
|--------|-------|:-----------------------:|
| Precision | **81.15%** | +8.47% |
| Recall | **92.75%** | +7.13% |
| mAP@50 | **96.00%** | -- |
| mAP@50-95 | **69.57%** | -- |

**Strategy**: Stratified 80/10/10 split, oversample minority classes to >=15 instances, train YOLOv8m at 640x640.

**Observation**: Best result across all experiments. Stratified splitting combined with oversampling solved the class imbalance problem at the data level. Precision jumped to 81.15% (+8.47%) and recall reached 92.75% (+7.13%) -- both the highest ever. mAP@50 hit 96.00%, a massive leap over all previous experiments. This proved that fixing the data distribution matters more than scaling model size or resolution.

---

#### Experiment 9: YOLOv11m Stratified + Oversample (640x640)

| Metric | Value | Delta from Experiment 8 |
|--------|-------|:-----------------------:|
| Precision | 81.43% | +0.28% |
| Recall | 90.42% | -2.33% |
| mAP@50 | 93.48% | -2.52% |
| mAP@50-95 | 64.11% | -5.46% |

**Strategy**: Same stratified + oversample dataset, but train with YOLOv11m instead of YOLOv8m to compare architectures.

**Observation**: Strong results, but slightly behind YOLOv8m. YOLOv11m achieved very similar precision (81.43%) but lower recall (90.42% vs 92.75%) and mAP@50 (93.48% vs 96.00%). The newer YOLOv11 architecture did not provide a clear advantage on this dataset size.

---

#### Experiment 10: YOLOv11m Stratified + SGD Optimizer (640x640)

| Metric | Value | Delta from Experiment 9 |
|--------|-------|:-----------------------:|
| Precision | **82.48%** | +1.05% |
| Recall | 88.98% | -1.44% |
| mAP@50 | 93.68% | +0.20% |
| mAP@50-95 | 58.32% | -5.79% |

**Strategy**: Train YOLOv11m on stratified dataset (without oversampling) using explicit SGD optimizer.

**Observation**: SGD improved precision to 82.48% (second highest overall) but recall dropped to 88.98%. The optimizer helped the model make more confident and accurate predictions, but at the cost of missing more objects. mAP@50-95 was notably lower, suggesting less precise bounding box localization.

---

#### Experiment 11: YOLOv8m Stratified + SGD, No Oversample (640x640)

| Metric | Value | Delta from Experiment 8 |
|--------|-------|:-----------------------:|
| Precision | **83.45%** | +2.30% |
| Recall | 89.39% | -3.36% |
| mAP@50 | 95.66% | -0.34% |
| mAP@50-95 | **67.85%** | -1.72% |

**Strategy**: Train YOLOv8m on stratified dataset without oversampling, using SGD optimizer.

**Observation**: Highest precision of all experiments (83.45%) and very strong mAP@50-95 (67.85%). Without oversampling, the model saw only real images, producing higher quality predictions. However, recall was 89.39% vs 92.75% with oversampling. This variant is preferred when false positives are more costly than missed detections.

---

### Summary: All Experiments

```
                  Precision    Recall     Strategy
                  ---------    ------     ------------------------------------
Exp 1  v11n        74.87%      66.47%    Baseline
Exp 2  v11n LC     67.84%      66.90%    Lower confidence -- no help
Exp 3  v11n Aug    76.88%      68.78%    Augmentation -- moderate gain
Exp 4  v11m        80.99%      76.58%    Bigger model -- big jump
Exp 5  v11m@1280   72.68%      85.62%    Higher resolution -- good recall
Exp 6  DETR-L      65.55%      85.76%    Transformer -- low precision
Exp 7  DETR-X      63.67%      85.56%    Bigger transformer -- overfitting
Exp 8  v8m+Strat   81.15%      92.75%    Stratified+OS -- [BEST MODEL]
Exp 9  v11m+Strat  81.43%      90.42%    Stratified+OS -- strong runner-up
Exp 10 v11m+SGD    82.48%      88.98%    Stratified+SGD -- precision focused
Exp 11 v8m+SGD     83.45%      89.39%    Stratified+SGD no OS -- highest P
```

**Key Takeaway**: The biggest improvement came from fixing the dataset split (stratified sampling + oversampling), not from model architecture or resolution changes. The stratified approach increased recall from 85.62% to 92.75% and mAP@50 from ~84% to 96.00%. **Data quality outweighs model complexity.**

---

## Data Preprocessing and Rebalancing

Based on the class imbalance analysis, a multi-step rebalancing pipeline was developed (see `notebooks/01_preprocessing.ipynb` and `notebooks/03_stratified_training.py`).

### Step 1: Pooling All Images

The original split was discarded and all images were pooled into a single collection:

| Metric | Value |
|--------|-------|
| Original train | 924 images |
| Original valid | 40 images |
| Original test | 35 images |
| **Total pooled** | **999 images** |
| **Total annotations** | **4,323** |
| **Classes** | **76** |

#### Classes with Fewest Instances (Entire Pool)

| Class | Instances |
|-------|:---------:|
| q163 | 10 |
| q190 | 9 |
| q169 | 6 |
| q130 | 6 |
| q232 | 6 |
| q271 | 6 |
| q79 | 6 |
| q229 | 6 |
| q46 | 5 |
| q178 | 2 |

### Step 2: Stratified Split (80/10/10)

1. **Assign dominant class** per image (most frequent class in that image's annotations)
2. **Identify rare classes**: Classes with fewer than 3 images as dominant were forced into train (1 class: q46)
3. **First split**: 80% train / 20% temp, stratified on dominant class
4. **Handle singletons**: 22 classes had only 1 sample in the temp set -- these were alternately assigned to valid/test instead of crashing the stratified splitter
5. **Second split**: Remaining temp split 50/50 into valid/test, stratified on dominant class

**Result:**

| Split | Images | Percentage |
|-------|:------:|:----------:|
| Train | 799 | 80.0% |
| Valid | 100 | 10.0% |
| Test | 100 | 10.0% |

### Step 3: Stratification Quality Check

The key improvement: **zero-train classes dropped from 3 to 0**, and every class now has proportional representation across splits.

<details>
<summary>Full per-class breakdown: Original Train vs Stratified Train/Valid/Test (click to expand)</summary>

| Class | Orig Train | Strat Train | Strat Val | Strat Test |
|-------|:----------:|:-----------:|:---------:|:----------:|
| q280 | 393 | 359 | 40 | 44 |
| q13 | 384 | 353 | 61 | 22 |
| q145 | 357 | 294 | 30 | 45 |
| q64 | 225 | 169 | 28 | 28 |
| q91 | 221 | 181 | 25 | 19 |
| q262 | 162 | 131 | 19 | 17 |
| q289 | 114 | 89 | 12 | 21 |
| q214 | 99 | 86 | 19 | 13 |
| q121 | 99 | 79 | 10 | 14 |
| q202 | 99 | 84 | 6 | 12 |
| q109 | 93 | 81 | 6 | 12 |
| q193 | 87 | 80 | 12 | 5 |
| q31 | 63 | 75 | 0 | 22 |
| q250 | 81 | 72 | 8 | 4 |
| q256 | 69 | 54 | 11 | 9 |
| q286 | 60 | 54 | 6 | 8 |
| q4 | 60 | 55 | 4 | 5 |
| q220 | 60 | 54 | 3 | 6 |
| q142 | 57 | 49 | 5 | 6 |
| q293 | 42 | 45 | 10 | 3 |
| q175 | 48 | 46 | 6 | 5 |
| q7 | 54 | 46 | 4 | 6 |
| q112 | 48 | 45 | 5 | 4 |
| q16 | 45 | 35 | 6 | 8 |
| q199 | 45 | 39 | 6 | 3 |
| q22 | 48 | 36 | 6 | 6 |
| q299 | 44 | 39 | 5 | 2 |
| q82 | 42 | 30 | 6 | 6 |
| q94 | 42 | 34 | 3 | 5 |
| q151 | 39 | 31 | 2 | 8 |
| q52 | 39 | 30 | 6 | 4 |
| q67 | 33 | 25 | 6 | 4 |
| q115 | 30 | 29 | 5 | 1 |
| q61 | 24 | 24 | 4 | 2 |
| q70 | 30 | 22 | 8 | 0 |
| q58 | 30 | 24 | 4 | 2 |
| q106 | 21 | 23 | 2 | 2 |
| q157 | 18 | 18 | 0 | 6 |
| q40 | 24 | 20 | 4 | 0 |
| q100 | 21 | 19 | 2 | 1 |
| q1 | 21 | 18 | 3 | 0 |
| q136 | 21 | 17 | 2 | 2 |
| q88 | 21 | 17 | 2 | 2 |
| q34 | 21 | 17 | 2 | 2 |
| q291 | 21 | 16 | 3 | 2 |
| q25 | 18 | 14 | 2 | 2 |
| q133 | 18 | 14 | 2 | 2 |
| q55 | 18 | 14 | 2 | 2 |
| q196 | 18 | 15 | 1 | 2 |
| q274 | 18 | 15 | 1 | 2 |
| q97 | 18 | 15 | 3 | 0 |
| q265 | 12 | 12 | 2 | 2 |
| q187 | 12 | 13 | 1 | 2 |
| q184 | 15 | 13 | 2 | 1 |
| q148 | 15 | 12 | 3 | 0 |
| q118 | 12 | 10 | 2 | 2 |
| q49 | 9 | 9 | 0 | 5 |
| q211 | 12 | 11 | 0 | 2 |
| q103 | 12 | 9 | 3 | 0 |
| q76 | 12 | 8 | 0 | 4 |
| q19 | 12 | 9 | 2 | 1 |
| q37 | 12 | 9 | 2 | 1 |
| q268 | 9 | 10 | 1 | 1 |
| q10 | 6 | 8 | 0 | 2 |
| q247 | 9 | 7 | 1 | 2 |
| q73 | 9 | 9 | 1 | 0 |
| q163 | 9 | 9 | 0 | 1 |
| q190 | 9 | 6 | 0 | 3 |
| q130 | 6 | 4 | 0 | 2 |
| q232 | 6 | 6 | 0 | 0 |
| q169 | 6 | 6 | 0 | 0 |
| q229 | **0** | 4 | 2 | 0 |
| q79 | 6 | 4 | 0 | 2 |
| q271 | 6 | 5 | 0 | 1 |
| q46 | **0** | 5 | 0 | 0 |
| q178 | **0** | 2 | 0 | 0 |

</details>

### Step 4: Oversampling Minority Classes

Minority classes (fewer than 15 training instances) were oversampled by duplicating images containing those classes:

| Metric | Value |
|--------|-------|
| Images oversampled | 115 |
| Final training images | 914 |

### Summary

| Metric | Original Split | Stratified Split |
|--------|:--------------:|:----------------:|
| Zero-train classes | 3 | **0** |
| Train images | 924 | 914 (799 + 115 oversampled) |
| Valid images | 40 | 100 |
| Test images | 35 | 100 |

### Targeted Augmentation (Applied During Training)

Augmented copies of minority class images were created using 6 random transforms:

- **Brightness** (0.6x to 1.4x)
- **Contrast** adjustment
- **Horizontal flip** (with label coordinate correction)
- **Color jitter** (HSV space)
- **Gaussian blur**
- **Gaussian noise**

---

## Share of Shelf Analytics

The entire test dataset was treated as a single representative store shelf. The best model (YOLOv8m Stratified + Oversample) was used to detect all products, and the percentage share of each SKU was calculated.

### Key Findings

| Metric | Value |
|--------|-------|
| Total products detected | 198 |
| Most dominant SKU | q64 (14.1%) |
| Top 5 SKUs coverage | 54.1% of total shelf |
| Top 10 SKUs coverage | 73.8% of total shelf |

The analysis reveals a highly concentrated shelf layout where the **top 10 SKUs account for nearly three-quarters of all shelf space**, consistent with typical retail planogram strategies where high-demand products receive disproportionate shelf allocation.

### Top 10 SKUs by Share of Shelf

| Rank | SKU | Detections | Share (%) |
|------|-----|:----------:|:---------:|
| 1 | q64 | 28 | 14.1% |
| 2 | q280 | 24 | 12.1% |
| 3 | q91 | 22 | 11.1% |
| 4 | q13 | 18 | 9.1% |
| 5 | q145 | 15 | 7.6% |
| 6 | q262 | 10 | 5.1% |
| 7 | q289 | 8 | 4.0% |
| 8 | q40 | 8 | 4.0% |
| 9 | q8 | 7 | 3.5% |
| 10 | q66 | 6 | 3.0% |

### Share of Shelf Distribution

<!-- Place your doughnut chart image here. See instructions below. -->
![Share of Shelf Distribution - Top 10 SKUs vs Others](assets/share_of_shelf_doughnut.png)

> The full analysis with 5 visualizations (bar chart, full distribution, doughnut, confidence scatter, detection heatmap) is generated by `notebooks/colab_share_of_shelf.py`.

---

## API Reference

The FastAPI server exposes 5 endpoints with interactive documentation at `/docs`.

### POST /api/detect

Upload an image and receive object detection results.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | Required | JPEG or PNG image |
| `confidence` | float | 0.25 | Min confidence threshold (0.05 to 0.95) |
| `return_image` | bool | true | Return annotated image or JSON |

**Response (return_image=true)**: JPEG image with bounding boxes drawn. Detection count and details in response headers (X-Detection-Count, X-Detections).

**Response (return_image=false)**:
```json
{
  "success": true,
  "count": 5,
  "detections": [
    {
      "class_id": 52,
      "class_name": "q280",
      "confidence": 0.91,
      "bbox": {"x1": 120, "y1": 80, "x2": 250, "y2": 310}
    }
  ]
}
```

### POST /api/share-of-shelf

Upload an image and get SKU percentage breakdown.

```json
{
  "success": true,
  "total_products": 12,
  "top_skus": [
    {"sku": "q280", "count": 4, "percentage": 33.3}
  ],
  "all_skus": [...]
}
```

### GET /api/health

Health check endpoint for container orchestration and load balancers.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/best.pt"
}
```

### GET /api/metrics

Returns the full experiment log with all training run results.

### GET /api/drift-status

Returns the current drift monitoring status.

```json
{
  "status": "HEALTHY",
  "psi_score": 0.0523,
  "avg_confidence": 0.72,
  "predictions_logged": 150,
  "message": "Model predictions are stable."
}
```

---

## Frontend Dashboard

The web UI is served at `http://localhost:8000` and provides 4 tabs:

| Tab | Feature |
|-----|---------|
| **Detect** | Drag-and-drop image upload, annotated result with bounding boxes + detection table |
| **Share of Shelf** | Upload image, horizontal bar chart + doughnut chart of SKU distribution |
| **Experiments** | Auto-loaded experiment log table + grouped bar chart comparing all models |
| **Drift Monitor** | Live PSI score, avg confidence, prediction count, and status badge |

**Design**: Dark glassmorphism theme with gradient accents, Chart.js visualizations, responsive layout, Inter font.

---

## MLOps Pipeline

### Apache Airflow DAGs

Two DAGs are provided for ML workflow orchestration:

#### 1. ML Training Pipeline (airflow/dags/ml_pipeline.py)

Triggered manually or by the drift monitor.

```
validate_data -> preprocess -> train_model -> evaluate_model -> check_metrics
                                                                   |
                                                         +---------+---------+
                                                         v                   v
                                                    deploy_model        alert_team
                                                   (recall > 80%)     (recall <= 80%)
                                                         |                   |
                                                         v                   v
                                                                  end
```

#### 2. Drift Monitor DAG (airflow/dags/drift_monitor.py)

Runs daily at 2 AM.

```
collect_predictions -> compute_drift -> check_drift
                                          |
                                +---------+----------+
                                v                    v
                         log_drift_alert        log_healthy -> end
                                |
                                v
                       trigger_retraining
                                |
                                v
                       evaluate_new_model -> check_improvement
                                                   |
                                          +--------+--------+
                                          v                 v
                                     swap_model       keep_current
                                          |                 |
                                          v                 v
                                                   end
```

**To view Airflow UI:**

```bash
docker compose --profile full up --build
# Open http://localhost:8080 (login: admin / admin)
```

---

## Data Drift Monitoring and Auto-Retraining

The system continuously monitors for prediction distribution drift using two metrics:

### Population Stability Index (PSI)

Compares the distribution of predicted classes against the training data baseline.

```
PSI = Sum[ (P_recent - P_training) * ln(P_recent / P_training) ]

PSI < 0.1   -> HEALTHY         (no action)
PSI 0.1-0.2 -> WARNING         (monitoring closely)
PSI > 0.2   -> DRIFT_DETECTED  (auto-retrain recommended)
```

### Confidence Decay

Tracks average prediction confidence against the baseline (65%). A drop >15% indicates the model is encountering unfamiliar inputs.

### How It Works

1. Every `/api/detect` call silently logs predictions to a rolling buffer (last 500)
2. PSI is computed against the training class distribution
3. Average confidence is tracked
4. The Drift Monitor Airflow DAG checks daily and triggers retraining if drift is detected
5. After retraining, the new model is compared against the current one -- only swapped if metrics improve

### Configuration (src/monitoring/drift_config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| PSI_THRESHOLD_WARNING | 0.1 | Moderate shift warning |
| PSI_THRESHOLD_DRIFT | 0.2 | Drift detected threshold |
| CONFIDENCE_BASELINE | 0.65 | Expected avg confidence |
| CONFIDENCE_DECAY_THRESHOLD | 0.15 | Max acceptable confidence drop |
| MONITORING_WINDOW | 500 | Rolling buffer size |
| RETRAIN_COOLDOWN_HOURS | 24 | Min hours between retrain triggers |

---

## CI/CD Pipeline

### GitHub Actions Workflows

#### ci.yml -- Continuous Integration

Triggers on every push to main/develop and on pull requests.

```
Push/PR -> Ruff Lint -> Mypy Type Check -> Pytest -> Docker Build Test
```

#### docker-build.yml -- Continuous Delivery

Triggers on push to main branch or version tags (v*).

```
Push to main -> Build Docker Image -> Tag with SHA -> Push to GitHub Container Registry
```

---


## Project Structure

```
retail-shelf-detection/
|-- .github/
|   +-- workflows/
|       |-- ci.yml                  # Lint + Test + Docker Build
|       +-- docker-build.yml        # Build and Push to GHCR
|
|-- airflow/
|   +-- dags/
|       |-- ml_pipeline.py          # Training pipeline DAG
|       +-- drift_monitor.py        # Drift detection + auto-retrain DAG
|
|-- dataset/                        # Training data (not in Docker)
|   |-- train/images/ and labels/
|   |-- valid/images/ and labels/
|   +-- test/images/ and labels/
|
|-- experiments/
|   +-- experiment_log.json         # All training run results (11 experiments)
|
|-- frontend/
|   |-- index.html                  # 4-tab SPA
|   |-- style.css                   # Dark glassmorphism theme
|   +-- app.js                      # Drag-drop, API calls, Chart.js
|
|-- models/
|   |-- best.pt                     # Trained model weights (download separately)
|   +-- model_card.md               # Google-format model documentation
|
|-- notebooks/
|   |-- 01_preprocessing.ipynb      # Data analysis + class rebalancing
|   |-- 02_training.ipynb           # YOLOv11m training + evaluation
|   |-- 03_stratified_training.py   # Stratified split + YOLOv11m (Kaggle)
|   |-- colab_yolov8_training.py    # Stratified + YOLOv8m (Google Colab)
|   +-- colab_yolov8_no_oversample.py  # Stratified no OS + YOLOv8m (Colab)
|
|-- src/
|   |-- api/
|   |   |-- main.py                 # FastAPI app + lifespan
|   |   |-- routes.py               # 5 API endpoints
|   |   +-- schemas.py              # Pydantic models
|   |-- inference/
|   |   |-- detector.py             # Thread-safe YOLO singleton
|   |   +-- share_of_shelf.py       # SKU distribution analytics
|   |-- monitoring/
|   |   |-- drift_config.py         # PSI thresholds + settings
|   |   +-- drift_detector.py       # Rolling buffer + PSI engine
|   +-- config.py                   # Central configuration
|
|-- tests/
|   |-- test_api.py                 # API endpoint tests
|   |-- test_inference.py           # Inference + Share of Shelf tests
|   +-- test_preprocessing.py       # Drift monitoring tests
|
|-- .dockerignore
|-- .env.example                    # Environment variable template
|-- .gitignore
|-- Dockerfile                      # Multi-stage production build
|-- docker-compose.yml              # API + optional Airflow
|-- pyproject.toml                  # Project metadata + tool config
|-- requirements.txt                # Production dependencies
|-- requirements-dev.txt            # Dev + test dependencies
+-- README.md                       # This file
```

---

## Full Experiment Log

### Phase 1: Architecture Exploration (Original Dataset)

| # | Model | ImgSz | Precision | Recall | mAP@50 | mAP@50-95 | Observation |
|---|-------|:-----:|:---------:|:------:|:------:|:---------:|-------------|
| 1 | YOLOv11n | 640 | 74.87% | 66.47% | 76.87% | 46.38% | Baseline -- decent P, low R |
| 2 | YOLOv11n | 640 | 67.84% | 66.90% | 69.84% | 44.86% | Lower conf -- no help |
| 3 | YOLOv11n | 640 | 76.88% | 68.78% | 79.21% | 33.54% | Augmentation -- moderate gain |
| 4 | YOLOv11m | 640 | 80.99% | 76.58% | 84.52% | 50.89% | Bigger model -- big jump |
| 5 | YOLOv11m | 1280 | 72.68% | 85.62% | -- | -- | Higher res -- good recall |
| 6 | RT-DETR-L | 640 | 65.55% | 85.76% | 88.36% | 54.09% | Transformer -- low precision |
| 7 | RT-DETR-X | 640 | 63.67% | 85.56% | 86.93% | 47.75% | Overfitting on small data |

### Phase 2: Data-Centric Improvements (Stratified Split)

| # | Model | Dataset Variant | Precision | Recall | mAP@50 | mAP@50-95 | Observation |
|---|-------|-----------------|:---------:|:------:|:------:|:---------:|-------------|
| 8 | **YOLOv8m** | **Strat + Oversample** | **81.15%** | **92.75%** | **96.00%** | **69.57%** | **BEST -- highest recall and mAP@50** |
| 9 | YOLOv11m | Strat + Oversample | 81.43% | 90.42% | 93.48% | 64.11% | Strong runner-up |
| 10 | YOLOv11m + SGD | Stratified | 82.48% | 88.98% | 93.68% | 58.32% | SGD improved precision |
| 11 | YOLOv8m + SGD | Strat (no OS) | 83.45% | 89.39% | 95.66% | 67.85% | Highest precision (83.45%) |

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- [Ultralytics](https://docs.ultralytics.com/) -- YOLOv8/v11 framework
- [Roboflow](https://roboflow.com/) -- Dataset hosting and annotation
- [FastAPI](https://fastapi.tiangolo.com/) -- High-performance API framework
- [Chart.js](https://www.chartjs.org/) -- Frontend visualizations
- [Apache Airflow](https://airflow.apache.org/) -- ML pipeline orchestration
