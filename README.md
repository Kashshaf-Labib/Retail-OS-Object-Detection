<p align="center">
  <h1 align="center">🔍 ShelfVision AI</h1>
  <p align="center">
    <strong>End-to-End Retail Shelf Detection with MLOps</strong><br>
    YOLOv8m · YOLOv11m · 76 SKU Classes · FastAPI · Docker · Airflow · CI/CD
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv11m-Ultralytics-purple?logo=pytorch" alt="YOLO">
  <img src="https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-Compose-blue?logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/Airflow-2.8-red?logo=apacheairflow" alt="Airflow">
  <img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?logo=githubactions" alt="CI/CD">
</p>

---

## 📖 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
  - [With Docker (Recommended)](#option-a-docker-recommended)
  - [Without Docker](#option-b-without-docker-manual-setup)
- [Dataset Analysis](#-dataset-analysis)
- [Model Training Strategy & Evolution](#-model-training-strategy--evolution)
- [Data Preprocessing & Rebalancing](#-data-preprocessing--rebalancing)
- [API Reference](#-api-reference)
- [Frontend Dashboard](#-frontend-dashboard)
- [MLOps Pipeline](#-mlops-pipeline)
- [Data Drift Monitoring & Auto-Retraining](#-data-drift-monitoring--auto-retraining)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Testing](#-testing)
- [Project Structure](#-project-structure)
- [Experiment Log](#-experiment-log)

---

## 🎯 Project Overview

**ShelfVision AI** is a production-grade retail shelf product detection system that identifies and counts 76 unique product SKUs from store shelf images. The project delivers:

1. **Object Detection** — YOLOv11m model trained at 1280×1280 for detecting fine-grained retail products
2. **Share of Shelf Analytics** — Computes percentage distribution of each SKU for shelf space analysis
3. **Data Drift Monitoring** — PSI-based drift detection with automatic retraining triggers
4. **Full MLOps Pipeline** — Docker, CI/CD, Airflow orchestration, experiment tracking, and model versioning

### Key Results (Best Model — YOLOv8m Stratified+Oversample)

| Metric | Value |
|--------|-------|
| **Precision** | 81.15% |
| **Recall** | 92.75% |
| **mAP@50** | 96.00% |
| **mAP@50-95** | 69.57% |
| **Classes** | 76 retail SKUs |
| **Model** | YOLOv8m (Stratified Split + Oversample) |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ShelfVision AI                                 │
│                                                                         │
│   ┌──────────┐    ┌──────────────────┐    ┌───────────────────────┐    │
│   │ Frontend │───▶│  FastAPI Server   │───▶│  YOLOv11m Detector   │    │
│   │ (Web UI) │    │  (routes.py)      │    │  (detector.py)        │    │
│   └──────────┘    └──────┬───────────┘    └───────────┬───────────┘    │
│                          │                             │                │
│                          │         ┌───────────────────▼──────────┐    │
│                          │         │  Share of Shelf Analytics    │    │
│                          │         │  (share_of_shelf.py)         │    │
│                          │         └─────────────────────────────┘    │
│                          │                                             │
│                          ▼                                             │
│                   ┌──────────────┐    ┌──────────────────────────┐    │
│                   │ Drift Monitor│───▶│  Auto-Retrain Trigger    │    │
│                   │ (PSI Engine) │    │  (Airflow DAG)           │    │
│                   └──────────────┘    └──────────────────────────┘    │
│                                                                         │
│   ┌──────────────────────────────────────────────────────────┐        │
│   │  GitHub Actions CI/CD → Lint → Test → Docker Build/Push  │        │
│   └──────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | For Docker | For Manual |
|-------------|:----------:|:----------:|
| Docker Desktop | ✅ Required | ❌ |
| Python 3.10+ | ❌ | ✅ Required |
| Git | ✅ Required | ✅ Required |
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
# → API:     http://localhost:8000
# → Airflow: http://localhost:8080  (login: admin / admin)
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

## 📊 Dataset Analysis

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

- **Most common class**: `q280` — 443 instances (10.25%)
- **Least common class**: `q178` — 2 instances (0.05%)
- **Imbalance ratio**: **221.5×** between most and least common classes
- **4 classes with zero training samples** (`q178`, `q199`, `q212`, `q299`)
- **23 classes with fewer than 10 training samples**

```
╔═══════════════════════════════════════════════════════╗
║  CLASS IMBALANCE STATISTICS                            ║
╠═══════════════════════════════════════════════════════╣
║  Most common:    q280 (443 total, 10.25%)              ║
║  Least common:   q178 (2 total, 0.05%)                 ║
║  Imbalance ratio: 221.5×                               ║
║  Classes < 5 training samples:  13 / 76                ║
║  Classes < 10 training samples: 23 / 76                ║
║  Classes < 20 training samples: 38 / 76                ║
║  Median instances: 36  |  Mean: 56.9  |  Std: 74.5    ║
╚═══════════════════════════════════════════════════════╝
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
| 8 | q8 / q40 / q66 | 99 | — | — | ~99 | ~2.3% |

---

## 🧪 Model Training Strategy & Evolution

The model selection process was iterative. Each experiment addressed limitations discovered in the previous one. Below is the full journey with rationale and observations.

### Experiment 1: Baseline YOLOv11n (640×640)

| Metric | Value |
|--------|-------|
| Precision | 74.87% |
| Recall | **66.47%** |
| mAP@50 | 76.87% |
| mAP@50-95 | 46.38% |

**Strategy**: Start with the smallest YOLO variant to establish a performance baseline with default settings.

**Observation**: Decent precision but **recall was too low at 66.47%** — the model was missing one-third of products on the shelf. For retail analytics, missing products is worse than false positives, since inventory tracking requires finding every item.

---

### Experiment 2: YOLOv11n with Lower Confidence (0.15)

| Metric | Value | Δ from Baseline |
|--------|-------|:---------------:|
| Precision | 67.84% | ↓ 7.03% |
| Recall | 66.90% | ↑ 0.43% |
| mAP@50 | 69.84% | ↓ 7.03% |
| mAP@50-95 | 44.86% | ↓ 1.52% |

**Strategy**: Lower the confidence threshold from 0.25 to 0.15 to capture more detections.

**Observation**: ❌ **Did not work.** Marginal recall improvement (+0.43%) came at a heavy precision cost (−7.03%). The model was already uncertain on its predictions — lowering the threshold just added more wrong guesses rather than recovering truly missed objects. This confirmed the problem was in the model's feature extraction, not in the threshold.

---

### Experiment 3: YOLOv11n with Enhanced Augmentation

| Metric | Value | Δ from Baseline |
|--------|-------|:---------------:|
| Precision | 76.88% | ↑ 2.01% |
| Recall | 68.78% | ↑ 2.31% |
| mAP@50 | 79.21% | ↑ 2.34% |
| mAP@50-95 | 33.54% | ↓ 12.84% |

**Strategy**: Keep YOLOv11n but add aggressive augmentation — mosaic (1.0), mixup (0.15), copy-paste (0.1), rotation (±15°), scale (0.7), color jitter (HSV).

**Observation**: ✅ **Moderate improvement.** Recall improved by +2.3% and precision also improved. However, **mAP@50-95 dropped significantly** (−12.84%). This suggests that while the model found more objects, its bounding box localization became less precise at stricter IoU thresholds. The nano model simply didn't have enough capacity to handle both augmentation diversity AND precise localization for 76 classes.

---

### Experiment 4: YOLOv11m (640×640) with Enhanced Augmentation

| Metric | Value | Δ from Baseline |
|--------|-------|:---------------:|
| Precision | **80.99%** | ↑ 6.12% |
| Recall | **76.58%** | ↑ 10.11% |
| mAP@50 | **84.52%** | ↑ 7.65% |
| mAP@50-95 | **50.89%** | ↑ 4.51% |

**Strategy**: Scale up to the medium-sized YOLOv11m (20M params vs 3M for nano) while keeping the enhanced augmentation.

**Observation**: ✅ **Major breakthrough.** Every metric improved substantially — recall jumped from 66.47% to 76.58% (+10%) and precision from 74.87% to 80.99% (+6%). The larger model had enough capacity to learn discriminative features for 76 fine-grained product classes. This confirmed that the nano model was fundamentally too small for this task complexity.

---

### Experiment 5: YOLOv11m at 1280×1280

| Metric | Value | Δ from Experiment 4 |
|--------|-------|:-------------------:|
| Precision | 72.68% | ↓ 8.31% |
| Recall | **85.62%** | ↑ 9.04% |
| mAP@50 | — | — |
| mAP@50-95 | — | — |

**Strategy**: Double the input resolution from 640 to 1280 pixels to capture small products that were being missed.

**Observation**: ✅ **Strong recall improvement.** Recall reached **85.62%** — the model now finds 85% of all products on the shelf. Precision dropped to 72.68% as the higher resolution occasionally detected background textures as products. This confirmed that resolution scaling helps recall substantially, but the original imbalanced dataset was limiting both precision and overall mAP.

---

### Experiment 6: RT-DETR-L (Transformer Architecture)

| Metric | Value |
|--------|-------|
| Precision | 65.55% |
| Recall | **85.76%** |
| mAP@50 | **88.36%** |
| mAP@50-95 | **54.09%** |

**Strategy**: Try a completely different architecture — Real-Time DEtection TRansformer with global attention.

**Observation**: ⚠️ **Highest mAP but lowest precision.** The transformer's global attention mechanism excelled at finding objects (recall: 85.76%, mAP@50: 88.36% — both the highest across all experiments) but **struggled with fine-grained SKU classification** (precision: 65.55%). Transformers need significantly more data to learn discriminative features for 76 similar-looking product classes. The model was great at saying "there's a product here" but poor at saying "which product it is."

---

### Experiment 7: RT-DETR-X (Extra-Large Transformer)

| Metric | Value | Δ from RT-DETR-L |
|--------|-------|:-----------------:|
| Precision | 63.67% | ↓ 1.88% |
| Recall | 85.56% | ↓ 0.20% |
| mAP@50 | 86.93% | ↓ 1.43% |
| mAP@50-95 | 47.75% | ↓ 6.34% |

**Strategy**: Scale up the transformer to the extra-large variant, hoping more parameters would improve classification.

**Observation**: ❌ **No improvement — actually worse.** The larger transformer overfitted on the small dataset (924 training images). This confirmed that the **bottleneck is the dataset size and class imbalance, not the model architecture**. More parameters without more data simply leads to overfitting, especially for fine-grained classification.

---

### Experiment 8: YOLOv8m Stratified+Oversample (640×640) ⭐ Best Model

| Metric | Value | Δ from Experiment 5 |
|--------|-------|:-------------------:|
| Precision | **81.15%** | ↑ 8.47% |
| Recall | **92.75%** | ↑ 7.13% |
| mAP@50 | **96.00%** | — |
| mAP@50-95 | **69.57%** | — |

**Strategy**: Address the root cause — the dataset split itself. Pool all images, apply **stratified 80/10/10 split** based on dominant class per image (ensuring every class appears in train), then oversample minority classes to ≥15 instances. Train **YOLOv8m** at 640×640.

**Observation**: ✅ **Best result across all experiments.** Stratified splitting combined with oversampling solved the class imbalance problem at the data level. Precision jumped to **81.15%** (+8.47%) and recall reached **92.75%** (+7.13%) — both the highest ever. mAP@50 hit **96.00%**, a massive leap over all previous experiments. This proved that **fixing the data distribution matters more than scaling model size or resolution**. YOLOv8m's architecture was also well-suited for this task at 640 resolution.

---

### Experiment 9: YOLOv11m Stratified+Oversample (640×640)

| Metric | Value | Δ from Experiment 8 |
|--------|-------|:-------------------:|
| Precision | 81.43% | ↑ 0.28% |
| Recall | 90.42% | ↓ 2.33% |
| mAP@50 | 93.48% | ↓ 2.52% |
| mAP@50-95 | 64.11% | ↓ 5.46% |

**Strategy**: Same stratified+oversample dataset, but train with **YOLOv11m** instead of YOLOv8m to compare architectures on the fixed dataset.

**Observation**: ✅ **Strong results, but slightly behind YOLOv8m.** YOLOv11m achieved very similar precision (81.43%) but lower recall (90.42% vs 92.75%) and mAP@50 (93.48% vs 96.00%). The newer YOLOv11 architecture didn't provide a clear advantage on this dataset size. This suggests that YOLOv8m's training dynamics (convergence, augmentation response) are better tuned for smaller datasets with 76 fine-grained classes.

### Summary: Precision vs Recall Across All Experiments

```
               Precision    Recall     Strategy
               ─────────    ──────     ────────────────────────────────────
v11n            74.87%      66.47%    Baseline
v11n LC         67.84%      66.90%    Lower confidence → ❌ No help
v11n Aug        76.88%      68.78%    Augmentation → ↑ Moderate gain
v11m            80.99%      76.58%    Bigger model → ✅ Big jump
v11m@1280       72.68%      85.62%    Higher resolution → Good recall
DETR-L          65.55%      85.76%    Transformer → High recall, low precision
DETR-X          63.67%      85.56%    Bigger transformer → ❌ Overfitting
v8m+Strat       81.15%      92.75%    Stratified split → ⭐ BEST MODEL
v11m+Strat      81.43%      90.42%    Stratified + v11m → Strong runner-up
```

**Key Takeaway**: The biggest improvement came from **fixing the dataset split** (stratified sampling + oversampling), not from model architecture or resolution changes. The stratified approach increased recall from 85.62% → 92.75% and mAP@50 from ~84% → 96.00%. **Data quality > model complexity.**

---

## 🔄 Data Preprocessing & Rebalancing

Based on the class imbalance analysis, a 3-step rebalancing pipeline was developed (`notebooks/01_preprocessing.ipynb`):

### Step 1: Rescue Zero-Train Classes

4 classes had **zero training samples** but existed in the validation set. These images were copied to the training split.

```
q178 → 0 → rescued from valid
q199 → 0 → rescued from valid
q212 → 0 → rescued from valid
q299 → 0 → rescued from valid
```

### Step 2: Oversample Minority Classes

All classes with fewer than 30 training instances were oversampled by duplicating images containing those classes.

```
Target: ≥ 30 instances per class
Classes affected: 38 / 76
Method: Random duplication of existing images
```

### Step 3: Targeted Augmentation

Augmented copies of minority class images were created using 6 random transforms:

- **Brightness** (0.6× – 1.4×)
- **Contrast** adjustment
- **Horizontal flip** (with label coordinate correction)
- **Color jitter** (HSV space)
- **Gaussian blur**
- **Gaussian noise**

### Results

| Metric | Before | After |
|--------|--------|-------|
| Imbalance ratio | 221.5× | ~13× |
| Min instances | 0 | ≥30 |
| Zero-sample classes | 4 | 0 |
| Total training images | 924 | ~1,050+ |

> The full before/after comparison chart is generated by `notebooks/01_preprocessing.ipynb`, which produces side-by-side horizontal bar charts color-coded by severity (red < 20, yellow < 30, green ≥ 30).

---

## 📡 API Reference

The FastAPI server exposes 5 endpoints with interactive documentation at `/docs`.

### `POST /api/detect`

Upload an image and receive object detection results.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | Required | JPEG or PNG image |
| `confidence` | float | 0.25 | Min confidence threshold (0.05 – 0.95) |
| `return_image` | bool | true | Return annotated image or JSON |

**Response (return_image=true)**: JPEG image with bounding boxes drawn. Detection count and details in response headers (`X-Detection-Count`, `X-Detections`).

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

### `POST /api/share-of-shelf`

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

### `GET /api/health`

Health check endpoint for container orchestration and load balancers.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/best.pt"
}
```

### `GET /api/metrics`

Returns the full experiment log with all training run results.

### `GET /api/drift-status`

Returns the current drift monitoring status.

```json
{
  "status": "HEALTHY",
  "psi_score": 0.0523,
  "avg_confidence": 0.72,
  "predictions_logged": 150,
  "message": "✅ Model predictions are stable."
}
```

---

## 🖥 Frontend Dashboard

The web UI is served at `http://localhost:8000` and provides 4 tabs:

| Tab | Feature |
|-----|---------|
| **🎯 Detect** | Drag-and-drop image upload → annotated result with bounding boxes + detection table |
| **📊 Share of Shelf** | Upload image → horizontal bar chart + doughnut chart of SKU distribution |
| **🧪 Experiments** | Auto-loaded experiment log table + grouped bar chart comparing all models |
| **📡 Drift Monitor** | Live PSI score, avg confidence, prediction count, and pulsing status badge |

**Design**: Dark glassmorphism theme with gradient accents, Chart.js visualizations, responsive layout, Inter font.

---

## ⚙️ MLOps Pipeline

### Apache Airflow DAGs

Two DAGs are provided for ML workflow orchestration:

#### 1. ML Training Pipeline (`airflow/dags/ml_pipeline.py`)

Triggered manually or by the drift monitor.

```
validate_data → preprocess → train_model → evaluate_model → check_metrics
                                                                   │
                                                         ┌─────────┴─────────┐
                                                         ▼                   ▼
                                                    deploy_model        alert_team
                                                   (recall > 80%)     (recall ≤ 80%)
                                                         │                   │
                                                         ▼                   ▼
                                                                  end
```

#### 2. Drift Monitor DAG (`airflow/dags/drift_monitor.py`)

Runs daily at 2 AM.

```
collect_predictions → compute_drift → check_drift
                                          │
                                ┌─────────┴──────────┐
                                ▼                    ▼
                         log_drift_alert        log_healthy → end
                                │
                                ▼
                       trigger_retraining
                                │
                                ▼
                       evaluate_new_model → check_improvement
                                                   │
                                          ┌────────┴────────┐
                                          ▼                 ▼
                                     swap_model       keep_current
                                          │                 │
                                          ▼                 ▼
                                                   end
```

**To view Airflow UI:**

```bash
docker compose --profile full up --build
# Open http://localhost:8080 (login: admin / admin)
```

---

## 📉 Data Drift Monitoring & Auto-Retraining

The system continuously monitors for prediction distribution drift using two metrics:

### Population Stability Index (PSI)

Compares the distribution of predicted classes against the training data baseline.

```
PSI = Σ (P_recent - P_training) × ln(P_recent / P_training)

PSI < 0.1   → ✅ HEALTHY        (no action)
PSI 0.1-0.2 → 🟡 WARNING        (monitoring closely)
PSI > 0.2   → ⚠️ DRIFT_DETECTED (auto-retrain recommended)
```

### Confidence Decay

Tracks average prediction confidence against the baseline (65%). A drop >15% indicates the model is encountering unfamiliar inputs.

### How It Works

1. Every `/api/detect` call silently logs predictions to a rolling buffer (last 500)
2. PSI is computed against the training class distribution
3. Average confidence is tracked
4. The Drift Monitor Airflow DAG checks daily and triggers retraining if drift is detected
5. After retraining, the new model is compared against the current one — only swapped if metrics improve

### Configuration (`src/monitoring/drift_config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PSI_THRESHOLD_WARNING` | 0.1 | Moderate shift warning |
| `PSI_THRESHOLD_DRIFT` | 0.2 | Drift detected threshold |
| `CONFIDENCE_BASELINE` | 0.65 | Expected avg confidence |
| `CONFIDENCE_DECAY_THRESHOLD` | 0.15 | Max acceptable confidence drop |
| `MONITORING_WINDOW` | 500 | Rolling buffer size |
| `RETRAIN_COOLDOWN_HOURS` | 24 | Min hours between retrain triggers |

---

## 🔁 CI/CD Pipeline

### GitHub Actions Workflows

#### `ci.yml` — Continuous Integration

Triggers on every push to `main`/`develop` and on pull requests.

```
Push/PR → Ruff Lint → Mypy Type Check → Pytest → Docker Build Test ✅
```

#### `docker-build.yml` — Continuous Delivery

Triggers on push to `main` branch or version tags (`v*`).

```
Push to main → Build Docker Image → Tag with SHA → Push to GitHub Container Registry ✅
```

---

## ✅ Testing

The project includes **20 unit tests** across 3 test files:

| File | Tests | Covers |
|------|:-----:|--------|
| `tests/test_api.py` | 8 | Health, metrics, drift status, detect (image + JSON), share-of-shelf, frontend, invalid input |
| `tests/test_inference.py` | 8 | Share of Shelf (empty, single, multiple, sort, top-10, percentages), config validation |
| `tests/test_preprocessing.py` | 10 | Drift monitor (PSI balanced/skewed, confidence averaging, rolling window, retrain logic) |

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ -v --tb=short
```

---

## 📁 Project Structure

```
retail-shelf-detection/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Lint + Test + Docker Build
│       └── docker-build.yml        # Build & Push to GHCR
│
├── airflow/
│   └── dags/
│       ├── ml_pipeline.py          # Training pipeline DAG
│       └── drift_monitor.py        # Drift detection + auto-retrain DAG
│
├── dataset/                        # Training data (not in Docker)
│   ├── train/images/ & labels/
│   ├── valid/images/ & labels/
│   └── test/images/ & labels/
│
├── experiments/
│   └── experiment_log.json         # All training run results
│
├── frontend/
│   ├── index.html                  # 4-tab SPA
│   ├── style.css                   # Dark glassmorphism theme
│   └── app.js                      # Drag-drop, API calls, Chart.js
│
├── models/
│   ├── best.pt                     # Trained YOLOv11m weights (download separately)
│   └── model_card.md               # Google-format model documentation
│
├── notebooks/
│   ├── 01_preprocessing.ipynb      # Data analysis + class rebalancing
│   └── 02_training.ipynb           # YOLOv11m@1280 training + evaluation
│
├── src/
│   ├── api/
│   │   ├── main.py                 # FastAPI app + lifespan
│   │   ├── routes.py               # 5 API endpoints
│   │   └── schemas.py              # Pydantic models
│   ├── inference/
│   │   ├── detector.py             # Thread-safe YOLOv11 singleton
│   │   └── share_of_shelf.py       # SKU distribution analytics
│   ├── monitoring/
│   │   ├── drift_config.py         # PSI thresholds + settings
│   │   └── drift_detector.py       # Rolling buffer + PSI engine
│   └── config.py                   # Central configuration
│
├── tests/
│   ├── test_api.py                 # API endpoint tests
│   ├── test_inference.py           # Inference + Share of Shelf tests
│   └── test_preprocessing.py       # Drift monitoring tests
│
├── .dockerignore
├── .env.example                    # Environment variable template
├── .gitignore
├── Dockerfile                      # Multi-stage production build
├── docker-compose.yml              # API + optional Airflow
├── pyproject.toml                  # Project metadata + tool config
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Dev + test dependencies
└── README.md                       # You are here
```

---

## 📋 Experiment Log

| # | Model | ImgSz | Dataset | Precision | Recall | mAP@50 | mAP@50-95 | Key Observation |
|---|-------|:-----:|---------|:---------:|:------:|:------:|:---------:|-----------------|
| 1 | YOLOv11n | 640 | Original | 74.87% | 66.47% | 76.87% | 46.38% | Baseline — decent P, low R |
| 2 | YOLOv11n | 640 | Original | 67.84% | 66.90% | 69.84% | 44.86% | Lower conf → ❌ No help |
| 3 | YOLOv11n | 640 | Original | 76.88% | 68.78% | 79.21% | 33.54% | Augmentation → moderate gain |
| 4 | YOLOv11m | 640 | Original | 80.99% | 76.58% | 84.52% | 50.89% | Bigger model → ✅ Big jump |
| 5 | YOLOv11m | 1280 | Original | 72.68% | 85.62% | — | — | Higher res → good recall |
| 6 | RT-DETR-L | 640 | Original | 65.55% | 85.76% | 88.36% | 54.09% | Transformer — low precision |
| 7 | RT-DETR-X | 640 | Original | 63.67% | 85.56% | 86.93% | 47.75% | Bigger transformer → overfitting |
| 8 | **YOLOv8m** | **640** | **Stratified** | **81.15%** | **92.75%** | **96.00%** | **69.57%** | **⭐ BEST — stratified split** |
| 9 | YOLOv11m | 640 | Stratified | 81.43% | 90.42% | 93.48% | 64.11% | Strong runner-up |

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- [Ultralytics](https://docs.ultralytics.com/) — YOLOv11 framework
- [Roboflow](https://roboflow.com/) — Dataset hosting and annotation
- [FastAPI](https://fastapi.tiangolo.com/) — High-performance API framework
- [Chart.js](https://www.chartjs.org/) — Frontend visualizations
- [Apache Airflow](https://airflow.apache.org/) — ML pipeline orchestration
