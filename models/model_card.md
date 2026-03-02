# Model Card: Retail Shelf Product Detector

## Model Details

| Field | Value |
|-------|-------|
| **Model Name** | ShelfVision Retail Detector |
| **Architecture** | YOLOv8m / YOLOv11m (Ultralytics) |
| **Version** | 2.0.0 |
| **Input Size** | 640 x 640 pixels |
| **Parameters** | ~25M (YOLOv8m) / ~20M (YOLOv11m) |
| **Task** | Object Detection (multi-class) |
| **Classes** | 76 retail product SKUs |
| **Framework** | PyTorch + Ultralytics |
| **License** | MIT |

## Intended Use

### Primary Use Case
- **Retail shelf monitoring**: Detecting and counting individual products on supermarket shelves from overhead or front-facing camera images.
- **Share of Shelf analytics**: Computing the percentage distribution of each product SKU to understand shelf space allocation.

### Out-of-Scope Uses
- Detecting products in non-retail environments (warehouses, homes)
- Identifying product defects or expiration dates
- Real-time video processing (model optimized for single-image inference)

## Training Data

### Dataset Overview
| Split | Images | Annotations | Avg Objects/Image |
|-------|-------:|------------:|------------------:|
| Train | 924 | 3,979 | 4.3 |
| Valid | 40 | 199 | 5.0 |
| Test | 35 | 145 | 4.1 |

- **Source**: Roboflow (CC BY 4.0 license)
- **Image Resolution**: 640 x 640 (auto-oriented, resized)
- **Annotation Format**: YOLO (class_id, x_center, y_center, width, height)
- **Classes**: 76 unique product SKUs (labeled q1 through q299)

### Data Preprocessing
- **Class imbalance detected**: 221.5x ratio between most common (q280: 443) and least common (q178: 2) classes
- **4 classes with zero training samples** in the original split
- **Stratified splitting**: All images pooled and re-split 80/10/10 using stratified sampling based on dominant class per image, ensuring every class is represented in train
- **Oversampling**: Minority classes oversampled to a minimum threshold of 15 instances (in oversample variants)
- **Targeted augmentation**: Brightness, contrast, horizontal flip, color jitter, Gaussian blur, and noise applied to minority class images

## Training Configuration

### Best Model (YOLOv8m Stratified + Oversample)

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Image Size | 640 |
| Batch Size | 16 |
| Learning Rate | 0.01 (initial), 0.01 (final ratio) |
| Optimizer | SGD (default) |
| Mosaic | 1.0 |
| Mixup | 0.15 |
| Copy-Paste | 0.1 |
| Degrees | 15.0 |
| Scale | 0.5 |

## Evaluation Results

All experiments were evaluated on their respective test splits.

### Phase 1: Architecture and Hyperparameter Exploration (Original Dataset)

| # | Model | ImgSz | Precision | Recall | mAP@50 | mAP@50-95 |
|---|-------|:-----:|:---------:|:------:|:------:|:---------:|
| 1 | YOLOv11n (baseline) | 640 | 74.87% | 66.47% | 76.87% | 46.38% |
| 2 | YOLOv11n (conf=0.15) | 640 | 67.84% | 66.90% | 69.84% | 44.86% |
| 3 | YOLOv11n + augmentation | 640 | 76.88% | 68.78% | 79.21% | 33.54% |
| 4 | YOLOv11m | 640 | 80.99% | 76.58% | 84.52% | 50.89% |
| 5 | YOLOv11m | 1280 | 72.68% | 85.62% | -- | -- |
| 6 | RT-DETR-L | 640 | 65.55% | 85.76% | 88.36% | 54.09% |
| 7 | RT-DETR-X | 640 | 63.67% | 85.56% | 86.93% | 47.75% |

### Phase 2: Data-Centric Improvements (Stratified Split)

| # | Model | Dataset Variant | Precision | Recall | mAP@50 | mAP@50-95 |
|---|-------|-----------------|:---------:|:------:|:------:|:---------:|
| 8 | **YOLOv8m** | **Stratified + Oversample** | **81.15%** | **92.75%** | **96.00%** | **69.57%** |
| 9 | YOLOv11m | Stratified + Oversample | 81.43% | 90.42% | 93.48% | 64.11% |
| 10 | YOLOv11m + SGD | Stratified | 82.48% | 88.98% | 93.68% | 58.32% |
| 11 | YOLOv8m + SGD | Stratified (no oversample) | 83.45% | 89.39% | 95.66% | 67.85% |

### Selected Model

**YOLOv8m Stratified + Oversample** (Experiment #8) -- best balance of recall (92.75%) and mAP@50 (96.00%).

**Notable runner-up**: YOLOv8m Stratified + SGD without oversample (Experiment #11) achieved the highest precision (83.45%) with strong mAP@50-95 (67.85%).

### Key Findings

1. **Data quality matters more than model complexity**: Stratified splitting provided the largest single improvement across all experiments.
2. **YOLOv8m outperformed YOLOv11m** on this dataset size (999 images, 76 classes).
3. **Oversampling helped recall** (+3.36% from 89.39% to 92.75%) at a small precision cost (-2.30%).
4. **SGD optimizer improved precision** but slightly reduced recall compared to default optimizer.

## Limitations

1. **Small dataset**: Only 924 training images (999 total) -- larger datasets would improve generalizability
2. **Class imbalance**: Despite rebalancing, some SKUs have very few examples (may have lower per-class recall)
3. **Domain specificity**: Trained on a specific store's product set -- will not generalize to other retailers without fine-tuning
4. **Image quality dependency**: Performance degrades on blurry, poorly lit, or heavily occluded shelf images
5. **Fixed SKU set**: Cannot detect products not in the 76-class training vocabulary

## Ethical Considerations

- **Privacy**: No personal data is collected or processed. Model operates on product images only.
- **Bias**: Model may perform unevenly across SKU classes due to training data imbalance. Underrepresented classes may have higher miss rates.
- **Environmental impact**: Training was performed on cloud GPUs (Kaggle T4x2, Google Colab T4). Inference runs on CPU or single GPU.

## Monitoring

- **Data drift detection**: PSI (Population Stability Index) monitors prediction distribution shifts against training baseline
- **Confidence tracking**: Rolling average confidence is tracked to detect model degradation
- **Auto-retrain trigger**: When PSI > 0.2 or confidence drops >15%, the system flags for retraining
