# Model Card: Retail Shelf Product Detector

## Model Details

| Field | Value |
|-------|-------|
| **Model Name** | ShelfVision YOLOv11m |
| **Architecture** | YOLOv11m (Ultralytics) |
| **Version** | 1.0.0 |
| **Input Size** | 1280 × 1280 pixels |
| **Parameters** | ~20M |
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
- **Image Resolution**: 640 × 640 (auto-oriented, resized)
- **Annotation Format**: YOLO (class_id, x_center, y_center, width, height)
- **Classes**: 76 unique product SKUs (labeled q1 through q299)

### Data Preprocessing
- **Class imbalance detected**: 221.5× ratio between most common (q280: 443) and least common (q178: 2) classes
- **Rebalancing applied**: Rescued zero-train classes from validation set, oversampled minority classes to ≥30 instances, applied targeted augmentations

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Image Size | 1280 |
| Batch Size | 8 |
| Learning Rate | 0.005 (initial), 0.01 (final ratio) |
| Optimizer | SGD (Ultralytics default) |
| Mosaic | 1.0 |
| Mixup | 0.15 |
| Copy-Paste | 0.1 |
| Degrees | 15.0 |
| Scale | 0.7 |

## Evaluation Results

| Model | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|----------:|-------:|-------:|----------:|
| Baseline YOLOv11n | 74.87% | 66.47% | 76.87% | 46.38% |
| YOLOv11n + Augmentation | 76.88% | 68.78% | 79.21% | 33.54% |
| YOLOv11m | 80.99% | 76.58% | 84.52% | 50.89% |
| **YOLOv11m @1280** | **72.68%** | **85.62%** | **87.99** | **54.73** |
| RT-DETR-L | 65.55% | 85.76% | 88.36% | 54.09% |
| RT-DETR-X | 63.67% | 85.56% | 86.93% | 47.75% |

**Selected Model**: YOLOv11m @1280 — best balance of recall (85.62%) and precision (72.68%).

## Limitations

1. **Small dataset**: Only 924 training images — larger datasets would improve generalizability
2. **Class imbalance**: Despite rebalancing, some SKUs have very few examples (may have lower per-class recall)
3. **Domain specificity**: Trained on a specific store's product set — will not generalize to other retailers without fine-tuning
4. **Image quality dependency**: Performance degrades on blurry, poorly lit, or heavily occluded shelf images
5. **Fixed SKU set**: Cannot detect products not in the 76-class training vocabulary

## Ethical Considerations

- **Privacy**: No personal data is collected or processed. Model operates on product images only.
- **Bias**: Model may perform unevenly across SKU classes due to training data imbalance. Underrepresented classes may have higher miss rates.
- **Environmental impact**: Training was performed on cloud GPUs (Kaggle P100/T4). Inference runs on CPU or single GPU.

## Monitoring

- **Data drift detection**: PSI (Population Stability Index) monitors prediction distribution shifts against training baseline
- **Confidence tracking**: Rolling average confidence is tracked to detect model degradation
- **Auto-retrain trigger**: When PSI > 0.2 or confidence drops >15%, system flags for retraining
