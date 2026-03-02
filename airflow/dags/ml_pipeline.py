"""
ML Training Pipeline — Apache Airflow DAG

Orchestrates the end-to-end model training workflow:
  validate_data → preprocess → train_model → evaluate_model → deploy_model

Schedule: Manual trigger (or set schedule_interval for periodic retraining)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator


# ── DAG Configuration ──
default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    description="End-to-end YOLOv11 training pipeline: validate → preprocess → train → evaluate → deploy",
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "training", "yolov11"],
)


# ═══════════════════════════════════════════════════════════════════════════
#  TASK DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

def validate_data(**kwargs):
    """Check dataset integrity — file counts, label format, class coverage."""
    import os
    from pathlib import Path

    data_dir = Path(os.getenv("DATA_DIR", "/app/data"))
    errors = []

    for split in ["train", "valid", "test"]:
        img_dir = data_dir / split / "images"
        lbl_dir = data_dir / split / "labels"

        if not img_dir.exists():
            errors.append(f"Missing: {img_dir}")
            continue

        imgs = list(img_dir.glob("*.jpg"))
        lbls = list(lbl_dir.glob("*.txt"))

        if len(imgs) == 0:
            errors.append(f"{split}: No images found")
        if abs(len(imgs) - len(lbls)) > 5:
            errors.append(f"{split}: Image/label mismatch ({len(imgs)} imgs, {len(lbls)} labels)")

        print(f"  ✅ {split}: {len(imgs)} images, {len(lbls)} labels")

    if errors:
        raise ValueError(f"Data validation failed: {errors}")

    print("✅ Data validation passed")
    return True


def preprocess(**kwargs):
    """Run class rebalancing pipeline on the training data."""
    print("Running class rebalancing pipeline...")
    print("  → Rescuing zero-train classes from validation set")
    print("  → Oversampling minority classes to ≥30 instances")
    print("  → Applying targeted augmentations")

    # In production, this would call:
    # from src.data.preprocessing import rebalance_dataset
    # rebalance_dataset(data_dir="/app/data")

    print("✅ Preprocessing complete")
    return True


def train_model(**kwargs):
    """Train YOLOv11m with optimal hyperparameters."""
    print("Starting YOLOv11m training...")
    print("  → Epochs: 100")
    print("  → Image size: 1280")
    print("  → Augmentation: enhanced")

    # In production, this would call:
    # from ultralytics import YOLO
    # model = YOLO("yolo11m.pt")
    # model.train(data="data.yaml", epochs=100, imgsz=1280, ...)

    print("✅ Training complete")
    return True


def evaluate_model(**kwargs):
    """Validate trained model on test set and check metrics."""
    print("Evaluating model on test set...")

    # In production, this would load the model and validate:
    # metrics = model.val(data="data.yaml", split="test")
    # recall = metrics.box.mr

    # Simulated metrics for DAG demonstration
    recall = 0.8562
    precision = 0.7268

    kwargs["ti"].xcom_push(key="recall", value=recall)
    kwargs["ti"].xcom_push(key="precision", value=precision)

    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print("✅ Evaluation complete")
    return True


def check_metrics(**kwargs):
    """Branch: deploy if recall > 80%, otherwise alert."""
    ti = kwargs["ti"]
    recall = ti.xcom_pull(task_ids="evaluate_model", key="recall")

    if recall and recall > 0.80:
        print(f"✅ Recall {recall:.2%} > 80% — proceeding to deploy")
        return "deploy_model"
    else:
        print(f"⚠️ Recall {recall:.2%} < 80% — alerting team")
        return "alert_team"


def deploy_model(**kwargs):
    """Copy best.pt to serving directory and signal API to reload."""
    import shutil

    print("Deploying model...")
    # In production:
    # shutil.copy2("runs/train/weights/best.pt", "/app/models/best.pt")
    # requests.post("http://api:8000/api/reload")

    print("✅ Model deployed successfully")
    return True


def alert_team(**kwargs):
    """Send alert that model did not meet minimum recall threshold."""
    ti = kwargs["ti"]
    recall = ti.xcom_pull(task_ids="evaluate_model", key="recall")
    print(f"🚨 ALERT: Model recall ({recall:.2%}) below threshold (80%)")
    print("   Manual review required before deployment")
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  DAG STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════

t_validate = PythonOperator(
    task_id="validate_data",
    python_callable=validate_data,
    dag=dag,
)

t_preprocess = PythonOperator(
    task_id="preprocess",
    python_callable=preprocess,
    dag=dag,
)

t_train = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag,
)

t_evaluate = PythonOperator(
    task_id="evaluate_model",
    python_callable=evaluate_model,
    dag=dag,
)

t_check = BranchPythonOperator(
    task_id="check_metrics",
    python_callable=check_metrics,
    dag=dag,
)

t_deploy = PythonOperator(
    task_id="deploy_model",
    python_callable=deploy_model,
    dag=dag,
)

t_alert = PythonOperator(
    task_id="alert_team",
    python_callable=alert_team,
    dag=dag,
)

t_end = EmptyOperator(
    task_id="end",
    trigger_rule="none_failed_min_one_success",
    dag=dag,
)

# Pipeline flow
t_validate >> t_preprocess >> t_train >> t_evaluate >> t_check
t_check >> t_deploy >> t_end
t_check >> t_alert >> t_end
