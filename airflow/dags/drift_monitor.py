"""
Data Drift Monitor — Apache Airflow DAG

Periodically checks for prediction distribution drift and triggers
auto-retraining when drift is detected.

Schedule: Daily at 2 AM (configurable)

Flow:
  collect_predictions → compute_drift → [drift?]
    → Yes: log_alert → trigger_retrain → evaluate → [improved?]
        → Yes: swap_model
        → No:  keep_current
    → No:  log_healthy
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="drift_monitor",
    default_args=default_args,
    description="Monitor prediction drift and trigger auto-retraining when needed",
    schedule_interval="0 2 * * *",  # Daily at 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "monitoring", "drift"],
)


# ═══════════════════════════════════════════════════════════════════════════
#  TASK DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

def collect_predictions(**kwargs):
    """
    Collect recent prediction logs from the API's drift buffer.
    In production, this would query the drift monitor's rolling buffer
    or a prediction logging database.
    """
    import json
    from pathlib import Path

    drift_log = Path("/app/experiments/drift_log.json")

    if drift_log.exists():
        with open(drift_log, "r") as f:
            events = json.load(f)
        print(f"📊 Found {len(events)} drift events in log")
    else:
        events = []
        print("📊 No drift events logged yet")

    # Query the API for current status
    # In production: response = requests.get("http://api:8000/api/drift-status")
    # Simulated status for DAG demonstration
    status = {
        "status": "HEALTHY",
        "psi_score": 0.05,
        "avg_confidence": 0.62,
        "predictions_logged": 150,
    }

    kwargs["ti"].xcom_push(key="drift_status", value=status)
    print(f"  Status: {status['status']}")
    print(f"  PSI: {status['psi_score']}")
    print(f"  Avg Confidence: {status['avg_confidence']}")
    return True


def compute_drift(**kwargs):
    """Analyze collected predictions and determine drift status."""
    ti = kwargs["ti"]
    status = ti.xcom_pull(task_ids="collect_predictions", key="drift_status")

    psi = status.get("psi_score", 0)
    confidence = status.get("avg_confidence", 0)
    drift_status = status.get("status", "HEALTHY")

    print(f"  PSI Score: {psi:.4f} (threshold: 0.2)")
    print(f"  Avg Confidence: {confidence:.2%} (baseline: 65%)")
    print(f"  Status: {drift_status}")

    ti.xcom_push(key="drift_detected", value=(drift_status == "DRIFT_DETECTED"))
    ti.xcom_push(key="psi_score", value=psi)
    return True


def check_drift(**kwargs):
    """Branch: if drift detected → alert + retrain, else → log healthy."""
    ti = kwargs["ti"]
    drift_detected = ti.xcom_pull(task_ids="compute_drift", key="drift_detected")

    if drift_detected:
        print("⚠️ Drift detected — triggering retraining pipeline")
        return "log_drift_alert"
    else:
        print("✅ No drift detected — system healthy")
        return "log_healthy"


def log_drift_alert(**kwargs):
    """Log the drift alert with timestamp and metrics."""
    ti = kwargs["ti"]
    psi = ti.xcom_pull(task_ids="compute_drift", key="psi_score")

    alert = {
        "timestamp": datetime.now().isoformat(),
        "event": "drift_alert",
        "psi_score": psi,
        "action": "auto_retrain_triggered",
    }
    print(f"🚨 DRIFT ALERT: {alert}")

    # In production: save to database, send Slack notification, etc.
    return True


def trigger_retraining(**kwargs):
    """Trigger the ML training pipeline DAG."""
    from airflow.api.common.trigger_dag import trigger_dag

    print("🔄 Triggering ml_training_pipeline DAG...")

    # In production, this triggers the training DAG:
    # trigger_dag(
    #     dag_id="ml_training_pipeline",
    #     run_id=f"drift_retrain_{datetime.now().strftime('%Y%m%d_%H%M')}",
    #     conf={"triggered_by": "drift_monitor"},
    # )

    print("✅ Retraining triggered")
    kwargs["ti"].xcom_push(key="retrain_triggered", value=True)
    return True


def evaluate_new_model(**kwargs):
    """Compare the newly trained model against the current production model."""
    # In production:
    # current_model = YOLO("/app/models/best.pt")
    # new_model = YOLO("/app/runs/latest/weights/best.pt")
    # Compare metrics on the same test set

    # Simulated comparison
    current_recall = 0.8562
    new_recall = 0.8650  # Simulated improvement

    improved = new_recall > current_recall
    kwargs["ti"].xcom_push(key="improved", value=improved)
    kwargs["ti"].xcom_push(key="new_recall", value=new_recall)

    print(f"  Current model recall: {current_recall:.2%}")
    print(f"  New model recall:     {new_recall:.2%}")
    print(f"  Improved: {'✅ Yes' if improved else '❌ No'}")
    return True


def check_improvement(**kwargs):
    """Branch: swap model if improved, keep current otherwise."""
    ti = kwargs["ti"]
    improved = ti.xcom_pull(task_ids="evaluate_new_model", key="improved")
    return "swap_model" if improved else "keep_current"


def swap_model(**kwargs):
    """Replace production model with the improved version."""
    print("🔄 Swapping production model with new version...")
    # In production:
    # shutil.copy2("runs/latest/weights/best.pt", "/app/models/best.pt")
    # requests.post("http://api:8000/api/reload")
    print("✅ Model swapped successfully")
    return True


def keep_current(**kwargs):
    """Keep the current model — new model did not improve performance."""
    print("ℹ️ Keeping current model — retrained version did not improve metrics")
    return True


def log_healthy(**kwargs):
    """Log that the system is healthy with no drift detected."""
    print("✅ System healthy — no drift detected. No action needed.")
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  DAG STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════

t_collect = PythonOperator(
    task_id="collect_predictions",
    python_callable=collect_predictions,
    dag=dag,
)

t_compute = PythonOperator(
    task_id="compute_drift",
    python_callable=compute_drift,
    dag=dag,
)

t_check_drift = BranchPythonOperator(
    task_id="check_drift",
    python_callable=check_drift,
    dag=dag,
)

t_alert = PythonOperator(
    task_id="log_drift_alert",
    python_callable=log_drift_alert,
    dag=dag,
)

t_retrain = PythonOperator(
    task_id="trigger_retraining",
    python_callable=trigger_retraining,
    dag=dag,
)

t_evaluate = PythonOperator(
    task_id="evaluate_new_model",
    python_callable=evaluate_new_model,
    dag=dag,
)

t_check_improve = BranchPythonOperator(
    task_id="check_improvement",
    python_callable=check_improvement,
    dag=dag,
)

t_swap = PythonOperator(
    task_id="swap_model",
    python_callable=swap_model,
    dag=dag,
)

t_keep = PythonOperator(
    task_id="keep_current",
    python_callable=keep_current,
    dag=dag,
)

t_healthy = PythonOperator(
    task_id="log_healthy",
    python_callable=log_healthy,
    dag=dag,
)

t_end = EmptyOperator(
    task_id="end",
    trigger_rule="none_failed_min_one_success",
    dag=dag,
)

# Flow: collect → compute → check_drift
t_collect >> t_compute >> t_check_drift

# Drift detected path
t_check_drift >> t_alert >> t_retrain >> t_evaluate >> t_check_improve
t_check_improve >> t_swap >> t_end
t_check_improve >> t_keep >> t_end

# No drift path
t_check_drift >> t_healthy >> t_end
