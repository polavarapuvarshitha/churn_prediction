
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


def train_and_track(run_name="RandomForest_Baseline", params=None):

    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "class_weight": "balanced"
        }

    print(f"\n--- Starting MLflow Run: {run_name} ---")

    # 1. Load Data
    X_train = np.load("data/processed/X_train_final.npy")
    X_test = np.load("data/processed/X_test_final.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    # 2. Set MLflow Experiment
    mlflow.set_experiment("Telco_Churn_Prediction")

    with mlflow.start_run(run_name=run_name):

        # 3. Log Parameters
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForest")

        # 4. Train Model
        print("Training Random Forest model...")

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # 5. Make Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # 6. Calculate Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

        # 7. Log Metrics
        mlflow.log_metrics(metrics)

        print(
            f"Metrics logged: "
            f"Accuracy = {metrics['accuracy']:.4f} | "
            f"Precision = {metrics['precision']:.4f} | "
            f"Recall = {metrics['recall']:.4f} | "
            f"F1 = {metrics['f1_score']:.4f} | "
            f"ROC-AUC = {metrics['roc_auc']:.4f}"
        )

        # 8. Create Artifacts Folder
        os.makedirs("artifacts", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        # 9. Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            ax=ax_cm,
            cmap="Blues"
        )

        ax_cm.set_title(f"Confusion Matrix - {run_name}")

        cm_path = "artifacts/confusion_matrix.png"

        fig_cm.savefig(
            cm_path,
            bbox_inches="tight"
        )

        plt.close(fig_cm)

        mlflow.log_artifact(
            cm_path,
            artifact_path="plots"
        )

        # 10. ROC Curve
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))

        RocCurveDisplay.from_predictions(
            y_test,
            y_prob,
            ax=ax_roc
        )

        ax_roc.set_title(f"ROC Curve - {run_name}")

        roc_path = "artifacts/roc_curve.png"

        fig_roc.savefig(
            roc_path,
            bbox_inches="tight"
        )

        plt.close(fig_roc)

        mlflow.log_artifact(
            roc_path,
            artifact_path="plots"
        )

        # 11. Log Dataset Metadata
        metadata_path = "data/processed/dataset_metadata.json"

        if os.path.exists(metadata_path):
            mlflow.log_artifact(
                metadata_path,
                artifact_path="metadata"
            )

        # 12. Log Model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle"
        )

        # 13. Save Local Model Backup
        joblib.dump(
            model,
            "models/random_forest_model.pkl"
        )

        print(f"Run '{run_name}' successfully tracked!")


# Main Program
if __name__ == "__main__":

    print("Starting MLflow Experiment Tracking...")

    # Baseline Random Forest Run
    train_and_track(
        run_name="RandomForest"
    )

    print("\nMLflow tracking completed successfully!")
