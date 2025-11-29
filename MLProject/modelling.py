import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import argparse
import os

# --- 1. Fungsi Utama untuk Eksperimen ---
# Fungsi ini murni berfokus pada training dan logging ke RUN yang sudah AKTIF
def run_experiment(n_estimators=100, max_depth=None):
    # 1. Setup MLflow
    # PENTING: set_experiment() harus ada di sini untuk mengaitkan RUN ID dengan nama eksperimen.
    mlflow.set_experiment("Placement_CI_Workflow")

    # 2. Load Data (Diasumsikan data berada di MLProject/placement_preprocessing/)
    data_path = 'placement_preprocessing/placement_preprocessed.csv'

    # KITA TIDAK MEMANGGIL mlflow.start_run() di sini.
    # MLflow CLI sudah memulai RUN ID.

    # Load Data
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}. Please ensure it is committed.")
        return

    TARGET_COLUMN = 'PlacementStatus'
    X = df.drop(columns=[TARGET_COLUMN], axis=1)
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. Model Training & Manual Logging ---
    print(f"Starting training run with n_estimators={n_estimators}, max_depth={max_depth}")

    # Model Training
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Manual Logging Parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Manual Logging Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    mlflow.log_metrics(metrics)

    # Log Model Artefact
    mlflow.sklearn.log_model(model, "model")

    # Verifikasi ID Run yang aktif
    try:
        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}. Model dan metrik tercatat.")
    except Exception:
        print("Model dan metrik tercatat, tetapi tidak ada RUN ID aktif yang ditemukan.")


# --- 2. Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Argumen yang akan dipetakan ke parameter MLProject
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)

    args = parser.parse_args()

    # Jalankan fungsi utama
    run_experiment(n_estimators=args.n_estimators, max_depth=args.max_depth)
