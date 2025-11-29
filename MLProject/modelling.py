import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import argparse
import os

# --- 1. Fungsi Utama untuk Eksperimen ---
def run_experiment(n_estimators=100, max_depth=None):
    
    # 1. Dapatkan Run ID yang dibuat oleh MLflow CLI
    # MLflow CLI mengatur environment variable MLFLOW_RUN_ID
    cli_run_id = os.environ.get("MLFLOW_RUN_ID")
    
    # KITA HARUS BERGABUNG DENGAN RUN INI
    # Gunakan with start_run() dengan run_id yang ditentukan untuk bergabung dengan run yang sudah ada
    # Ini memastikan bahwa semua logging terjadi di RUN ID yang sama (b51a350fea1247cba2e6a2da33b2f8e3)
    with mlflow.start_run(run_id=cli_run_id) as run:
        
        # 2. Set Experiment (setelah run dimulai)
        mlflow.set_experiment("Placement_CI_Workflow")

        # 3. Load Data
        data_path = 'placement_preprocessing/placement_preprocessed.csv'
        
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"Error: Data file not found at {data_path}. Please ensure it is committed.")
            return

        TARGET_COLUMN = 'PlacementStatus'
        X = df.drop(columns=[TARGET_COLUMN], axis=1)
        y = df[TARGET_COLUMN]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- 4. Model Training & Logging ---
        print(f"Starting training run with n_estimators={n_estimators}, max_depth={max_depth}")
            
        # Model Training
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Logging Parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Logging Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        mlflow.log_metrics(metrics)
        
        # Log Model Artefact
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Run ID: {run.info.run_id}. Model dan metrik tercatat dengan sukses.")


# --- 2. Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    
    args = parser.parse_args()
    
    run_experiment(n_estimators=args.n_estimators, max_depth=args.max_depth)
