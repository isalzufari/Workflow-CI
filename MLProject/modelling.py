import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import argparse
import os

def run_experiment(n_estimators=100, max_depth=None):
    mlflow.set_experiment("Placement_CI_Workflow")
    data_path = 'placement_preprocessing/placement_preprocessed.csv'
    
    if mlflow.active_run() is None:
        mlflow.start_run() 

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}. Please ensure it is committed.")
        return

    TARGET_COLUMN = 'PlacementStatus'
    X = df.drop(columns=[TARGET_COLUMN], axis=1)
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Starting training run with n_estimators={n_estimators}, max_depth={max_depth}")
        
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
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
    
    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}. Model dan metrik tercatat.")
    
    # mlflow.end_run() 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    
    args = parser.parse_args()
    
    run_experiment(n_estimators=args.n_estimators, max_depth=args.max_depth)
