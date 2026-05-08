import pandas as pd
import yaml
import os
import joblib
import json
from google.cloud import storage
import tempfile

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ----------------------------
# Load Config
# ----------------------------
def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


# ----------------------------
# Load Data
# ----------------------------
def load_data(path):
    return pd.read_csv(path)


# ----------------------------
# Split Data
# ----------------------------
def split_data(df, config):
    X = df.drop(columns=["Response"])
    y = df["Response"]

    return train_test_split(
        X,
        y,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state'],
        stratify=y
    )


# ----------------------------
# Train Model
# ----------------------------
def train_model(X_train, y_train, y_full, config):
    model_type = config['model']['type']

    if model_type == "xgboost":
        from xgboost import XGBClassifier

        params = config['model']['xgboost']

        # Handle imbalance
        neg = (y_full == 0).sum()
        pos = (y_full == 1).sum()
        scale_pos_weight = neg / pos

        model = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=config['training']['random_state'],
            n_jobs=-1
        )

    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        params = config['model']['random_forest']

        model = RandomForestClassifier(
            **params,
            class_weight='balanced',
            random_state=config['training']['random_state'],
            n_jobs=-1
        )

    else:
        raise ValueError("Unsupported model type")

    model.fit(X_train, y_train)
    return model


# ----------------------------
# Evaluate Model
# ----------------------------
def evaluate_model(model, X_val, y_val, config):
    probs = model.predict_proba(X_val)[:, 1]

    threshold = config['training']['threshold']
    preds = (probs > threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds, zero_division=0),
        "recall": recall_score(y_val, preds, zero_division=0),
        "f1_score": f1_score(y_val, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_val, probs)
    }

    cm = confusion_matrix(y_val, preds).tolist()

    return metrics, cm


# ----------------------------
# Save Model
# ----------------------------
# def save_model(model, path):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     joblib.dump(model, path)

def save_model(model, gcs_path):
    client = storage.Client()
    bucket_name = gcs_path.split("/")[2]
    blob_path = "/".join(gcs_path.split("/")[3:])

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    with tempfile.NamedTemporaryFile() as tmp:
        joblib.dump(model, tmp.name)
        blob.upload_from_filename(tmp.name)

    print(f"Model saved to {gcs_path}")


# ----------------------------
# Save Metrics
# ----------------------------
# def save_metrics(metrics, cm, path):
#     os.makedirs(os.path.dirname(path), exist_ok=True)

#     output = {
#         "metrics": metrics,
#         "confusion_matrix": cm
#     }

#     with open(path, "w") as f:
#         json.dump(output, f, indent=4)

def save_metrics(metrics, cm, gcs_path):
    from google.cloud import storage
    import tempfile
    import json

    client = storage.Client()
    bucket_name = gcs_path.split("/")[2]
    blob_path = "/".join(gcs_path.split("/")[3:])

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    output = {
        "metrics": metrics,
        "confusion_matrix": cm
    }

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
        json.dump(output, tmp)
        tmp.flush()
        blob.upload_from_filename(tmp.name)

    print(f"Metrics saved to {gcs_path}")


# ----------------------------
# Main Run
# ----------------------------
def run():
    print("🚀 Training started...")

    config = load_config()

    # Load processed data
    df = load_data(config['paths']['processed_train'])

    # Show class distribution
    print("\nClass distribution:")
    print(df['Response'].value_counts(normalize=True))

    # Split
    X_train, X_val, y_train, y_val = split_data(df, config)

    # Train
    model = train_model(X_train, y_train, df["Response"], config)

    # Evaluate
    metrics, cm = evaluate_model(model, X_val, y_val, config)

    # Print metrics
    print("\n📊 Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n📉 Confusion Matrix:")
    print(cm)

    # Save outputs
    save_model(model, config['paths']['model_path'])
    save_metrics(metrics, cm, config['paths']['metrics_path'])

    print("\n✅ Training completed successfully!")


if __name__ == "__main__":
    run()