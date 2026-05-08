# from kfp.v2.dsl import component, pipeline
# from kfp.v2 import compiler

# BUCKET = "mlops-bkt-1"


# # ---------------- INGESTION ---------------- #

# @component(
#     base_image="python:3.10",
#     packages_to_install=["google-cloud-storage"]
# )
# def ingestion_component():
#     from google.cloud import storage

#     print("Starting ingestion...")

#     client = storage.Client()
#     bucket = client.bucket("mlops-bkt-1")

#     # Download raw data
#     bucket.blob("data/train.csv").download_to_filename("/tmp/train.csv")
#     bucket.blob("data/test.csv").download_to_filename("/tmp/test.csv")

#     # Upload to pipeline folder
#     bucket.blob("pipeline/train.csv").upload_from_filename("/tmp/train.csv")
#     bucket.blob("pipeline/test.csv").upload_from_filename("/tmp/test.csv")

#     print("Ingestion completed")


# # ---------------- PREPROCESSING ---------------- #

# @component(
#     base_image="python:3.10",
#     packages_to_install=["pandas", "google-cloud-storage"]
# )
# def preprocessing_component():
#     import pandas as pd
#     from google.cloud import storage

#     print("Starting preprocessing...")

#     client = storage.Client()
#     bucket = client.bucket("mlops-bkt-1")

#     # Download from GCS
#     bucket.blob("pipeline/train.csv").download_to_filename("/tmp/train.csv")
#     bucket.blob("pipeline/test.csv").download_to_filename("/tmp/test.csv")

#     def preprocess(df):
#         if 'id' in df.columns:
#             df = df.drop(columns=['id'])

#         df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
#         df['Vehicle_Damage'] = df['Vehicle_Damage'].map({'Yes': 1, 'No': 0})

#         df['Vehicle_Age'] = df['Vehicle_Age'].map({
#             '< 1 Year': 0,
#             '1-2 Year': 1,
#             '> 2 Years': 2
#         })

#         return df

#     train = pd.read_csv("/tmp/train.csv")
#     test = pd.read_csv("/tmp/test.csv")

#     train = preprocess(train)
#     test = preprocess(test)

#     train.to_csv("/tmp/train_processed.csv", index=False)
#     test.to_csv("/tmp/test_processed.csv", index=False)

#     # Upload processed files
#     bucket.blob("pipeline/train_processed.csv").upload_from_filename("/tmp/train_processed.csv")
#     bucket.blob("pipeline/test_processed.csv").upload_from_filename("/tmp/test_processed.csv")

#     print("Preprocessing completed")


# # ---------------- TRAINING ---------------- #

# @component(
#     base_image="python:3.10",
#     packages_to_install=[
#         "pandas",
#         "scikit-learn",
#         "xgboost",
#         "joblib",
#         "google-cloud-storage"
#     ]
# )
# def training_component():
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score
#     from xgboost import XGBClassifier
#     import joblib
#     import tempfile
#     from google.cloud import storage

#     print("Starting training...")

#     client = storage.Client()
#     bucket = client.bucket("mlops-bkt-1")

#     # Download processed data
#     bucket.blob("pipeline/train_processed.csv").download_to_filename("/tmp/train.csv")

#     df = pd.read_csv("/tmp/train.csv")

#     X = df.drop(columns=["Response"])
#     y = df["Response"]

#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )

#     model = XGBClassifier(n_estimators=100)
#     model.fit(X_train, y_train)

#     preds = model.predict(X_val)
#     print("Validation Accuracy:", accuracy_score(y_val, preds))

#     # Save model to GCS
#     with tempfile.NamedTemporaryFile() as tmp:
#         joblib.dump(model, tmp.name)
#         bucket.blob("models/model.pkl").upload_from_filename(tmp.name)

#     print("Model saved to GCS")


# # ---------------- MODEL REGISTRATION ---------------- #

# @component(
#     base_image="python:3.10",
#     packages_to_install=["google-cloud-aiplatform"]
# )
# def register_model_component():
#     from google.cloud import aiplatform

#     print("Registering model...")

#     aiplatform.init(
#         project="mlopsproject-493611",
#         location="us-central1"
#     )

#     model = aiplatform.Model.upload(
#         display_name="vehicle-insurance-model",
#         artifact_uri="gs://mlops-bkt-1/models/",
#         serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
#     )

#     print("Model registered:", model.resource_name)


# # ---------------- PIPELINE ---------------- #

# @pipeline(name="mlops-pipeline")
# def ml_pipeline():

#     ingestion = ingestion_component()

#     preprocessing = preprocessing_component()
#     preprocessing.after(ingestion)

#     training = training_component()
#     training.after(preprocessing)

#     register = register_model_component()
#     register.after(training)


# # ---------------- COMPILE ---------------- #

# if __name__ == "__main__":
#     compiler.Compiler().compile(
#         pipeline_func=ml_pipeline,
#         package_path="pipeline.json"
#     )




from kfp.v2.dsl import component, pipeline
from kfp.v2 import compiler

BUCKET = "mlops-bkt-1"


# ---------------- INGESTION ---------------- #

@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-storage"]
)
def ingestion_component():
    from google.cloud import storage

    print("Starting ingestion...")

    client = storage.Client()
    bucket = client.bucket("mlops-bkt-1")

    # Download raw data
    bucket.blob("data/train.csv").download_to_filename("/tmp/train.csv")
    bucket.blob("data/test.csv").download_to_filename("/tmp/test.csv")

    # Upload to pipeline folder
    bucket.blob("pipeline/train.csv").upload_from_filename("/tmp/train.csv")
    bucket.blob("pipeline/test.csv").upload_from_filename("/tmp/test.csv")

    print("Ingestion completed")


# ---------------- PREPROCESSING ---------------- #

@component(
    base_image="python:3.10",
    packages_to_install=["pandas", "google-cloud-storage"]
)
def preprocessing_component():
    import pandas as pd
    from google.cloud import storage

    print("Starting preprocessing...")

    client = storage.Client()
    bucket = client.bucket("mlops-bkt-1")

    # Download from GCS
    bucket.blob("pipeline/train.csv").download_to_filename("/tmp/train.csv")
    bucket.blob("pipeline/test.csv").download_to_filename("/tmp/test.csv")

    def preprocess(df):
        if 'id' in df.columns:
            df = df.drop(columns=['id'])

        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        df['Vehicle_Damage'] = df['Vehicle_Damage'].map({'Yes': 1, 'No': 0})

        df['Vehicle_Age'] = df['Vehicle_Age'].map({
            '< 1 Year': 0,
            '1-2 Year': 1,
            '> 2 Years': 2
        })

        return df

    train = pd.read_csv("/tmp/train.csv")
    test = pd.read_csv("/tmp/test.csv")

    train = preprocess(train)
    test = preprocess(test)

    train.to_csv("/tmp/train_processed.csv", index=False)
    test.to_csv("/tmp/test_processed.csv", index=False)

    # Upload processed files
    bucket.blob("pipeline/train_processed.csv").upload_from_filename("/tmp/train_processed.csv")
    bucket.blob("pipeline/test_processed.csv").upload_from_filename("/tmp/test_processed.csv")

    print("Preprocessing completed")


# ---------------- TRAINING ---------------- #

@component(
    base_image="python:3.10",
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "xgboost",
        "google-cloud-storage"
    ]
)
def training_component():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier
    import tempfile
    from google.cloud import storage

    print("Starting training...")

    client = storage.Client()
    bucket = client.bucket("mlops-bkt-1")

    # Download processed data
    bucket.blob("pipeline/train_processed.csv").download_to_filename("/tmp/train.csv")

    df = pd.read_csv("/tmp/train.csv")

    X = df.drop(columns=["Response"])
    y = df["Response"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, preds))

    # ✅ Save model in XGBoost native format
    with tempfile.NamedTemporaryFile(suffix=".bst") as tmp:
        model.save_model(tmp.name)
        bucket.blob("models/model.bst").upload_from_filename(tmp.name)

    print("Model saved to GCS (XGBoost format)")


# ---------------- MODEL REGISTRATION ---------------- #

@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-aiplatform"]
)
def register_model_component():
    from google.cloud import aiplatform

    print("Registering model...")

    aiplatform.init(
        project="mlopsproject-493611",
        location="us-central1"
    )

    model = aiplatform.Model.upload(
        display_name="vehicle-insurance-model",
        artifact_uri="gs://mlops-bkt-1/models/",
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-0:latest"
    )

    print("Model registered:", model.resource_name)


# ---------------- PIPELINE ---------------- #

@pipeline(name="mlops-pipeline")
def ml_pipeline():

    ingestion = ingestion_component()

    preprocessing = preprocessing_component()
    preprocessing.after(ingestion)

    training = training_component()
    training.after(preprocessing)

    register = register_model_component()
    register.after(training)


# ---------------- COMPILE ---------------- #

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=ml_pipeline,
        package_path="pipeline.json"
    )