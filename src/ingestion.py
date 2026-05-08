import os
import yaml
from google.cloud import storage


def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def download_from_gcs(project_id, bucket_name, source_blob, destination):
    try:
        client = storage.Client(project=project_id)

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob)

        blob.download_to_filename(destination)

        print(f"Downloaded {source_blob} → {destination}")

    except Exception as e:
        print(f"Error downloading {source_blob}: {e}")
        raise


def create_directories(paths):
    for path in paths:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)


def run():
    print("Starting data ingestion...")

    config = load_config()

    # Create required directories
    create_directories([
        config['paths']['train_local'],
        config['paths']['test_local']
    ])

    # Download train file
    download_from_gcs(
        config['gcp']['project_id'],
        config['gcp']['bucket_name'],
        config['gcp']['train_file'],
        config['paths']['train_local']
    )

    # Download test file
    download_from_gcs(
        config['gcp']['project_id'],
        config['gcp']['bucket_name'],
        config['gcp']['test_file'],
        config['paths']['test_local']
    )

    print("✅ Data ingestion completed successfully!")


if __name__ == "__main__":
    run()