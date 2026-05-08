from google.cloud import aiplatform

aiplatform.init(
    project="mlopsproject-493611",
    location="us-central1"
)

job = aiplatform.PipelineJob(
    display_name="mlops-pipeline-run",
    template_path="pipeline.json"
)

job.run()