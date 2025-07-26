# promote model

import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

def promote_model():
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    dagshub_user = os.getenv("DAGSHUB_USER_NAME")
    repo_owner = dagshub_user
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    if not dagshub_token or not dagshub_user or not repo_name:
        raise EnvironmentError("DAGSHUB_TOKEN, DAGSHUB_USER_NAME, or DAGSHUB_REPO_NAME not set.")

    # Authenticated MLflow URI
    mlflow.set_tracking_uri(
        f"https://{dagshub_user}:{dagshub_token}@dagshub.com/{repo_owner}/{repo_name}.mlflow"
    )

    client = mlflow.MlflowClient()

    model_name = "my_model"
    # Get the latest version in staging
    latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version

    # Archive the current production model
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote the new model to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()