import os
import mlflow

dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Milan123-star"
repo_name = "stress_project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

client=mlflow.MlflowClient()

def promote_model():
    model_name='model'
    latest_ver_sta=client.get_latest_versions(model_name,stages=["Staging"])[0].version
    prod_ver=client.get_latest_versions(model_name,stages=["Production"])

    for version in prod_ver:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage='Archived'
        )
    client.transition_model_version_stage(
        name=model_name,
        version=latest_ver_sta,
        stage='Production'
    )

def main():
    promote_model()
main()        