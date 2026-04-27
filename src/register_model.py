import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.metrics import accuracy_score
from app_logging import *
import os
import yaml
import pickle
import dagshub
import mlflow
import mlflow.sklearn
import json

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

def load_model_info(file_path):
    with open (file_path,'r') as file:
        model_info=json.load(file)
        return model_info
def register_model(model_name,model_info):
    model_uri= model_info["model_uri"]
    model_version=mlflow.register_model(model_uri,model_name)
    client=mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging"
    )    
def main():
    file_path=r'C:\Users\milan\Desktop\stress_project\reports\experiment.json'
    model_info=load_model_info(file_path)
    model_name="model"
    register_model(model_name,model_info)
main()    
