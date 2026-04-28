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
                        
X=pd.read_csv(os.path.join("data","feature_engg","X.csv"))
y=pd.read_csv(os.path.join("data","feature_engg","y.csv")).values.ravel()

def load_params(file_path):
    try:
        with open(file_path,'r') as file:
            params=yaml.safe_load(file)
            param_grid=params['model_building']['param_grid']
            logger.debug(f"param_grid,{param_grid}")
            return param_grid    
    except Exception as e:
        logger.error(e) 


def model_building(param_grid):
    xgb=XGBClassifier()
    grid = GridSearchCV(xgb,param_grid=param_grid,cv=10,scoring='accuracy')
    grid.fit(X,y)
    return grid.best_params_,grid.best_estimator_,grid.best_score_

def save_model(best_estimator):
    os.makedirs(os.path.join('data', 'model'), exist_ok=True)      
    with open(os.path.join('data','model','model.pkl'),'wb') as f:
        pickle.dump(best_estimator,f)

def save_model_info(run_id,model_path,file_path):
    model_info={'run_id':run_id,'model_uri':model_path}
    with open(file_path,'w') as file:
        json.dump(model_info,file,indent=4)


    
def main():
    param_grid=load_params(os.path.join('params.yaml'))
    best_params,best_estimator,best_score=model_building(param_grid)
    save_model(best_estimator)
    print(best_params,best_estimator,best_score)
    mlflow.set_experiment("dvc-pipeline_new")
    with mlflow.start_run() as run:
        mlflow.log_metric("accuracy_score", best_score)
        mlflow.log_params(best_params)

        mlflow.sklearn.log_model(best_estimator, artifact_path="model")

        # IMPORTANT: save model URI, not just "model"
        save_model_info(
            run.info.run_id,
            model_uri, 
            'reports/experiment.json'
        )

        mlflow.log_artifact('reports/experiment.json')



        
main()
    




    


