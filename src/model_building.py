import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.metrics import accuracy_score
from app_logging import *
import os
import yaml
import pickle
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
    return None

def main():
    param_grid=load_params(r'C:\Users\milan\Desktop\stress_project\params.yaml')
    best_params,best_estimator,best_score=model_building(param_grid)
    save_model(best_estimator)
    print(best_params,best_estimator,best_score)

        
main()
    




    


