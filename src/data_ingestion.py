import pandas as pd
import numpy as np
from app_logging import *
import yaml
from sklearn.model_selection import train_test_split
import os

def load_params(file_path):
    try:
        with open(file_path,'r') as file:
            params=yaml.safe_load(file)
            test_size=params['data_ingestion']['test_size']
            logger.debug(f"test_size,{test_size}")
    except Exception as e:
        logger.error(e)    
def load_data(file_path):
    try:
        df=pd.read_csv(file_path)
        logger.debug("dataframe is created")
        return df 
    
    except Exception as e:
        logger.error(e)






def main():
    test_size=load_params(r'C:\Users\milan\Desktop\stress_project\params.yaml')
    df=load_data(r'C:\Users\milan\Desktop\stress_project\Teen_Mental_Health_Dataset.csv')
    train,test=train_test_split(df,test_size=test_size,random_state=42)
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    df.to_csv(os.path.join("data","raw",'test.csv'),index=False)
    df.to_csv(os.path.join("data","raw",'train.csv'),index=False)
    df.to_csv(os.path.join("data","raw",'model.csv'),index=False)

main()    