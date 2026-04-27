import pandas as pd
import numpy as np
from app_logging import *
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder
import os
X=pd.read_csv(os.path.join("data","raw","model.csv")).iloc[:,:-1]
y=pd.read_csv(os.path.join("data","raw","model.csv")).iloc[:,-1]

def load_params(file_path):
    try:
        with open(file_path,'r') as file:
            params=yaml.safe_load(file)
            encoder=params['feature_engineering']['encoder']
            logger.debug(f"encoder:{encoder}")
            return encoder
    except Exception as e:
        logger.error(e) 


def get_encoder(encoder_name):
    if encoder_name == "OneHotEncoder":
        return OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    elif encoder_name == "LabelEncoder":
        return LabelEncoder()
    elif encoder_name == "OrdinalEncoder":
        return OrdinalEncoder()
    else:
        raise ValueError("Unsupported encoder")


def encoding(X,encoder):
    oe = OneHotEncoder(sparse_output=False, handle_unknown='ignore',drop=None)
    columns=['gender','platform_usage','social_interaction_level']
    X_trans=oe.fit_transform(X[columns])

    encoded_df = pd.DataFrame(
        X_trans,
        columns=oe.get_feature_names_out(columns),
        index=X.index
    )
    X = X.drop(columns=columns)
    X_trans = pd.concat([X, encoded_df], axis=1)
    return X_trans

def main():
    encoder_name=load_params(os.path.join('params.yaml'))
    encoder=get_encoder(encoder_name)
    X_trans = encoding(X, encoder)
    os.makedirs(os.path.join("data", "feature_engg"), exist_ok=True)
    X_trans.to_csv(os.path.join("data","feature_engg","X.csv"),index=False)  
    y.to_csv(os.path.join("data","feature_engg","y.csv"),index=False)    
main()


