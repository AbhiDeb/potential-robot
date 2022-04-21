import traceback
import pandas as pd
import numpy as np
import os
import pickle
import mlflow
from mlflow.tracking import MlflowClient
from shapash.explainer.smart_explainer import SmartExplainer

relative_model_dev_path = '../modeldevelopment/'

# import sys
# sys.path.insert(1, relative_model_dev_path)
from modeldevelopment.auto_feat import MakeDataSet
md = MakeDataSet()

def find_registered_model(name, uri):
    model_name = name
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_ids=['1'])

    client = MlflowClient()
    logged_model = None
    for reg_mod in client.list_registered_models():
        for versions in reg_mod.latest_versions:
            if versions.current_stage == 'Production':
                logged_model = versions.source
                return logged_model

def load_dataframe():
    """Load"""
    try:
        data_frame = pd.read_csv('../datasource/heart.csv')
        shuffled_df = data_frame.sample(
            frac=1, random_state=107).reset_index(drop=True)
        return shuffled_df
    except Exception as e:
        print(e)

if __name__ == '__main__':
    #     Load model
    loaded_model = None
    try:
        model_path = find_registered_model(name = "Bias Mitigation Heart Disease", uri = f"sqlite:///{relative_model_dev_path}/Heart_Disease_MLFlow.db")
        with open(f'{relative_model_dev_path}/{model_path[2:]}/model.pkl', "rb") as f:
            loaded_model = pickle.load(f)
    except Exception as e:
        print(traceback.format_exc())
        
#     Get data
    data_aif360 = None
    try:
        data_aif360 = md.decode_dataset(data_frame=load_dataframe())
        pass
    except Exception as e:
        print(traceback.format_exc())
        
#         Shapash
    try:
        response_dict = {0: 'No Disease', 1:'Heart Problem'}
        dict1 = {}
        for col in data_aif360.feature_names:
            dict1[col] = col
        xpl = SmartExplainer(features_dict=dict1,label_dict=response_dict)
        df = data_aif360.convert_to_dataframe()[0]
        xpl.compile(x=df.drop('target',axis=1),model=loaded_model)
        app = xpl.run_app(title_story='Heart Disease Model - XAI')
        pass
    except Exception as e:
        print(traceback.format_exc())