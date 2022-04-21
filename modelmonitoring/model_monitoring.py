import pandas as pd
import numpy as np
import os
import pickle
import mlflow
from mlflow.tracking import MlflowClient

from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab, CatTargetDriftTab, ClassificationPerformanceTab
from evidently.tabs import NumTargetDriftTab, RegressionPerformanceTab
from sklearn.model_selection import train_test_split

relative_model_dev_path = '../modeldevelopment/'

# import sys
# sys.path.insert(1, relative_model_dev_path)
from modeldevelopment.auto_feat import MakeDataSet
md = MakeDataSet()

def find_registered_model(name, uri):
    model_name = name
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_ids=1)

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
            frac=1, random_state=42).reset_index(drop=True)
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
        print(e)
        
#     Get data
    data_aif360_reference = None
    data_aif360_production = None
    df_train = None
    df_test  = None
    y_train  = None
    y_test   = None
    try:
        df_train,df_test, y_train, y_test = train_test_split(load_dataframe().drop('target',axis=1), load_dataframe()['target'], test_size=0.3, random_state=42)
#         data_aif360_reference = md.decode_dataset(data_frame=df_train)
#         data_aif360_production = md.decode_dataset(data_frame=df_test)
        pass
    except Exception as e:
        print(e)
        
#         Model_Monitoring
    try:
        ref_prediction  = loaded_model.predict(df_train)
        prod_prediction = loaded_model.predict(df_test)
        
        reference  = pd.concat([df_train,y_train],axis=1)
        production = pd.concat([df_test ,y_test ],axis=1)
        
        reference['prediction'] = ref_prediction
        production['prediction'] = prod_prediction
        
        
        column_mapping = {}

        column_mapping['target'] = 'target'
        column_mapping['prediction'] = 'prediction'
        column_mapping['categorical_features'] = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        
        classification_perfomance_dashboard = Dashboard(tabs=[ClassificationPerformanceTab, CatTargetDriftTab,DataDriftTab])
        classification_perfomance_dashboard.calculate(reference,production,column_mapping)
        classification_perfomance_dashboard.save('./monitoring_reports/Heart_Model_Monitoring.html')
  
        pass
    except Exception as e:
        print(e)