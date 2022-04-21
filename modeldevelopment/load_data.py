import logging

import pandas as pd

import modeldevelopment.settings as settings
import sweetviz as sv

logger = logging.getLogger(__name__)

target = settings.Y_COLUMN[0]


def load_dataframe():
    """Load"""
    try:
        data_frame = pd.read_csv(settings.DATASET_PATH)
        shuffled_df = data_frame.sample(
            frac=1, random_state=107).reset_index(drop=True)
        
        my_report = sv.compare_intra(shuffled_df,shuffled_df[settings.Y_COLUMN[0]] == 1,["Heart Disease", "No disease"])
        my_report.show_html(filepath='./reports/eda/eda_report.html',open_browser=False)
        
        return shuffled_df
    except Exception as e:
        logger.error(e)


def save_dataframe(prediction_labels, name):
    """
    prediction_labels:
    name:
    """
    try:
        df = pd.read_csv("./data/model results datasets/Test_data.csv")
        df[str(name)] = prediction_labels
        df.to_csv(f"./data/model results datasets/Test_data.csv", index=False)
    except Exception as e:
        logger.error(e)
        print(e)
