import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from setuptools.sandbox import save_path
from prediction_model import config

# load dataset
def load_dataset(file_name):
    """This function loads the dataset for training and prediction purposes"""
    filepath = os.path.join(config.DATA_PATH,file_name)
    _data = pd.read_csv(filepath)
    return _data

# Serialization
def save_pipeline(pipeline_to_save):
    """This function saves the pipeline model to a specified path."""
    save_path = os.path.join(config.MODEL_PATH,config.MODEL_NAME)
    joblib.dump(pipeline_to_save,save_path)
    print(f"Model has been saved under the name {config.MODEL_NAME} at {config.MODEL_PATH}")

# Deserialization
def load_pipeline(pipeline_to_load):
    """This function loads a saved pipeline model from a specified path"""
    load_path = os.path.join(config.MODEL_PATH,config.MODEL_NAME)
    try:
        model_loaded = joblib.load(load_path)
        print(f"Model has been loaded at {load_path}")
        return model_loaded
    except FileNotFoundError:
        print(f"This model file {config.MODEL_NAME} was not found at {config.MODEL_PATH}")
        raise
    except Exception as e:
        print(f"An error occured while loading the model: {e}")
        raise


# data splitting
def split_dataset(data,test_size = 0.2,random_state=1):
    """This function splits the data into training and test sets"""
    train_data,test_data = train_test_split(data,test_size=test_size,random_state=random_state)
    return train_data,test_data

   