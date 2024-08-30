# contains configuration settings and hyperparameters that are essential
# for ML pipeline 

import pathlib
import os
from src import prediction_model

# General settings 
PROJECT_NAME = "Insurance Premium Prediction"
## finds the path of the prediction_model
PROJECT_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent
DATA_DIR = os.path.join(PROJECT_ROOT,"datasets")
MODEL_DIR = os.path.join(PROJECT_ROOT,"models")

# Data setting
DATA_PATH = os.path.join(DATA_DIR,"Medical_insurance.csv")
TARGET_COLUMN = "charges"

MODEL_NAME = "medical_model_v1.pkl"
MODEL_PATH = os.path.join(MODEL_DIR,"trained_models")

FEATURES = ['Age', 'Sex', 'BMI', 'number of Children', 'Smoker', 'Region']
NUM_FEATURES = ['Age', 'BMI', 'number of Children', 'Smoker']
CAT_FEATURES = ['Sex','Smoker','Region']
FEATURES_TO_ENCODE = ['Sex','Smoker','Region']

# Model settings
MODEL_TYPE = "XGBRegressor"
HYPERPARAMETERS = {
    "learning_rate": 0.1,
    "subsample": 0.8,
    "gamma": 0.5,
    "colsample_bytree": 0.8,
    "n_estimators": 100
}

# Training_settings
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.2

ENVIRONMENT = "development"