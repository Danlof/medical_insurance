# contains configuration settings and hyperparameters that are essential
# for ML pipeline 

import pathlib
import os


# General settings 
PROJECT_NAME = "Insurance Premium Prediction"
## finds the path of the prediction_model
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT,"datasets")
MODEL_PATH = os.path.join(PROJECT_ROOT,"trained_models")

# Data setting
DATA_PATH = os.path.join(DATA_DIR,"Medical_insurance.csv")
TARGET_COLUMN = "charges"

MODEL_NAME = "medical_model_v1.pkl"


FEATURES = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
NUM_FEATURES = ['age', 'bmi', 'children', 'smoker']
CAT_FEATURES = ['sex','smoker','region']
FEATURES_TO_ENCODE = ['sex','smoker','region']


ENVIRONMENT = "development"