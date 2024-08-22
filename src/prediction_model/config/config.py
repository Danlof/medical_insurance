# contains configuration settings and hyperparameters that are essential
# for ML pipeline 

import pathlib
import os 
import prediction_model

# General settings 
PROJECT_NAME = "Insurance Premium Prediction"

# finds the path of the prediction_model
PROJECT_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent
