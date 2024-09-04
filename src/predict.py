from prediction_model import config
from prediction_model.processing.data_handling import load_dataset,load_pipeline
import pandas as pd 
import numpy as np

regressor_pipeline = load_pipeline(config.MODEL_NAME)

def generate_predictions(data_inputs):
    data = pd.DataFrame(data_inputs)
    pred = regressor_pipeline.predict(data[config.FEATURES])
    #Ensure predictions are finite numbers
    output = np.clip(pred, a_min=0, a_max=None)
    results = {'output':output.tolist()}
    return results

if __name__ =="__main__":
    generate_predictions()