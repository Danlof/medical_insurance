from sklearn.pipeline import Pipeline
from prediction_model import config
import prediction_model.processing.preprocessing as pp
from sklearn.ensemble import RandomForestRegressor

print("pipeline process has started")
regressor_pipeline = Pipeline([
    ("LabelEncoder", pp.CustomEncoder(variables=config.FEATURES_TO_ENCODE)),
    ("LogTransformation", pp.LogTransforms(variables=config.NUM_FEATURES)),
    ("Standardization", pp.Standardization(variables=config.NUM_FEATURES)),
    ("RandomForestRegressor", RandomForestRegressor(random_state=1))
    
])
print("pipeline process has ended ")