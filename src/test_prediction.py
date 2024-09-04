import pytest 
from prediction_model import config
from prediction_model.processing.data_handling import load_dataset,split_dataset
from predict import generate_predictions

# 3 tests for the insurance premium prediction model
# output from predict script is not null
# output from predict script is of float type
# output is within a realistic range for insurance premiums

@pytest.fixture
def single_prediction():
    data = load_dataset(config.DATA_PATH)
    train_data,test_data = split_dataset(data)
    # select a single row for testing
    single_row = test_data[:1]
    # generate prediction
    results = generate_predictions(single_row)
    return results

# Test 1: Ensuring prediction is not none
def single_pred_not_none(single_prediction):
    assert single_prediction is not None

# Test 2: Ensure the prediction is of type float
def test_single_pred_float_type(single_prediction):
    assert isinstance(single_prediction.get("output")[0],float)

# Test 3: Ensure premiums are within a realistic range 
def test_single_pred_within_range(single_prediction):
    pred_value = single_prediction.get("output")[0]
    assert 500 <= pred_value <= 99000
    
