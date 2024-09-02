import numpy as np
import pandas as pd
from prediction_model import config
from sklearn.model_selection import RandomizedSearchCV,KFold
from prediction_model.processing.data_handling import load_dataset,save_pipeline,split_dataset
import prediction_model.processing.preprocessing as pp
import prediction_model.pipeline as pipe


def perform_training():
    data = load_dataset(config.DATA_PATH)
    train_data,test_data = split_dataset(data)
    # separate features and targets for training
    X_train = train_data[config.FEATURES]
    y_train = train_data[config.TARGET_COLUMN]

    print(X_train.head)

    # define parameters for the randomized search cv
    param_grid = {
    'RandomForestRegressor__n_estimators': np.arange(50, 500, 50),
    'RandomForestRegressor__max_depth': np.arange(3, 20, 2),
    'RandomForestRegressor__min_samples_split': np.arange(2, 10),
    'RandomForestRegressor__min_samples_leaf': np.arange(1, 10),
    'RandomForestRegressor__max_features': [None, 'sqrt', 'log2'],
    'RandomForestRegressor__bootstrap': [True, False]
    }


    # for cross-validation we use the kfold method
    kf = KFold(n_splits=5, shuffle = True, random_state = 4)

    random_search = RandomizedSearchCV(
        estimator= pipe.regressor_pipeline,
        param_distributions=param_grid,
        n_iter= 50,
        scoring= "neg_root_mean_squared_error",
        n_jobs=-1,
        cv = kf,
        random_state = 1
    )
    
    print("Random search started")
    # fit the model with randomized search
    random_search.fit(X_train,y_train)

    # save best model 
    best_model = random_search.best_estimator_
    save_pipeline(best_model)

    print("Training was successful .Best parameters found:")
    print(random_search.best_params_)

if __name__ == "__main__":
    perform_training()
