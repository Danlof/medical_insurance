## Medical insurance Price prediction

- The aim of this project is to develop a model that can predict medical charges faced by individuals using features such as `Age, Sex, BMI, number of Children, Smoker, and Region`.

### Problem statement
1. What are the most important factors that affect medical expenses/
2. How well can machine learning models predict medical expenses?
3. How can machine learning models be used to imporove the efficiency and profitability of health insurance companies?

## Project structure 
```
/project_root/
├── datasets/
│   └── Medical_insurance.csv
├── models/
│   └── trained_models/
├── prediction_model/
│   ├── __init__.py
│   ├── config.py
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── data_handling.py
│   │   └── preprocessing.py
│   ├── pipeline.py
│   └── ...
├── setup.py
├── training_pipeline.py
├── test_prediction.py
├── Dockerfile
└── main.py
```

# pytest
- It is important to check if our model is going to work by check the following :
    - output from predict script is not null
    - output from predict script is of float type
    - output is within a realistic range for insurance premiums
- To run it use `pytest test_prediction.py`

### Fastapi backend
- Using `BaseModel` to ensure consistency of data that will be required by the model for prediction
- `@app.on_event("startup")` is used to start tasks that are required by the application before prediction begins, such loading the pretrained model,loading data , setting up logging and monitoring services 
- Health check helps to check if the server is working properly

RUN it by `uvicorn main:app --reload`