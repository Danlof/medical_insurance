from fastapi import FastAPI,HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from typing import List
import prediction_model.config as config
from prediction_model.processing.data_handling import load_pipeline

app = FastAPI()

origins = [
    '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Initialize the Prometheus Instrumentator
instrumentator = Instrumentator()

# Instrument the FastAPI app
instrumentator.instrument(app).expose(app)

# load trained model 
@app.on_event("startup")
def load_model():
    global model
    model = load_pipeline(config.MODEL_NAME)
    print(f"Model has been loaded {config.MODEL_PATH}")

# Defining input data using pydantic 
class PremiumPredData(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

class InsuranceDataList(BaseModel):
    data: List[PremiumPredData]

@app.get("/")
def index():
    return {'Message':'Welcome to Insurance Premium Prediction App'}

# prediction endpoint 
@app.post("/predict")
def predict(data: InsuranceDataList):
    try:
        # converting input data to a dataframe
        data_dict= [item.model_dump() for item in data.data]
        input_data = pd.DataFrame(data_dict)

        # predictions 
        predictions = model.predict(input_data[config.FEATURES])
        return {"predictions":predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

# health check endpoints 
@app.get("/health")
def health_check():
    return {'status':"healthy"}

if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0",port=8000)