import pandas as pd
from fastapi import FastAPI
import pydantic
from pydantic import BaseModel, Field
import joblib
from typing import Annotated, Literal
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "churn_model.pkl"

model_bundle = joblib.load(model_path)
model = model_bundle['pipeline']
feature_cols = model_bundle['feature_cols']

class customer(BaseModel):
    gender : Literal['Male','Female']
    Partner : Literal['Yes','No']
    Dependents : Literal['Yes','No']
    PhoneService : Literal['Yes','No']
    MultipleLines : Literal['No phone service', 'No', 'Yes']
    InternetService : Literal['DSL', 'Fiber optic', 'No']
    OnlineSecurity :Literal['No', 'Yes', 'No internet service']
    OnlineBackup : Literal['Yes', 'No', 'No internet service']
    DeviceProtection : Literal['No', 'Yes', 'No internet service']
    TechSupport : Literal['No', 'Yes', 'No internet service']
    StreamingTV : Literal['No', 'Yes', 'No internet service']
    StreamingMovies : Literal['No', 'Yes', 'No internet service']
    Contract : Literal['Month-to-month', 'One year', 'Two year']
    PaperlessBilling : Literal['Yes', 'No']
    PaymentMethod : Literal['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)']
    SeniorCitizen : Annotated[int, Field(...,ge=0,le=1)]
    tenure : Annotated[float,Field(...,examples=[34])]
    MonthlyCharges : Annotated[float,Field(...,examples=[29.5])]
    TotalCharges : Annotated[float,Field(...,examples=[29.5])]

app = FastAPI()

@app.get('/')
def hello():
    return {'hello brother'}

class PredictionResponse(BaseModel):
    churn : int

@app.post('/predict')
def predict(data:customer):
    input_df = pd.DataFrame([data.model_dump()
    ])
    input_df = input_df[feature_cols]

    prediction = model.predict(input_df)[0]
    return PredictionResponse(churn=int(prediction))
