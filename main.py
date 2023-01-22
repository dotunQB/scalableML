import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel, Field
from ml.data import process_data
from ml.model import inference, compute_model_metrics

model_input = load(open('./model_slice.pkl','rb'))
encoder_out = load(open('./encoder_out.pkl','rb'))
lb_out = load(open('./lb_out.pkl','rb'))
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",

]

app = FastAPI()
class CensusValue(BaseModel):
    workclass: str
    education: str
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str = Field(alias="native-country")
    salary: str

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "workclass": "Private",
                "education": "Bachelors",
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "native-country": "Cuba",
                "salary": "<=50K"
            }
        }


@app.get("/")
async def root():
    return {"message": "Welcome to Deploying ML Model to Cloud Application Exam with Census Data"}


@app.post("/predict_salary")
def predict_salary(body: CensusValue):
    values = pd.DataFrame(columns=list(body.dict().keys()))
    values = values.append(pd.Series(list(body.dict().values()) , index=list(body.dict().keys())),ignore_index=True)
    values.rename(columns={'marital_status': 'marital-status', 'native_country': 'native-country'}, inplace=True)
    X_test, y_test, encoder, lb = process_data(
        values, categorical_features=cat_features, label="salary", training=False,encoder=encoder_out, lb=lb_out
    )
    preds = inference(model_input, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    return {
        "precision is ": precision,
        "recall is ": recall,
        "fbeta is ": fbeta
    }