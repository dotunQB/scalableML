import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel, Field
from ml.data import process_data
from ml.model import inference, compute_model_metrics

model_input = load(open('./model_out.pkl','rb'))
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
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")
    salary: str

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "age": 30,
                "workclass": "Private",
                "fnlgt": 123455,
                "education": "Bachelors",
                "education-num": 10,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 10,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "Cuba",
                "salary": "<=50K"
            }
        }


@app.get("/")
async def root():
    return {"message": "Welcome to Deploying ML Model to Cloud Application Exam with Census Data"}


@app.post("/predict_salary")
def predict_salary(body: CensusValue):
    print(list(body.dict().keys()))
    print(list(body.dict().values()))
    values = pd.DataFrame(columns=list(body.dict().keys()))
    values = values.append(pd.Series(list(body.dict().values()) , index=list(body.dict().keys())),ignore_index=True)
    values.rename(columns={'marital_status': 'marital-status', 'native_country': 'native-country'}, inplace=True)
    X_test, y_test, encoder, lb = process_data(
        values, categorical_features=cat_features, label="salary", training=False,encoder=encoder_out, lb=lb_out
    )
    preds = inference(model_input, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(preds)
    return {
        "prediction is ": str(preds),
        "precision is ": precision,
        "recall is ": recall,
        "fbeta is ": fbeta
    }