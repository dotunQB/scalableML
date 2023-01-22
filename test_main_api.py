import json
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to Deploying ML Model to Cloud Application Exam with Census Data"}

def test_post_data_sucess():
    data ={"workclass":'Private', "education":'Bachelors', "marital_status":'Never-married',
           "occupation":'Adm-clerical', "relationship":'Not-in-family',"race":'White',
           "sex":'Male', "native_country":'Cuba', "salary":'<=50K'}
    r = client.post("/predict_salary", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json()

def test_post_data_fail():
    data ={"workclass": 0, "education":'Bachelors', "marital_status":'Never-married',
           "occupation":'Adm-clerical', "relationship":'Not-in-family',"race":'White',
           "sex":'Male', "native_country":'Cuba'}
    r = client.post("/predict_salary", data=json.dumps(data))
    assert r.status_code == 422
    assert r.json()['detail'][0]['type'] == 'value_error.missing'