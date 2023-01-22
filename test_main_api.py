"""
Author: Dotun Opasina
"""
import json
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to Deploying ML Model to Cloud Application Exam with Census Data"}

def test_post_data_sucess_greater_50K():
    data ={"age": 52,
      "workclass": "Self-emp-not-inc",
      "fnlgt": 209642,
      "education": "HS-grad",
      "education-num": 9,
      "marital-status": "Married-civ-spouse",
      "occupation": "Exec-managerial",
      "relationship": "Wife",
      "race": "White",
      "sex": "Female",
      "capital-gain": 15024,
      "capital-loss": 0,
      "hours-per-week": 45,
      "native-country": "United-States",
      "salary": ">50K"
    }
    r = client.post("/predict_salary", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == {'prediction is ': '[1]', 'precision is ': 0.0, 'recall is ': 1.0, 'fbeta is ': 0.0}

def test_post_data_sucess_less_50k():
    data ={"age": 30,
      "workclass": "Private",
      "fnlgt": 123456,
      "education": "Bachelors",
      "education-num": 10,
      "marital-status": "Never-married",
      "occupation": "Adm-clerical",
      "relationship": "Not-in-family",
      "race": "White",
      "sex": "Male",
      "capital-gain": 0,
      "capital-loss": 0,
      "hours-per-week": 40,
      "native-country": "Cuba",
      "salary": "<=50K"
    }
    r = client.post("/predict_salary", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == {'prediction is ': '[0]', 'precision is ': 1.0, 'recall is ': 1.0, 'fbeta is ': 1.0}

def test_post_data_fail():
    data ={"workclass": 0, "education":'Bachelors', "marital_status":'Never-married',
           "occupation":'Adm-clerical', "relationship":'Not-in-family',"race":'White',
           "sex":'Male', "native_country":'Cuba'}
    r = client.post("/predict_salary", data=json.dumps(data))
    assert r.status_code == 422
    assert r.json()['detail'][0]['type'] == 'value_error.missing'