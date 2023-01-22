"""
Test ml model on cloud deployed link through render
"""
from requests.auth import HTTPBasicAuth
import requests
import json

url = 'https://api_url'
headers = {'Accept': 'application/json',
           "Authorization": "Bearer rnd_iqq5CkM7BgAOqvR9jucjYUivERzl" }
response = requests.get('https://render-scalableml.onrender.com/')
print(response.status_code)
print(response.json())

data ={"age": 30,
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
response = requests.post('https://render-scalableml.onrender.com/predict_salary',
                         headers=headers, data=json.dumps(data))
print(response.status_code)
print(response.json())