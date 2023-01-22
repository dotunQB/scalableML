"""
Test ml model on cloud deployed link through render
"""
import requests
response = requests.get('https://render-scalableml.onrender.com/')
print(response.status_code)
print(response.json())