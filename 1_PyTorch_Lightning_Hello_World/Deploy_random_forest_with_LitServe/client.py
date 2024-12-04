import requests

# features
data = {"input": [-1.66853167, -1.29901346,  0.2746472 , -0.60362044]}
response = requests.post("http://127.0.0.1:8000/predict", json=data)
print(response.json())
