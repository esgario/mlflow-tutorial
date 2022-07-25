import torch
import requests

X = torch.randn(1, 28, 28)

inference_request = {
    "inputs": X.tolist(),
}

endpoint = "http://localhost:6000/invocations"
response = requests.post(endpoint, json=inference_request)

print(response.json())