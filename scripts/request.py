import sys
import torch
import requests


def request(url):
    print("Endpoint URL:\n", url, "\n")

    X = torch.randn(1, 28, 28)
    inference_request = {"inputs": X.tolist()}
    response = requests.post(url, json=inference_request)
    
    print("Response:")
    try:
        print(response.json())
    except:
        print(response)


if __name__ == "__main__":

    # env options: (local, kubernetes)
    env = sys.argv[1] if len(sys.argv) >= 2 else "local"

    if env == "local":
        endpoint = "http://localhost:8080/invocations"

    elif env == "kubernetes":
        endpoint = "http://localhost:8080/seldon/default/mlflow-model/invocations"

    request(endpoint)
