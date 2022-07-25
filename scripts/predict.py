"""Fetching and running an MLFlow model from the Model Registry."""
import torch
import mlflow.pytorch
from scripts.model import device


model_name = "pytorch_simplenn_mnist"
model_version = 1

model = mlflow.pytorch.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

X = torch.randn(1, 28, 28).to(device)

print("Testing model with random data.")
print("Output:", model(X))
