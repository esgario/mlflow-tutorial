import torch
from torch import nn
from mlflow.models.signature import infer_signature


device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the model.
class SimpleNN(nn.Module):
    def __init__(self, input_shape):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_shape[0] * input_shape[1], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits


def build_model(input_shape=(28, 28)):
    # get model
    model = SimpleNN(input_shape).to(device)
    print(model, "\n")

    # get model signature
    x = torch.randn(2, *input_shape)
    y = model(x.to(device)).cpu()
    signature = infer_signature(x.numpy(), y.detach().numpy())
    print(signature)

    return model, signature
