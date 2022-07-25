import os
import torch
from torch import nn

import mlflow
import mlflow.pytorch

from model import build_model, device
from dataset import build_dataloaders


tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
mlflow.set_tracking_uri(tracking_uri)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

    test_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    test_acc = correct * 100.0
    print(f"Test Error: \n Accuracy: {(test_acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_acc, test_loss


def run_training(learning_rate=1e-2, batch_size=64, epochs=3):
    train_dataloader, test_dataloader = build_dataloaders(batch_size)
    model, signature = build_model()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    best_acc = 0.0

    with mlflow.start_run():
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            test_acc, test_loss = test(test_dataloader, model, loss_fn)

            mlflow.log_metric("test_acc", test_acc, step=t)
            mlflow.log_metric("test_loss", test_loss, step=t)

            if test_acc > best_acc:
                best_acc = test_acc
                mlflow.log_metric("best_acc", best_acc, step=t)
                mlflow.pytorch.log_model(model, "model", signature=signature)

    print("Done!")


if __name__ == "__main__":
    run_training()
