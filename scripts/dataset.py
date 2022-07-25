from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def build_dataloaders(batch_size=64):
    # Create datasets.
    train_data = datasets.MNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )

    # Create data loaders.
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    X, y = next(iter(test_dataloader))
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}\n")

    return train_dataloader, test_dataloader