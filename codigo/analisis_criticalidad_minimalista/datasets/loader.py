"""
Carga y preprocesamiento de datos para análisis de criticalidad minimalista.
"""
import torch
from torchvision import datasets, transforms

class MNISTLoader:
    """Carga y preprocesa MNIST para experimentos minimalistas."""
    def __init__(self, root='./data', batch_size=64):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=self.transform)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1000, shuffle=False)

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader
