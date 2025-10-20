"""
Carga y preprocesamiento de datos para análisis de criticalidad minimalista.
"""
import torch
from torchvision import datasets, transforms

class MNISTLoader:
    """Carga y preprocesa MNIST para experimentos minimalistas."""
    def __init__(self, root='./data', batch_size=64, img_size=64):
        self.img_size = img_size
        
        transform_list = [transforms.ToTensor()]
        
        # Redimensionar si es necesario
        if img_size != 28:
            transform_list.insert(0, transforms.Resize((img_size, img_size)))
        
        # Normalización estándar MNIST
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        
        self.transform = transforms.Compose(transform_list)
        
        self.train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=self.transform)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader
    
    def get_mnist(self, batch_size=1, train_split=False):
        """Método compatible con DatasetFactory."""
        if train_split:
            train_loader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=True
            )
            test_loader = torch.utils.data.DataLoader(
                self.test_dataset, batch_size=batch_size, shuffle=False
            )
            return train_loader, test_loader
        else:
            test_loader = torch.utils.data.DataLoader(
                self.test_dataset, batch_size=batch_size, shuffle=False
            )
            return None, test_loader
