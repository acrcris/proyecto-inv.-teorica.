#!/usr/bin/env python3
"""
Script rápido para probar entrenamiento con diferentes valores de C.
Compara: C por debajo, C crítico, y C por encima del valor crítico.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kuramoto.modelo import KBlock

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Valores de C a probar (centrados en C_crítico = 0.1769)
C_VALUES = {
    'bajo': 0.10,       # ~0.6x del crítico
    'critico': 0.1769,  # Valor crítico encontrado
    'alto': 0.30        # ~1.7x del crítico
}

# Parámetros del modelo (idénticos al notebook kuramoto_pruebas_basico.ipynb)
CH = 3
N = 3
T_STEPS = 30
KSIZE = 3
INIT_OMG = 0.1
IMG_SIZE = 64
GAMMA = 0.7
DEL_T = 0.9

# Parámetros de entrenamiento
EPOCHS = 10
BATCH_SIZE = 128
LR = 0.001
N_TRAIN_SAMPLES = 10000  # Solo 10k imágenes para rapidez
N_TEST_SAMPLES = 2000    # 2k para test

SEED = 1  # Consistente con otros scripts del proyecto
RESULTS_DIR = Path("./resultados_prueba_c_critico")
RESULTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# MODELO
# =============================================================================

class SimpleKuramotoClassifier(nn.Module):
    def __init__(self, ch, n, c_value):
        super().__init__()
        self.c_value = c_value  # Guardar el valor de C
        self.kblock = KBlock(
            ch=ch,
            n=n,
            T=T_STEPS,
            ksize=KSIZE,
            init_omg=INIT_OMG,
            connectivity='conv',
            c_norm=None,
            use_omega_c=False
        )
        # Readout simple
        self.readout = nn.Sequential(
            nn.Conv2d(ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10)
        )
    
    def forward(self, x):
        # Aplicar C como acoplamiento multiplicando la entrada
        # Similar al notebook: c = img_channels.unsqueeze(0) * c_value
        c = x * self.c_value
        # Usar gamma y del_t fijos del notebook
        x, _, _ = self.kblock(x, c, T=T_STEPS, gamma=GAMMA, del_t=DEL_T, return_xs=True, return_es=True)
        return self.readout(x)

# =============================================================================
# FUNCIONES DE ENTRENAMIENTO
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, pred = outputs.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(loader), 100. * correct / total

def test_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Testing", leave=False):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            _, pred = outputs.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    
    return total_loss / len(loader), 100. * correct / total

def train_model(c_value, c_name, device):
    print(f"\n{'='*70}")
    print(f"Entrenando con C = {c_value} ({c_name})")
    print(f"{'='*70}\n")
    
    # Dataset (con normalización idéntica al notebook)
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.Grayscale(num_output_channels=CH),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Media y desviación estándar de MNIST
    ])
    
    train_dataset = datasets.MNIST(
        root='../data',
        train=True,
        transform=transform,
        download=True
    )
    
    test_dataset = datasets.MNIST(
        root='../data',
        train=False,
        transform=transform
    )
    
    # Subset para rapidez
    train_indices = np.random.RandomState(SEED).choice(
        len(train_dataset), N_TRAIN_SAMPLES, replace=False
    )
    test_indices = np.random.RandomState(SEED).choice(
        len(test_dataset), N_TEST_SAMPLES, replace=False
    )
    
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Modelo
    model = SimpleKuramotoClassifier(CH, N, c_value).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Entrenamiento
    history = {
        'c_value': c_value,
        'c_name': c_name,
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_acc = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), RESULTS_DIR / f"best_model_c_{c_name}.pt")
    
    # Guardar historial
    with open(RESULTS_DIR / f"history_c_{c_name}.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Mejor accuracy: {best_acc:.2f}%")
    return history

# =============================================================================
# MAIN
# =============================================================================

def main():
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = 'CUDA'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = 'MPS'
    else:
        device = torch.device('cpu')
        device_name = 'CPU'
    
    print(f"Usando dispositivo: {device_name}")
    
    # Seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Entrenar para cada C
    all_results = {}
    
    for c_name, c_value in C_VALUES.items():
        history = train_model(c_value, c_name, device)
        all_results[c_name] = history
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    
    for c_name, history in all_results.items():
        final_test_acc = history['test_acc'][-1]
        best_test_acc = max(history['test_acc'])
        c_value = history['c_value']
        print(f"\nC = {c_value} ({c_name}):")
        print(f"  - Accuracy final: {final_test_acc:.2f}%")
        print(f"  - Mejor accuracy: {best_test_acc:.2f}%")
    
    # Guardar resumen
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'n_train': N_TRAIN_SAMPLES,
            'n_test': N_TEST_SAMPLES,
            'ch': CH,
            'n': N,
            't_steps': T_STEPS
        },
        'results': all_results
    }
    
    with open(RESULTS_DIR / "resumen_completo.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📁 Resultados guardados en: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
