#!/usr/bin/env python3
"""
Script para configurar MNIST en AKOrN
Descarga y prepara el dataset MNIST para usar con train_classification.py
"""

import os
import torch
from torchvision import datasets, transforms
from pathlib import Path

def setup_mnist():
    """Descarga y prepara MNIST para AKOrN"""
    
    print("=" * 80)
    print("CONFIGURANDO MNIST PARA AKORN")
    print("=" * 80)
    
    # Crear estructura de directorios
    data_dir = Path("./data/MNIST")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[1/3] Creando estructura de directorios...")
    print(f"      Directorio: {data_dir.absolute()}")
    
    # Descargar MNIST
    print(f"\n[2/3] Descargando dataset MNIST...")
    print("      → Dataset de entrenamiento (60,000 imágenes)")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    try:
        trainset = datasets.MNIST(
            str(data_dir),
            train=True,
            download=True,
            transform=transform
        )
        print(f"      ✓ Entrenamiento: {len(trainset)} imágenes")
    except Exception as e:
        print(f"      ✗ Error descargando entrenamiento: {e}")
        return False
    
    print("      → Dataset de validación (10,000 imágenes)")
    
    try:
        testset = datasets.MNIST(
            str(data_dir),
            train=False,
            download=True,
            transform=transform
        )
        print(f"      ✓ Validación: {len(testset)} imágenes")
    except Exception as e:
        print(f"      ✗ Error descargando validación: {e}")
        return False
    
    # Verificar descarga
    print(f"\n[3/3] Verificando integridad...")
    
    # Cargar un batch de prueba
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=False
    )
    
    try:
        batch_x, batch_y = next(iter(trainloader))
        print(f"      ✓ Batch de prueba cargado correctamente")
        print(f"      - Shape entrada: {batch_x.shape}")
        print(f"      - Shape etiquetas: {batch_y.shape}")
        print(f"      - Rango valores: [{batch_x.min():.3f}, {batch_x.max():.3f}]")
        print(f"      - Etiquetas únicas: {sorted(batch_y.unique().tolist())}")
    except Exception as e:
        print(f"      ✗ Error validando dataset: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✓ MNIST CONFIGURADO EXITOSAMENTE")
    print("=" * 80)
    print("\nPara entrenar AKOrN en MNIST, ejecuta:")
    print("\n  python train_classification.py mnist_test \\")
    print("    --data mnist \\")
    print("    --epochs 10 \\")
    print("    --n 2 \\")
    print("    --L 2 \\")
    print("    --T 3")
    print("\n" + "=" * 80)
    
    return True


if __name__ == "__main__":
    success = setup_mnist()
    exit(0 if success else 1)
