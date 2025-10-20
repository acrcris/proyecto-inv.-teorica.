"""
Script: run_kuramoto_por_clase.py

Corre la dinámica de Kuramoto (KBlock) sobre una imagen representativa de cada clase (0-9) del dataset MNIST.
Guarda los estados de la evolución para cada clase en archivos separados.
"""

import os
import sys
import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np

# Agregar el path del módulo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kuramoto.modelo import KBlock

# Carpeta de salida para los resultados
RESULTS_DIR = "resultados_kuramoto_por_clase"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Preparar el dataset MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 2. Seleccionar una imagen por clase (0-9)
imagenes_por_clase = {i: None for i in range(10)}
for img, label in train_dataset:
    if imagenes_por_clase[label] is None:
        imagenes_por_clase[label] = img
    if all(v is not None for v in imagenes_por_clase.values()):
        break

# 3. Configuración del modelo Kuramoto
ch = 4
n = 4
h, w = 64, 64
T = 100

def procesar_imagen(img, clase):
    # Redimensionar y replicar canales
    img_resized = TF.resize(img, [h, w])
    img_channels = img_resized.repeat(ch, 1, 1)
    c = img_channels.unsqueeze(0)
    # Estados iniciales aleatorios
    x = torch.randn(1, ch, h, w)
    # Modelo
    kblock = KBlock(n=n, ch=ch, connectivity='conv', T=T, ksize=3, init_omg=0.1, c_norm=None, use_omega_c=False)
    # Evolución
    x_final, xs, es = kblock(x, c, T=T, gamma=0.7, del_t=0.9, return_xs=True, return_es=True)
    # Guardar resultados
    out_path = os.path.join(RESULTS_DIR, f"clase_{clase}.pt")
    torch.save({
        'img': img,
        'xs': xs,
        'es': es
    }, out_path)
    print(f"✅ Guardado: {out_path}")

# 4. Ejecutar dinámica para cada clase
def main():
    print("Ejecutando dinámica de Kuramoto sobre una imagen por clase (0-9)...")
    for clase in range(10):
        img = imagenes_por_clase[clase]
        print(f"Procesando clase {clase}...")
        procesar_imagen(img, clase)
    print("\n¡Listo! Los estados de la evolución están guardados en la carpeta:")
    print(f"  {RESULTS_DIR}")

if __name__ == "__main__":
    main()
