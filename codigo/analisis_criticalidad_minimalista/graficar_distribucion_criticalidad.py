#!/usr/bin/env python3
"""
graficar_distribucion_criticalidad.py

Genera histogramas y boxplots de la distribución de α_c por clase
usando el archivo de distribución más grande generado por el análisis incremental.

Uso:
    python graficar_distribucion_criticalidad.py --input distribuciones_impares/distribucion_acumulativa_0700imgs.pt
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser(description='Graficar distribución de α_c por clase')
parser.add_argument('--input', type=str, required=True, help='Archivo de distribución acumulativa (.pt)')
parser.add_argument('--output', type=str, default='graficas_distribucion_criticalidad', help='Directorio de salida para las gráficas')
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

# Fix PyTorch >=2.6: weights_only=False para cargar archivos propios
try:
    data = torch.load(args.input, map_location='cpu', weights_only=False)
except TypeError:
    data = torch.load(args.input, map_location='cpu')
clases = data['clases_impares'] if 'clases_impares' in data else data['clases_pares']

fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# Histograma por clase
for clase in clases:
    alphas_c = np.array(data['alphas_c'][clase])
    n, bins, patches = axs[0].hist(alphas_c, bins=30, alpha=0.7, label=f'Clase {clase}')
    mean = np.mean(alphas_c)
    std = np.std(alphas_c)
    # Usar el color del primer patch del histograma
    color = patches[0].get_facecolor() if patches else 'k'
    axs[0].axvline(mean, color=color, linestyle='--', linewidth=2)
    print(f"Clase {clase}: α_c = {mean:.5f} ± {std:.5f} (n={len(alphas_c)})")

axs[0].set_title('Histograma de α_c por clase')
axs[0].set_xlabel('α_c')
axs[0].set_ylabel('Frecuencia')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Boxplot por clase
alphas_c_all = [np.array(data['alphas_c'][clase]) for clase in clases]
axs[1].boxplot(alphas_c_all, labels=[str(c) for c in clases], notch=True, patch_artist=True)
axs[1].set_title('Boxplot de α_c por clase')
axs[1].set_xlabel('Clase')
axs[1].set_ylabel('α_c')
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(args.output, 'distribucion_criticalidad.png'), dpi=150)
print(f"Gráfica guardada en: {os.path.join(args.output, 'distribucion_criticalidad.png')}")
plt.show()
