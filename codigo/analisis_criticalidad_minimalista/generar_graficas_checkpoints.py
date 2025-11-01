#!/usr/bin/env python3
"""
generar_graficas_checkpoints.py

Genera gráficas (histograma, boxplot y serie temporal) de las listas `alphas_c`
almacenadas en los checkpoints de `checkpoints_impares/`. Selecciona para cada
clase el checkpoint con más imágenes y guarda los PNGs en `temp/`.

Uso:
    python generar_graficas_checkpoints.py

"""
import re
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(__file__)
CHK_DIR = os.path.join(ROOT, 'checkpoints_impares')
OUT_DIR = os.path.join(ROOT, 'temp')
os.makedirs(OUT_DIR, exist_ok=True)

CKPT_RE = re.compile(r'checkpoint_clase(\d+)_(\d+)imgs\.pt')

def find_largest_checkpoints():
    best = {}
    for fname in os.listdir(CHK_DIR):
        m = CKPT_RE.match(fname)
        if not m:
            continue
        clase = int(m.group(1))
        n = int(m.group(2))
        cur = best.get(clase)
        if cur is None or n > cur[0]:
            best[clase] = (n, os.path.join(CHK_DIR, fname))
    return best

def load_checkpoint(path):
    # PyTorch >=2.6 safety: use weights_only=False to allow loading custom objects
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        data = torch.load(path, map_location='cpu')
    return data

def plot_for_class(clase, n_imgs, path, out_dir=OUT_DIR):
    data = load_checkpoint(path)
    alphas = np.array(data.get('alphas_c', []))
    if alphas.size == 0:
        print(f'Clase {clase}: no hay datos en {path} — saltando')
        return None

    fig, axs = plt.subplots(3,1, figsize=(8,10))

    # Histograma
    counts, bins, patches = axs[0].hist(alphas, bins=40, color='C0', alpha=0.8)
    mean = float(np.mean(alphas))
    std = float(np.std(alphas))
    axs[0].axvline(mean, color='red', linestyle='--', linewidth=2, label=f'μ={mean:.5f}')
    axs[0].set_title(f'Clase {clase} — Histograma α_c (n={len(alphas)})')
    axs[0].set_xlabel('α_c')
    axs[0].legend()

    # Anotar la media y desviación estándar
    axs[0].text(0.98, 0.95, f'μ={mean:.5f}\nσ={std:.5f}', transform=axs[0].transAxes,
                horizontalalignment='right', verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    # Boxplot
    axs[1].boxplot(alphas, notch=True, patch_artist=True, boxprops=dict(facecolor='C1'))
    axs[1].set_title('Boxplot de α_c')
    axs[1].set_xticks([1]); axs[1].set_xticklabels([str(clase)])

    # Serie temporal (índice de imagen)
    axs[2].plot(np.arange(len(alphas)), alphas, marker='.', ms=3, lw=0.6)
    axs[2].axhline(mean, color='red', linestyle='--', linewidth=1.5, label=f'μ={mean:.5f}')
    axs[2].set_title('α_c por índice de imagen')
    axs[2].set_xlabel('Índice de imagen')
    axs[2].set_ylabel('α_c')
    axs[2].legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'distribucion_clase{clase}_{n_imgs}imgs.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Guardado: {out_path}  (μ={mean:.6f}, σ={std:.6f}, n={len(alphas)})')
    return out_path

def main():
    best = find_largest_checkpoints()
    if not best:
        print('No se encontraron checkpoints en', CHK_DIR)
        return
    print('Checkpoints seleccionados (mayor n por clase):')
    for clase, (n, path) in sorted(best.items()):
        print(f' - Clase {clase}: {n} imgs -> {os.path.basename(path)}')

    for clase, (n, path) in sorted(best.items()):
        plot_for_class(clase, n, path)

if __name__ == '__main__':
    main()
