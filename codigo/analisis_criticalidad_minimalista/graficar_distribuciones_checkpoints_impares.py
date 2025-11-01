"""
Genera gráficas de la distribución de α_c usando los checkpoints
impares con 100 imágenes por clase, ubicados en:

  codigo/analisis_criticalidad_minimalista/checkpoints_impares/

Lee cada archivo checkpoint_clase{1,3,5,7,9}_0100imgs.pt (formato zip con
data.pkl) y extrae la lista 'alphas_c'. Produce:
  - histogramas_por_clase.png (subplots por clase)
  - boxplot_clases.png (boxplot por clase)
  - violin_clases.png (violin por clase)
  - histograma_global.png (histograma global de todas las clases)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os
from zipfile import ZipFile
import pickle
import numpy as np
"""Configurar caché de Matplotlib en carpeta local y backend Agg."""
_mpl_dir = Path(__file__).parent / "_mplconfig"
try:
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))
except Exception:
    # En caso de fallo, continuar y dejar que Matplotlib use un tmp
    pass

import matplotlib
matplotlib.use("Agg")  # Backend no interactivo para entornos sin display
import matplotlib.pyplot as plt


def load_alphas_c_from_checkpoint(path: Path) -> tuple[int, np.ndarray]:
    """Abre un checkpoint .pt (zip) y retorna (clase, alphas_c como float np.array)."""
    with ZipFile(path, 'r') as z:
        # Buscar '.../data.pkl' dentro del zip
        name = next(n for n in z.namelist() if n.endswith('/data.pkl'))
        data = z.read(name)
    obj = pickle.loads(data)
    clase = int(obj.get('clase'))
    alphas_c = np.array(obj.get('alphas_c'), dtype=float)
    return clase, alphas_c


def collect_distributions(checkpoints_dir: Path) -> dict[int, np.ndarray]:
    """Carga los alphas_c por clase (1,3,5,7,9) para archivos de 100 imgs."""
    per_class: dict[int, np.ndarray] = {}
    files = sorted(checkpoints_dir.glob('checkpoint_clase*_0100imgs.pt'))
    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos *_0100imgs.pt en {checkpoints_dir}")
    for f in files:
        clase, vals = load_alphas_c_from_checkpoint(f)
        per_class[clase] = vals
    return per_class


def make_histograms_by_class(per_class: dict[int, np.ndarray], out_dir: Path) -> Path:
    clases = sorted(per_class.keys())
    all_vals = np.concatenate([per_class[c] for c in clases])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    # Asegurar rango algo ancho
    rng = vmax - vmin if vmax > vmin else 1e-3
    vmin_plot = max(0.0, vmin - 0.05 * rng)
    vmax_plot = vmax + 0.05 * rng

    fig, axes = plt.subplots(1, len(clases), figsize=(4*len(clases), 3.5), sharey=True)
    if len(clases) == 1:
        axes = [axes]
    for ax, c in zip(axes, clases):
        vals = per_class[c]
        ax.hist(vals, bins=20, range=(vmin_plot, vmax_plot), color='steelblue', alpha=0.75, edgecolor='black')
        mu = float(np.mean(vals))
        med = float(np.median(vals))
        ax.axvline(mu, color='red', linestyle='--', linewidth=2, label=f'μ={mu:.4f}')
        ax.axvline(med, color='purple', linestyle=':', linewidth=2, label=f'med={med:.4f}')
        ax.set_title(f'Clase {c}')
        ax.set_xlabel('α_c')
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('Frecuencia')
    # Una sola leyenda combinada
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = out_dir / 'histogramas_por_clase.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


def make_boxplot(per_class: dict[int, np.ndarray], out_dir: Path) -> Path:
    clases = sorted(per_class.keys())
    data = [per_class[c] for c in clases]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bp = ax.boxplot(data, labels=clases, patch_artist=True)
    # Estética simple
    colors = plt.cm.viridis(np.linspace(0, 1, len(clases)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xlabel('Clase (dígito)')
    ax.set_ylabel('α_c')
    ax.set_title('Distribución de α_c por clase (boxplot)')
    ax.grid(axis='y', alpha=0.3)
    out_path = out_dir / 'boxplot_clases.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


def make_violin(per_class: dict[int, np.ndarray], out_dir: Path) -> Path:
    clases = sorted(per_class.keys())
    data = [per_class[c] for c in clases]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    parts = ax.violinplot(data, positions=list(range(1, len(clases)+1)), showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(clases)+1))
    ax.set_xticklabels(clases)
    ax.set_xlabel('Clase (dígito)')
    ax.set_ylabel('α_c')
    ax.set_title('Distribución de α_c por clase (violín)')
    ax.grid(axis='y', alpha=0.3)
    out_path = out_dir / 'violin_clases.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


def make_global_hist(per_class: dict[int, np.ndarray], out_dir: Path) -> Path:
    clases = sorted(per_class.keys())
    concat = np.concatenate([per_class[c] for c in clases])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(concat, bins=30, color='tab:orange', alpha=0.8, edgecolor='black')
    mu = float(np.mean(concat))
    med = float(np.median(concat))
    ax.axvline(mu, color='red', linestyle='--', linewidth=2, label=f'μ={mu:.4f}')
    ax.axvline(med, color='purple', linestyle=':', linewidth=2, label=f'med={med:.4f}')
    ax.set_xlabel('α_c (global impares)')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución global de α_c (clases impares, 100 imgs c/u)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    out_path = out_dir / 'histograma_global.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Graficar distribuciones de α_c de checkpoints impares (100 imgs).')
    parser.add_argument('--checkpoints-dir', type=Path, default=Path('codigo/analisis_criticalidad_minimalista/checkpoints_impares'),
                        help='Directorio con los checkpoints *_0100imgs.pt')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Directorio de salida para las gráficas')
    args = parser.parse_args()

    checkpoints_dir = args.checkpoints_dir
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f'No existe {checkpoints_dir}')

    per_class = collect_distributions(checkpoints_dir)

    # Directorio de salida por defecto junto al directorio de checkpoints
    out_dir = args.output_dir or (checkpoints_dir / 'graficas_0100imgs')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Clases cargadas: {sorted(per_class.keys())}")
    for c in sorted(per_class.keys()):
        vals = per_class[c]
        print(f"  Clase {c}: n={len(vals)} μ={np.mean(vals):.6f} σ={np.std(vals):.6f} min={vals.min():.6f} med={np.median(vals):.6f} max={vals.max():.6f}")

    paths = []
    paths.append(make_histograms_by_class(per_class, out_dir))
    paths.append(make_boxplot(per_class, out_dir))
    paths.append(make_violin(per_class, out_dir))
    paths.append(make_global_hist(per_class, out_dir))

    print("\n✓ Gráficas generadas:")
    for p in paths:
        print(f"  - {p}")


if __name__ == '__main__':
    main()
