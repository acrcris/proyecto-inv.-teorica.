#!/usr/bin/env python3
"""
analizar_alpha_por_clase.py

Barre valores de alpha (escala del campo c) y calcula R(alpha) para cada imagen.
Extrae alpha_c (punto de máximo cambio) por imagen y genera distribuciones por clase.

Salida:
 - resultados_alpha_por_imagen.csv (columns: idx, label, alpha_c, R_mean)
 - resumen_por_clase.csv (mediana, mean, std, normaltest p-value)
 - plots/ histogramas y qq-plots por clase
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from kuramoto.modelo import KBlock
from analisis.criticalidad import KuramotoMetrics
from datasets.loader import MNISTLoader
from scipy.stats import normaltest


RESULTS_DIR = "resultados_alpha"
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def compute_R_for_alpha(kblock, img, alpha, params):
    """Construye campo c escalado por alpha y ejecuta dinámica, devuelve R(t) series.

    Usa el argumento c_image y alpha del forward_with_params para construir c correctamente.
    """
    with torch.no_grad():
        device = next(kblock.parameters()).device
        # img viene como tensor [1,H,W] o [1,1,H,W]
        if img.ndim == 2:
            img_t = img.unsqueeze(0).unsqueeze(0).to(device)
        elif img.ndim == 3 and img.shape[0] == 1:
            img_t = img.unsqueeze(0).to(device)
        elif img.ndim == 4:
            img_t = img.to(device)
        else:
            img_t = img.unsqueeze(0).to(device)

        xs, es = kblock.forward_with_params(img_t, T_steps=params['T_steps'], gamma=params['gamma'], del_t=params['del_t'], c_image=img_t, alpha=alpha)
        R_series = KuramotoMetrics.order_parameter(xs, ch_pair=params.get('ch_pair', (0,1)))
        return R_series


def main(args):
    # parámetros
    alphas = np.arange(args.alpha_min, args.alpha_max + 1e-9, args.alpha_step)
    params = {
        'T_steps': args.T_steps,
        'gamma': args.gamma,
        'del_t': args.del_t,
        'ch_pair': tuple(args.ch_pair)
    }

    # cargar dataset (subset opcional)
    mn = MNISTLoader(root=args.data_root, batch_size=1, img_size=args.img_size)
    _, test_loader = mn.get_mnist(batch_size=1, train_split=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Usando dispositivo:', device)

    # modelo básico (usar ch del loader o fixed)
    kblock = KBlock(n=args.n, ch=args.ch, T=args.T_model, ksize=args.ksize, init_omg=args.init_omg).to(device)

    results = []

    # iterar sobre dataset
    max_images = args.max_images if args.max_images is not None else len(test_loader.dataset)
    count = 0
    save_every = 100
    for batch_idx, (img_batch, label_batch) in enumerate(tqdm(test_loader)):
        if count >= max_images:
            break
        img = img_batch[0]  # [1,H,W]
        label = int(label_batch[0].item())

        R_alpha_list = []
        for alpha in alphas:
            R_series = compute_R_for_alpha(kblock, img, alpha, params)
            R_t = np.array(R_series)
            R_mean_last = float(np.mean(R_t[-args.R_last:]))
            R_alpha_list.append(R_mean_last)

        R_alpha_arr = np.array(R_alpha_list)
        alpha_c, grad = KuramotoMetrics.compute_alpha_c(R_alpha_arr, alphas, smoothing=args.smoothing)

        results.append({'idx': batch_idx, 'label': label, 'alpha_c': alpha_c, 'R_mean': float(np.mean(R_alpha_arr))})
        count += 1

        # Guardado incremental
        if (count % save_every) == 0:
            pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, 'resultados_alpha_por_imagen_partial.csv'), index=False)


    df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, 'resultados_alpha_por_imagen.csv')
    df.to_csv(csv_path, index=False)
    print('Guardado:', csv_path)

    # resumen por clase
    resumen = []
    for cls in sorted(df['label'].unique()):
        vals = df[df['label'] == cls]['alpha_c'].values
        med = np.median(vals) if len(vals) > 0 else np.nan
        mean = np.mean(vals) if len(vals) > 0 else np.nan
        std = np.std(vals) if len(vals) > 0 else np.nan

        # elegir test de normalidad según tamaño de muestra
        p = np.nan
        test_name = None
        if len(vals) >= 8:
            try:
                stat, p = normaltest(vals)
                test_name = 'dagostino'
            except Exception:
                p = np.nan
        elif len(vals) >= 3:
            try:
                from scipy.stats import shapiro
                stat, p = shapiro(vals)
                test_name = 'shapiro'
            except Exception:
                p = np.nan
        else:
            p = np.nan

        resumen.append({'label': cls, 'n': len(vals), 'median': med, 'mean': mean, 'std': std, 'normaltest_pvalue': p, 'test_used': test_name})

        # plot histograma solo si hay suficientes muestras
        if len(vals) >= 5:
            plt.figure(figsize=(6,4))
            plt.hist(vals, bins=20, color='C0', alpha=0.7)
            plt.title(f'Clase {cls} - alpha_c (n={len(vals)})\n{test_name} p={p:.3g}')
            plt.xlabel('alpha_c')
            plt.ylabel('count')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(PLOTS_DIR, f'hist_alpha_c_class_{cls}.png'))
            plt.close()

        # qq-plot si al menos 8 muestras
        if len(vals) >= 8:
            try:
                import scipy.stats as st
                plt.figure(figsize=(5,5))
                st.probplot(vals, dist='norm', plot=plt)
                plt.title(f'QQ-plot Clase {cls}')
                plt.savefig(os.path.join(PLOTS_DIR, f'qq_alpha_c_class_{cls}.png'))
                plt.close()
            except Exception:
                pass

    df_res = pd.DataFrame(resumen)
    df_res.to_csv(os.path.join(RESULTS_DIR, 'resumen_por_clase.csv'), index=False)
    print('Resumen por clase guardado')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--alpha_min', type=float, default=0.0)
    parser.add_argument('--alpha_max', type=float, default=2.0)
    parser.add_argument('--alpha_step', type=float, default=0.01)
    parser.add_argument('--T_steps', type=int, default=100)
    parser.add_argument('--R_last', type=int, default=5, help='últimos pasos para promediar R')
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--del_t', type=float, default=0.9)
    parser.add_argument('--smoothing', type=int, default=3)
    parser.add_argument('--max_images', type=int, default=100)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--n', type=int, default=4)
    parser.add_argument('--ch', type=int, default=4)
    parser.add_argument('--T_model', type=int, default=4)
    parser.add_argument('--ksize', type=int, default=7)
    parser.add_argument('--init_omg', type=float, default=0.1)
    parser.add_argument('--ch_pair', nargs=2, type=int, default=[0,1])

    args = parser.parse_args()
    main(args)
