"""Calcular el parámetro de orden de Kuramoto R en el punto de criticalidad.

Usa `KBlock` para integrar la dinámica con un acoplamiento fijo C_crit
(y gamma) y `KuramotoMetrics.order_parameter` para obtener la serie R(t).

Se construyen distribuciones globales y por clase de un R estacionario
(definido como el promedio de los últimos `r_tail` pasos de la dinámica).
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from analisis_criticalidad_minimalista.kuramoto import KBlock
from analisis_criticalidad_minimalista.analisis.criticalidad import KuramotoMetrics


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_kblock(img_size=28, n=3, gamma=0.7, del_t=0.9, ksize=3, init_omg=0.1):
    """Construye un KBlock con parámetros razonables para MNIST.

    Nota: KBlock no usa directamente `img_size` en su constructor, pero lo
    dejamos aquí por si en el futuro se quiere validar dimensiones.
    """
    ch = n * 2  # dos canales por oscilador (real, imag) como en la práctica habitual
    kblock = KBlock(
        n=n,
        ch=ch,
        connectivity="conv",
        T=4,
        ksize=ksize,
        init_omg=init_omg,
        c_norm="gn",
        use_omega=True,
        use_omega_c=True,
    )
    # Guardamos parámetros dinámicos típicos; se pasan luego a forward
    kblock._gamma_default = gamma
    kblock._del_t_default = del_t
    return kblock


def prepare_input_from_mnist(x, n=3, device=torch.device("cpu")):
    """Convierte una imagen MNIST [1, 1, H, W] a estado y campo `c`.

    Estrategia simple: duplicar el canal de intensidad en (cos, sen) de una
    fase proporcional a la intensidad. Esto garantiza dos canales por
    oscilador y permite que KuramotoMetrics use ch_pair=(0,1).
    """
    # x: [B=1, 1, H, W], valores en [0,1]
    x = x.to(device)
    # Escalamos intensidad a una fase en [0, 2pi]
    phase = x * (2 * np.pi)
    cos_part = torch.cos(phase)
    sin_part = torch.sin(phase)
    # Concatenamos en el canal: [1, 2, H, W]
    state = torch.cat([cos_part, sin_part], dim=1)
    # Repetimos para n osciladores por canal agrupado: ch = 2 * n
    state = state.repeat(1, n, 1, 1)  # [1, 2*n, H, W]
    c = torch.zeros_like(state)
    return state, c


def main(
    data_root,
    batch_size,
    num_workers,
    c_crit,
    t_steps,
    r_tail,
    max_images,
    output_dir,
    n,
    gamma,
    del_t,
):
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    print(f"Usando dispositivo: {device}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # MNIST es 28x28; no redimensionamos por defecto
        ]
    )

    dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    if max_images is not None:
        indices = list(range(min(max_images, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    kblock = build_kblock(img_size=28, n=n, gamma=gamma, del_t=del_t)
    kblock.to(device)
    kblock.eval()

    metrics = KuramotoMetrics()

    R_global = []
    R_por_clase = defaultdict(list)

    total = len(loader)
    for idx, (x, y) in enumerate(loader):
        # Usamos batch_size=1 para asociar un R por imagen
        x = x.to(device)
        label = int(y[0].item())

        state, c = prepare_input_from_mnist(x, n=n, device=device)

        with torch.no_grad():
            # forward: devuelve (x_final, xs, es) si return_xs y return_es
            x_final, xs, es = kblock(
                state,
                c + c_crit,  # acoplamiento externo efectivo
                T=t_steps,
                gamma=gamma,
                del_t=del_t,
                return_xs=True,
                return_es=False,
            )

        # xs es una lista de tensores [B, ch, H, W]
        R_series = metrics.order_parameter(xs, ch_pair=(0, 1))  # np.array (T+1,)

        if r_tail == 1:
            R_final = float(R_series[-1])
        else:
            R_final = float(R_series[-r_tail:].mean())

        R_global.append(R_final)
        R_por_clase[label].append(R_final)

        if (idx + 1) % 500 == 0 or (idx + 1) == total:
            print(f"Procesadas {idx + 1}/{total} imágenes...")

    R_global = np.array(R_global)
    print(f"Total de imágenes procesadas: {len(R_global)}")
    print(f"R_global: media={R_global.mean():.4f}, std={R_global.std():.4f}")

    # Estadísticas por clase
    for c_label in sorted(R_por_clase.keys()):
        arr = np.array(R_por_clase[c_label])
        print(f"Clase {c_label}: n={len(arr)}, media={arr.mean():.4f}, std={arr.std():.4f}")

    # Histograma global
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(R_global, bins=40, alpha=0.8, color="C0")
    ax.set_xlabel("R en criticalidad")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Distribución global de R (C = {c_crit})")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"R_en_criticalidad_global_C_{c_crit}.pdf"))
    plt.close(fig)

    # Histogramas por clase
    fig, axes = plt.subplots(2, 5, figsize=(12, 5), sharex=True, sharey=True)
    axes = axes.ravel()
    for c_label in range(10):
        arr = np.array(R_por_clase.get(c_label, []))
        ax = axes[c_label]
        if len(arr) > 0:
            ax.hist(arr, bins=30, alpha=0.8)
        ax.set_title(f"Clase {c_label}")
    fig.suptitle(f"Distribución de R por clase (C = {c_crit})")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, f"R_en_criticalidad_por_clase_C_{c_crit}.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calcular parámetro de orden R en punto de criticalidad usando KuramotoMetrics."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../data",
        help="Directorio raíz de datos (MNIST).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Tamaño de batch (usar 1 para calcular R por imagen).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Num. de workers para DataLoader.",
    )
    parser.add_argument(
        "--c-crit",
        type=float,
        default=0.1769,
        help="Valor de acoplamiento en el punto crítico.",
    )
    parser.add_argument(
        "--t-steps",
        type=int,
        default=30,
        help="Número de pasos de la dinámica Kuramoto (longitud de xs).",
    )
    parser.add_argument(
        "--r-tail",
        type=int,
        default=5,
        help="Número de últimos pasos para promediar R (1 = solo último).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Máximo de imágenes a procesar (None = todas).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="resultados_R_en_criticalidad",
        help="Directorio de salida para las gráficas.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Número de osciladores por canal agrupado (n en KBlock).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        help="Factor de acoplamiento gamma.",
    )
    parser.add_argument(
        "--del-t",
        type=float,
        default=0.9,
        help="Paso temporal de integración.",
    )

    args = parser.parse_args()

    main(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        c_crit=args.c_crit,
        t_steps=args.t_steps,
        r_tail=args.r_tail,
        max_images=args.max_images,
        output_dir=args.output_dir,
        n=args.n,
        gamma=args.gamma,
        del_t=args.del_t,
    )
