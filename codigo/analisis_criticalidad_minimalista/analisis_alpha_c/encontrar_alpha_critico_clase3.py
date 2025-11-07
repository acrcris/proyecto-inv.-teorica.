"""
Barrido de alpha para estimar el acoplamiento crítico SOLO para la clase 3.
Usa parámetros específicos optimizados.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch

from datasets.loader import MNISTLoader
from kuramoto.modelo import KBlock
from analisis.criticalidad import KuramotoMetrics
from utils import get_device, generate_alphas, prepare_image


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calcula alpha crítico para la clase 3 con parámetros específicos.",
    )
    parser.add_argument("--mnist-root", type=Path, default=Path("./data"))
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--num-images", type=int, default=10,
                       help="Número de imágenes de la clase 3 a procesar")
    parser.add_argument("--target-class", type=int, default=3,
                       help="Clase objetivo (default: 3)")
    parser.add_argument("--alpha-start", type=float, default=0.0)
    parser.add_argument("--alpha-end", type=float, default=2.0)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    
    # Parámetros del modelo Kuramoto
    parser.add_argument("--timesteps", type=int, default=100,
                       help="T_steps - pasos temporales")
    parser.add_argument("--gamma", type=float, default=0.7,
                       help="γ - fuerza de acoplamiento")
    parser.add_argument("--delta-t", type=float, default=0.9,
                       help="Δt - paso de integración")
    parser.add_argument("--ksize", type=int, default=7,
                       help="Tamaño de ventana convolucional")
    parser.add_argument("--init-omg", type=float, default=0.1,
                       help="Escala de frecuencias naturales")
    parser.add_argument("--ch", type=int, default=4,
                       help="Número de canales")
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("alpha_c_clase3.json"))
    parser.add_argument("--device", type=str)
    parser.add_argument("--window", type=int, default=5,
                       help="Número de pasos finales a promediar en R(t)")
    return parser


def _estimate_alpha_curve(
    kblock: KBlock,
    c_base: torch.Tensor,
    alphas: np.ndarray,
    *,
    timesteps: int,
    gamma: float,
    delta_t: float,
    window: int,
    device: torch.device,
) -> np.ndarray:
    """
    Para cada alpha, ejecuta la dinámica y calcula R promedio en la ventana final.
    Consistente con código de referencia.
    """
    order_values: List[float] = []
    for alpha in alphas:
        # Escalar campo por alpha y añadir batch dimension (consistente con ref)
        c_scaled = c_base.unsqueeze(0) * float(alpha)  # [1, ch, H, W]
        x0 = torch.randn_like(c_scaled, device=device)  # [1, ch, H, W]
        
        with torch.no_grad():
            _, xs = kblock(
                x0,
                c_scaled,
                T=timesteps,
                gamma=gamma,
                del_t=delta_t,
                return_xs=True,
            )
        
        # Calcular R(t) para toda la serie
        r_series = KuramotoMetrics.order_parameter(xs)
        if len(r_series) == 0:
            order_values.append(0.0)
            continue
        
        # Promediar los últimos 'window' pasos
        tail = r_series[-window:] if window < len(r_series) else r_series
        order_values.append(float(np.mean(tail)))
    
    return np.asarray(order_values, dtype=np.float64)


def _critical_alpha(order_curve: np.ndarray, alphas: np.ndarray) -> float:
    """
    Encuentra el alpha donde dR/dα es máximo (transición más rápida).
    """
    if len(order_curve) < 2:
        return float("nan")
    gradient = np.gradient(order_curve, alphas)
    idx = int(np.argmax(gradient))
    return float(alphas[idx])


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.device or 'auto')
    
    print(f"\n{'='*70}")
    print(f"Análisis de α_c para la clase {args.target_class}")
    print(f"{'='*70}\n")
    print(f"Parámetros del modelo Kuramoto:")
    print(f"  • T_steps (timesteps): {args.timesteps}")
    print(f"  • γ (gamma): {args.gamma}")
    print(f"  • Δt (delta_t): {args.delta_t}")
    print(f"  • ksize: {args.ksize}")
    print(f"  • init_omg: {args.init_omg}")
    print(f"  • canales: {args.ch}")
    print(f"  • ventana promedio: {args.window}")
    print(f"\nBarrido de α: [{args.alpha_start}, {args.alpha_end}] step={args.alpha_step}")
    print(f"Número de imágenes: {args.num_images}")
    print(f"Device: {device}\n")

    # Cargar dataset
    loader = MNISTLoader(root=str(args.mnist_root), img_size=args.img_size)
    dataset = loader.train_dataset

    # Generar valores de alpha
    alphas = generate_alphas(args.alpha_start, args.alpha_end, args.alpha_step)
    print(f"Puntos de alpha a evaluar: {len(alphas)}\n")

    # Construir KBlock con los parámetros especificados
    kblock = KBlock(
        n=args.ch,
        ch=args.ch,
        connectivity="conv",
        T=args.timesteps,
        ksize=args.ksize,
        init_omg=args.init_omg,
        c_norm=None,
        use_omega_c=False,
    ).to(device)
    kblock.eval()

    # Buscar imágenes de la clase objetivo
    print(f"Buscando {args.num_images} imágenes de la clase {args.target_class}...")
    target_images = []
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if label == args.target_class:
            target_images.append(img)
            if len(target_images) >= args.num_images:
                break
    
    if len(target_images) < args.num_images:
        print(f"⚠️  Solo se encontraron {len(target_images)} imágenes de la clase {args.target_class}")
    else:
        print(f"✓ Se encontraron {len(target_images)} imágenes\n")

    # Procesar cada imagen
    print(f"Procesando imágenes...")
    curves = []
    alpha_c_values = []
    
    for i, img in enumerate(target_images, 1):
        print(f"  Imagen {i}/{len(target_images)}...", end=" ")
        
        c_base = prepare_image(img, args.img_size, args.ch, device)
        
        # Calcular curva R(alpha) para esta imagen
        order_curve = _estimate_alpha_curve(
            kblock,
            c_base,
            alphas,
            timesteps=args.timesteps,
            gamma=args.gamma,
            delta_t=args.delta_t,
            window=args.window,
            device=device,
        )
        
        # Encontrar alpha crítico
        alpha_c = _critical_alpha(order_curve, alphas)
        
        curves.append(order_curve)
        alpha_c_values.append(alpha_c)
        
        print(f"α_c = {alpha_c:.3f}")
    
    # Calcular estadísticas
    curves_arr = np.stack(curves)
    mean_curve = curves_arr.mean(axis=0)
    std_curve = curves_arr.std(axis=0)
    alpha_c_array = np.asarray(alpha_c_values)
    alpha_c_mean = float(alpha_c_array.mean())
    alpha_c_std = float(alpha_c_array.std())
    
    print(f"\n{'='*70}")
    print(f"Resultados para la clase {args.target_class}:")
    print(f"{'='*70}")
    print(f"  α_c promedio: {alpha_c_mean:.3f} ± {alpha_c_std:.3f}")
    print(f"  α_c mínimo:   {alpha_c_array.min():.3f}")
    print(f"  α_c máximo:   {alpha_c_array.max():.3f}")
    print(f"  α_c mediana:  {np.median(alpha_c_array):.3f}")
    print(f"{'='*70}\n")
    
    # Guardar resultados
    summary = {
        "settings": {
            "target_class": args.target_class,
            "num_images": len(target_images),
            "alphas": alphas.tolist(),
            "timesteps": args.timesteps,
            "gamma": args.gamma,
            "delta_t": args.delta_t,
            "ksize": args.ksize,
            "init_omg": args.init_omg,
            "channels": args.ch,
            "window": args.window,
            "img_size": args.img_size,
        },
        "results": {
            "alpha_c_values": alpha_c_array.tolist(),
            "alpha_c_mean": alpha_c_mean,
            "alpha_c_std": alpha_c_std,
            "alpha_c_min": float(alpha_c_array.min()),
            "alpha_c_max": float(alpha_c_array.max()),
            "alpha_c_median": float(np.median(alpha_c_array)),
            "order_curve_mean": mean_curve.tolist(),
            "order_curve_std": std_curve.tolist(),
        }
    }
    
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"✓ Resultados guardados en: {args.output}")
    print(f"  Ruta completa: {args.output.absolute()}\n")


if __name__ == "__main__":
    main()
