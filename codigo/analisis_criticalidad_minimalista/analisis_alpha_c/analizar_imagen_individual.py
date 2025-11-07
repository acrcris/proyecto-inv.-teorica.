"""
Análisis detallado de UNA imagen individual.
Calcula y grafica la curva R(α) para una sola imagen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from datasets.loader import MNISTLoader
from kuramoto.modelo import KBlock
from analisis.criticalidad import KuramotoMetrics
from utils import get_device, generate_alphas, prepare_image, setup_matplotlib, save_figure

# Configurar matplotlib
setup_matplotlib()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analiza UNA imagen individual y grafica su curva R(α).",
    )
    parser.add_argument("--mnist-root", type=Path, default=Path("./data"))
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--target-class", type=int, default=1,
                       help="Clase objetivo")
    parser.add_argument("--image-index", type=int, default=0,
                       help="Índice de la imagen dentro de la clase (0=primera)")
    
    # Rango de alpha
    parser.add_argument("--alpha-start", type=float, default=0.0)
    parser.add_argument("--alpha-end", type=float, default=2.0)
    parser.add_argument("--alpha-step", type=float, default=0.02,
                       help="Paso más fino para curva suave")
    
    # Parámetros del modelo Kuramoto
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--delta-t", type=float, default=0.9)
    parser.add_argument("--ksize", type=int, default=7)
    parser.add_argument("--init-omg", type=float, default=0.1)
    parser.add_argument("--ch", type=int, default=4)
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("imagen_individual.json"))
    parser.add_argument("--device", type=str)
    parser.add_argument("--window", type=int, default=5)
    
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.device or 'auto')
    
    print(f"\n{'='*70}")
    print(f"Análisis de UNA imagen individual")
    print(f"{'='*70}\n")
    print(f"Clase objetivo: {args.target_class}")
    print(f"Imagen dentro de la clase: #{args.image_index} (0=primera)\n")
    print(f"Parámetros del modelo:")
    print(f"  • T_steps: {args.timesteps}")
    print(f"  • γ: {args.gamma}")
    print(f"  • Δt: {args.delta_t}")
    print(f"  • ksize: {args.ksize}")
    print(f"  • init_omg: {args.init_omg}")
    print(f"  • canales: {args.ch}")
    print(f"\nDevice: {device}\n")

    # Cargar dataset
    loader = MNISTLoader(root=str(args.mnist_root), img_size=args.img_size)
    dataset = loader.train_dataset

    # Buscar la imagen específica
    print(f"Buscando imagen #{args.image_index} de la clase {args.target_class}...")
    count = 0
    target_img = None
    
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if label == args.target_class:
            if count == args.image_index:
                target_img = img
                break
            count += 1
    
    if target_img is None:
        raise ValueError(f"No se encontró la imagen #{args.image_index} de la clase {args.target_class}")
    
    print(f"✓ Imagen encontrada\n")

    # Construir KBlock
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

    # Preparar imagen
    c_base = prepare_image(target_img, args.img_size, args.ch, device)

    # Generar valores de alpha (más denso para curva suave)
    alphas = generate_alphas(args.alpha_start, args.alpha_end, args.alpha_step)
    print(f"Evaluando {len(alphas)} puntos de α en [{args.alpha_start}, {args.alpha_end}]...\n")

    # Calcular R(α) - Consistente con código de referencia
    order_values = []
    print("Progreso:", end=" ")
    
    for i, alpha in enumerate(alphas):
        if i % 10 == 0:
            print(f"{i}/{len(alphas)}", end="...", flush=True)
        
        # Escalar y añadir batch dimension (consistente con ref)
        c_scaled = c_base.unsqueeze(0) * float(alpha)  # [1, ch, H, W]
        x0 = torch.randn_like(c_scaled, device=device)  # [1, ch, H, W]
        
        with torch.no_grad():
            _, xs = kblock(
                x0,
                c_scaled,
                T=args.timesteps,
                gamma=args.gamma,
                del_t=args.delta_t,
                return_xs=True,
            )
        
        # Calcular R(t) usando método correcto
        r_series = KuramotoMetrics.order_parameter(xs)
        if len(r_series) == 0:
            order_values.append(0.0)
            continue
        
        # Promediar ventana final (últimos 'window' pasos)
        tail = r_series[-args.window:] if args.window < len(r_series) else r_series
        order_values.append(float(np.mean(tail)))
    
    print(f" {len(alphas)}/{len(alphas)} ✓\n")
    
    order_curve = np.array(order_values)
    
    # Encontrar α_c (máximo del gradiente)
    gradient = np.gradient(order_curve, alphas)
    alpha_c_idx = int(np.argmax(gradient))
    alpha_c = float(alphas[alpha_c_idx])
    r_at_alpha_c = order_curve[alpha_c_idx]
    
    print(f"{'='*70}")
    print(f"Resultados:")
    print(f"{'='*70}")
    print(f"  α_c (punto crítico): {alpha_c:.4f}")
    print(f"  R(α_c):              {r_at_alpha_c:.4f}")
    print(f"  dR/dα máximo:        {gradient[alpha_c_idx]:.4f}")
    print(f"{'='*70}\n")
    
    # Guardar resultados
    results = {
        "settings": {
            "target_class": args.target_class,
            "image_index": args.image_index,
            "timesteps": args.timesteps,
            "gamma": args.gamma,
            "delta_t": args.delta_t,
            "ksize": args.ksize,
            "init_omg": args.init_omg,
            "channels": args.ch,
            "window": args.window,
            "img_size": args.img_size,
        },
        "alphas": alphas.tolist(),
        "order_curve": order_curve.tolist(),
        "gradient": gradient.tolist(),
        "alpha_c": alpha_c,
        "r_at_alpha_c": r_at_alpha_c,
        "max_gradient": float(gradient[alpha_c_idx]),
    }
    
    args.output.write_text(json.dumps(results, indent=2))
    print(f"✓ Datos guardados en: {args.output}\n")
    
    # GRAFICAR
    print("Generando gráficas...\n")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Panel 1: R(α)
    ax1 = axes[0]
    ax1.plot(alphas, order_curve, linewidth=3, color='steelblue', label='R(α)')
    ax1.scatter(alpha_c, r_at_alpha_c, s=300, color='red', marker='*', 
               edgecolors='black', linewidths=2, zorder=10,
               label=f'α_c = {alpha_c:.4f}')
    ax1.axvline(alpha_c, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axhline(r_at_alpha_c, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    
    ax1.set_xlabel('α (acoplamiento externo)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('R (parámetro de orden)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Curva R(α) - Clase {args.target_class}, Imagen #{args.image_index}',
                 fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([alphas[0], alphas[-1]])
    ax1.set_ylim([0, 1])
    
    # Panel 2: Gradiente dR/dα
    ax2 = axes[1]
    ax2.plot(alphas, gradient, linewidth=2.5, color='darkgreen', label='dR/dα')
    ax2.scatter(alpha_c, gradient[alpha_c_idx], s=300, color='red', marker='*',
               edgecolors='black', linewidths=2, zorder=10,
               label=f'Máximo en α_c = {alpha_c:.4f}')
    ax2.axvline(alpha_c, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.4)
    
    ax2.set_xlabel('α (acoplamiento externo)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('dR/dα (gradiente)', fontsize=13, fontweight='bold')
    ax2.set_title('Gradiente de la transición de fase', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=12, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([alphas[0], alphas[-1]])
    
    plt.tight_layout()
    
    # Guardar gráfica
    output_plot = args.output.parent / f'curva_R_alpha_clase{args.target_class}_img{args.image_index}.png'
    save_figure(fig, output_plot)
    print(f"✓ Gráfica guardada en: {output_plot}\n")
    
    # Mostrar gráfica
    plt.show()


if __name__ == "__main__":
    main()
