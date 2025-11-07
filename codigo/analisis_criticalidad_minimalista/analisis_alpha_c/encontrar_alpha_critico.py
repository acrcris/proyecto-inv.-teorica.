"""Barrido de alpha para estimar el acoplamiento crítico por clase en el modelo Kuramoto minimalista."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from datasets.loader import MNISTLoader
from kuramoto.modelo import KBlock
from analisis.criticalidad import KuramotoMetrics
from utils import get_device, generate_alphas, prepare_image


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calcula alpha crítico promedio por clase usando el bloque Kuramoto minimalista.",
    )
    parser.add_argument("--mnist-root", type=Path, default=Path("./data"))
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--max-images-per-class", type=int, default=25)
    parser.add_argument("--alpha-start", type=float, default=0.0)
    parser.add_argument("--alpha-end", type=float, default=2.0)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--delta-t", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=str)
    parser.add_argument("--window", type=int, default=5, help="Número de pasos finales a promediar en R(t)")
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
    order_values: List[float] = []
    for alpha in alphas:
        c_scaled = c_base * float(alpha)
        x0 = torch.randn_like(c_base, device=device)
        with torch.no_grad():
            _, xs = kblock(
                x0,
                c_scaled,
                T=timesteps,
                gamma=gamma,
                del_t=delta_t,
                return_xs=True,
            )
        r_series = KuramotoMetrics.order_parameter(xs)
        if len(r_series) == 0:
            order_values.append(0.0)
            continue
        tail = r_series[-window:] if window < len(r_series) else r_series
        order_values.append(float(np.mean(tail)))
    return np.asarray(order_values, dtype=np.float64)


def _critical_alpha(order_curve: np.ndarray, alphas: np.ndarray) -> float:
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

    loader = MNISTLoader(root=str(args.mnist_root), img_size=args.img_size)
    dataset = loader.train_dataset

    alphas = generate_alphas(args.alpha_start, args.alpha_end, args.alpha_step)

    ch = 4
    kblock = KBlock(
        n=ch,
        ch=ch,
        connectivity="conv",
        T=args.timesteps,
        ksize=3,
        init_omg=0.1,
        c_norm=None,
        use_omega_c=False,
    ).to(device)
    kblock.eval()

    per_class_curves: Dict[int, List[np.ndarray]] = defaultdict(list)
    per_class_alphas: Dict[int, List[float]] = defaultdict(list)

    counts = defaultdict(int)
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if counts[label] >= args.max_images_per_class:
            continue

        c_base = prepare_image(img, args.img_size, ch, device)
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
        alpha_c = _critical_alpha(order_curve, alphas)
        per_class_curves[int(label)].append(order_curve)
        per_class_alphas[int(label)].append(alpha_c)
        counts[label] += 1

        if all(counts[c] >= args.max_images_per_class for c in range(10)):
            break

    results = {}
    for cls in range(10):
        curves = per_class_curves.get(cls, [])
        alphas_cls = per_class_alphas.get(cls, [])
        if not curves:
            continue
        curves_arr = np.stack(curves)
        mean_curve = curves_arr.mean(axis=0)
        std_curve = curves_arr.std(axis=0)
        alpha_c_array = np.asarray(alphas_cls)
        results[cls] = {
            "alpha_c_values": alpha_c_array.tolist(),
            "alpha_c_mean": float(alpha_c_array.mean()),
            "alpha_c_std": float(alpha_c_array.std()),
            "order_curve_mean": mean_curve.tolist(),
            "order_curve_std": std_curve.tolist(),
        }

    summary = {
        "settings": {
            "alphas": alphas.tolist(),
            "max_images_per_class": args.max_images_per_class,
            "timesteps": args.timesteps,
            "gamma": args.gamma,
            "delta_t": args.delta_t,
            "window": args.window,
            "img_size": args.img_size,
        },
        "results": results,
    }

    for cls in sorted(results):
        data = results[cls]
        print(
            f"Clase {cls}: alpha_c_mean={data['alpha_c_mean']:.3f} +/- {data['alpha_c_std']:.3f}"
        )

    if args.output:
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"Guardado resumen en {args.output}")


if __name__ == "__main__":
    main()
