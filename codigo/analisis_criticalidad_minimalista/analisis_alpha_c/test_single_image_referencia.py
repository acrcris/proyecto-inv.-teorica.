#!/usr/bin/env python3
"""
test_single_image_referencia.py

Test de R vs α para UNA imagen usando la función de referencia
de analizar_incremental_clases_impares.py (consistente con rama main).

Uso:
    python test_single_image_referencia.py --clase 3 --img-index 0
"""
import os
import sys
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from pathlib import Path
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kuramoto.modelo import KBlock
from datasets.loader import MNISTLoader


def get_device(device_arg='auto'):
    """Detecta el mejor dispositivo disponible."""
    if device_arg == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("✅ Usando GPU: Metal Performance Shaders (MPS)")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✅ Usando GPU: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            device = torch.device('cpu')
            print("⚠️  Usando CPU (GPU no disponible)")
    else:
        device = torch.device(device_arg)
        print(f"✅ Usando dispositivo: {device_arg}")
    
    return device


def calcular_curva_imagen(kblock, img, alphas, ch=4, h=64, w=64, T=100, device='cpu'):
    """
    Calcula curva R(α) para una imagen.
    Esta es la función de REFERENCIA de la rama main.
    """
    # Preparar imagen
    if img.ndim == 3 and img.shape[0] == 1:
        img_single = img[0]
    else:
        img_single = img
    
    img_resized = TF.resize(img_single.unsqueeze(0), [h, w])[0]
    img_channels = img_resized.repeat(ch, 1, 1).to(device)
    
    order_params = []
    
    for alpha in alphas:
        with torch.no_grad():
            c_scaled = img_channels.unsqueeze(0) * alpha
            x0 = torch.randn(1, ch, h, w, device=device)
            
            x_final, xs, es = kblock(
                x0, c_scaled, T=T, gamma=0.7, del_t=0.9,
                return_xs=True, return_es=True
            )
            
            R_t = []
            for x_t in xs[-5:]:
                x_complex = torch.view_as_complex(
                    x_t[0, 0:2].permute(1, 2, 0).contiguous()
                )
                # Fix para MPS: torch.angle no soportado en MPS
                if x_complex.device.type == 'mps':
                    x_complex_cpu = x_complex.cpu()
                    phases = torch.angle(x_complex_cpu)
                else:
                    phases = torch.angle(x_complex)
                R = np.abs(np.mean(np.exp(1j * phases.cpu().detach().numpy())))
                R_t.append(R)
            
            R = np.mean(R_t)
            order_params.append(R)
    
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return np.array(order_params)


def main(args):
    print("="*70)
    print("TEST R vs α - FUNCIÓN DE REFERENCIA (rama main)")
    print("="*70)
    print(f"\nClase: {args.clase}")
    print(f"Índice de imagen: {args.img_index}")
    print()
    
    # Configuración
    device = get_device(args.device)
    ch, n, h, w, T = 4, 4, 64, 64, args.timesteps
    
    # Cargar dataset
    print("Cargando dataset MNIST...")
    mn = MNISTLoader(root=args.data_root, batch_size=1, img_size=64)
    _, test_loader = mn.get_mnist(batch_size=1, train_split=False)
    
    # Buscar imagen específica
    print(f"Buscando imagen #{args.img_index} de clase {args.clase}...")
    target_img = None
    count = 0
    
    for img_batch, label_batch in test_loader:
        if int(label_batch[0].item()) == args.clase:
            if count == args.img_index:
                target_img = img_batch[0]
                break
            count += 1
    
    if target_img is None:
        print(f"❌ No se encontró imagen #{args.img_index} de clase {args.clase}")
        return
    
    print(f"✓ Imagen encontrada\n")
    
    # Modelo
    print(f"Inicializando modelo Kuramoto...")
    print(f"  T_steps: {T}")
    print(f"  γ: 0.7")
    print(f"  Δt: 0.9")
    print(f"  ksize: 7")
    print(f"  init_omg: 0.1")
    print(f"  canales: {ch}")
    print()
    
    kblock = KBlock(n=n, ch=ch, T=T, ksize=7, init_omg=0.1).to(device)
    
    # Array de alphas
    alphas = np.arange(args.alpha_min, args.alpha_max + args.alpha_step/2, args.alpha_step)
    print(f"Evaluando {len(alphas)} puntos de α en [{alphas[0]:.4f}, {alphas[-1]:.4f}]...")
    
    # Calcular curva usando función de REFERENCIA
    order_params = calcular_curva_imagen(
        kblock, target_img, alphas,
        ch=ch, h=h, w=w, T=T, device=device
    )
    
    # Calcular α_c
    dR = np.gradient(order_params, alphas)
    idx_crit = np.argmax(dR)
    alpha_c = alphas[idx_crit]
    R_c = order_params[idx_crit]
    dR_max = dR[idx_crit]
    
    print(f"\n{'='*70}")
    print("RESULTADOS:")
    print(f"{'='*70}")
    print(f"  α_c (punto crítico): {alpha_c:.6f}")
    print(f"  R(α_c):              {R_c:.6f}")
    print(f"  dR/dα máximo:        {dR_max:.6f}")
    print(f"{'='*70}\n")
    
    # Guardar datos
    output_data = {
        'clase': args.clase,
        'img_index': args.img_index,
        'alphas': alphas.tolist(),
        'order_params': order_params.tolist(),
        'gradient': dR.tolist(),
        'alpha_c': float(alpha_c),
        'R_c': float(R_c),
        'dR_max': float(dR_max),
        'parametros': {
            'T': T,
            'gamma': 0.7,
            'delta_t': 0.9,
            'ksize': 7,
            'init_omg': 0.1,
            'ch': ch
        }
    }
    
    output_file = f"test_referencia_clase{args.clase}_img{args.img_index}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Datos guardados: {output_file}")
    
    # Generar gráfica
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Panel 1: R(α)
    ax1.plot(alphas, order_params, 'b-', linewidth=2, label='R(α)')
    ax1.axvline(alpha_c, color='r', linestyle='--', linewidth=2, label=f'α_c = {alpha_c:.6f}')
    ax1.scatter([alpha_c], [R_c], color='r', s=100, zorder=5)
    ax1.set_xlabel('α', fontsize=12)
    ax1.set_ylabel('R (parámetro de orden)', fontsize=12)
    ax1.set_title(f'Curva R vs α - Clase {args.clase}, Imagen #{args.img_index}\n(Función de referencia - rama main)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Panel 2: Gradiente dR/dα
    ax2.plot(alphas, dR, 'g-', linewidth=2, label='dR/dα')
    ax2.axvline(alpha_c, color='r', linestyle='--', linewidth=2, label=f'α_c = {alpha_c:.6f}')
    ax2.scatter([alpha_c], [dR_max], color='r', s=100, zorder=5)
    ax2.set_xlabel('α', fontsize=12)
    ax2.set_ylabel('dR/dα', fontsize=12)
    ax2.set_title('Gradiente (máximo = punto crítico)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    plot_file = f"test_referencia_clase{args.clase}_img{args.img_index}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Gráfica guardada: {plot_file}")
    
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test R vs α para una imagen usando función de referencia',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--clase', type=int, required=True,
                       help='Clase MNIST (0-9)')
    parser.add_argument('--img-index', type=int, default=0,
                       help='Índice de imagen dentro de la clase')
    
    # Parámetros del modelo
    parser.add_argument('--timesteps', type=int, default=100,
                       help='Número de timesteps T')
    
    # Rango de alpha
    parser.add_argument('--alpha-min', type=float, default=0.0,
                       help='α mínimo')
    parser.add_argument('--alpha-max', type=float, default=2.0,
                       help='α máximo')
    parser.add_argument('--alpha-step', type=float, default=0.02,
                       help='Paso de α')
    
    # Configuración
    parser.add_argument('--data-root', default='./data',
                       help='Directorio raíz de datos')
    parser.add_argument('--device', default='auto',
                       choices=['auto', 'mps', 'cuda', 'cpu'],
                       help='Dispositivo a usar')
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
