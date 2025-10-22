#!/usr/bin/env python3
"""
generar_curvas_finas_R_alpha.py

Genera curvas R(α) con alta resolución (paso 0.01) para detectar
el punto crítico α_c donde dR/dα es máximo.

Similar al código proporcionado pero aplicado a múltiples imágenes.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kuramoto.modelo import KBlock
from datasets.loader import MNISTLoader


def calcular_curva_R_alpha_fino(kblock, img, alphas, ch=4, T_steps=80, gamma=0.7, del_t=0.9):
    """
    Calcula curva R(α) con resolución fina para una imagen.
    Reproduce el método exacto del notebook.
    
    Args:
        kblock: Modelo de Kuramoto
        img: Imagen de entrada [1, H, W] o [H, W]
        alphas: Array de valores alpha
        ch: Número de canales (default: 4)
        T_steps: Número de pasos temporales
        gamma, del_t: Parámetros de la dinámica
    
    Returns:
        order_params: Array con valores de R para cada alpha
    """
    device = next(kblock.parameters()).device
    h, w = 64, 64
    
    # Preparar imagen igual que en el notebook
    if img.ndim == 2:
        img_single = img
    elif img.ndim == 3 and img.shape[0] == 1:
        img_single = img[0]
    elif img.ndim == 4:
        img_single = img[0, 0]
    else:
        img_single = img[0]
    
    # Resize a 64x64 si es necesario
    import torchvision.transforms.functional as TF
    if img_single.shape != (h, w):
        img_resized = TF.resize(img_single.unsqueeze(0), [h, w])[0]
    else:
        img_resized = img_single
    
    # Repetir canales: [1, 64, 64] -> [4, 64, 64]
    img_channels = img_resized.repeat(ch, 1, 1).to(device)
    
    order_params = []
    
    for alpha in alphas:
        with torch.no_grad():
            # Escalar c con alpha (como en el notebook)
            c_scaled = img_channels.unsqueeze(0) * alpha  # [1, 4, 64, 64]
            
            # Estado inicial aleatorio
            x0 = torch.randn(1, ch, h, w).to(device)
            
            # Ejecutar dinámica
            x_final, xs_full, es_full = kblock(
                x0,
                c_scaled,
                T=T_steps,
                gamma=gamma,
                del_t=del_t,
                return_xs=True,
                return_es=True
            )
            
            # Calcular R promediando últimos 5 pasos
            R_t = []
            for x_t in xs_full[-5:]:
                # Extraer canales 0 y 1 como parte real e imaginaria
                x_complex = torch.view_as_complex(
                    x_t[0, 0:2].permute(1, 2, 0).contiguous()
                )
                phases = torch.angle(x_complex)
                
                # Coherencia global
                R = torch.abs(torch.mean(torch.exp(1j * phases)))
                R_t.append(R.item())
            
            # Promedio de últimos pasos
            R_mean = np.mean(R_t)
            order_params.append(R_mean)
    
    return np.array(order_params)


def calcular_alpha_critico(alphas, order_params):
    """
    Calcula el punto crítico α_c donde dR/dα es máximo.
    
    Args:
        alphas: Array de valores alpha
        order_params: Array de valores R
    
    Returns:
        alpha_c: Valor crítico de alpha
        dR: Gradiente dR/dα
        idx_c: Índice del punto crítico
    """
    dR = np.gradient(order_params, alphas)
    idx_c = np.argmax(dR)
    alpha_c = alphas[idx_c]
    
    return alpha_c, dR, idx_c


def main(args):
    print(f"{'='*70}")
    print(f"GENERACIÓN DE CURVAS R(α) CON ALTA RESOLUCIÓN")
    print(f"{'='*70}")
    print(f"Rango alpha: {args.alpha_min} a {args.alpha_max}")
    print(f"Paso: {args.alpha_step}")
    print(f"Número de alphas: {int((args.alpha_max - args.alpha_min) / args.alpha_step) + 1}")
    print(f"Imágenes por clase: {args.images_per_class}")
    print()
    
    # Array de alphas con alta resolución
    alphas = np.arange(args.alpha_min, args.alpha_max + args.alpha_step/2, args.alpha_step)
    print(f"Alphas generados: {len(alphas)} valores")
    
    # Cargar dataset
    mn = MNISTLoader(root=args.data_root, batch_size=1, img_size=args.img_size)
    _, test_loader = mn.get_mnist(batch_size=1, train_split=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Usando dispositivo: {device}')
    
    # Modelo
    kblock = KBlock(
        n=args.n, 
        ch=args.ch, 
        T=args.T_model, 
        ksize=args.ksize, 
        init_omg=args.init_omg
    ).to(device)
    
    # Recolectar imágenes por clase
    images_by_class = {i: [] for i in range(10)}
    
    print("\nRecolectando imágenes por clase...")
    for img_batch, label_batch in test_loader:
        label = int(label_batch[0].item())
        if len(images_by_class[label]) < args.images_per_class:
            images_by_class[label].append(img_batch[0])
        
        if all(len(imgs) >= args.images_per_class for imgs in images_by_class.values()):
            break
    
    for cls, imgs in images_by_class.items():
        print(f"  Clase {cls}: {len(imgs)} imágenes")
    
    # Calcular curvas para todas las imágenes
    print(f"\nCalculando curvas R(α) con alta resolución...")
    
    results = {
        'alphas': alphas,
        'curvas': {},  # {clase: {img_idx: {'R': [], 'alpha_c': float, 'dR': []}}}
        'params': {
            'T_steps': args.T_steps,
            'gamma': args.gamma,
            'del_t': args.del_t,
            'alpha_min': args.alpha_min,
            'alpha_max': args.alpha_max,
            'alpha_step': args.alpha_step
        }
    }
    
    total_imgs = sum(len(imgs) for imgs in images_by_class.values())
    
    with tqdm(total=total_imgs, desc="Procesando imágenes") as pbar:
        for clase in sorted(images_by_class.keys()):
            results['curvas'][clase] = {}
            
            for img_idx, img in enumerate(images_by_class[clase]):
                # Calcular curva R(α)
                order_params = calcular_curva_R_alpha_fino(
                    kblock, img, alphas,
                    ch=args.ch,
                    T_steps=args.T_steps,
                    gamma=args.gamma,
                    del_t=args.del_t
                )
                
                # Calcular punto crítico
                alpha_c, dR, idx_c = calcular_alpha_critico(alphas, order_params)
                
                results['curvas'][clase][img_idx] = {
                    'R': order_params,
                    'alpha_c': alpha_c,
                    'dR': dR,
                    'idx_c': idx_c,
                    'R_at_alpha_c': order_params[idx_c],
                    'dR_max': dR[idx_c]
                }
                
                pbar.update(1)
                pbar.set_postfix({
                    'Clase': clase, 
                    'Img': img_idx,
                    'α_c': f'{alpha_c:.3f}'
                })
    
    # Guardar resultados
    output_path = args.output
    torch.save(results, output_path)
    print(f"\n✅ Datos guardados en: {output_path}")
    print(f"   Estructura: {len(images_by_class)} clases × {args.images_per_class} imgs × {len(alphas)} alphas")
    
    # Resumen de puntos críticos
    print(f"\n{'='*70}")
    print("RESUMEN DE PUNTOS CRÍTICOS α_c")
    print(f"{'='*70}")
    
    for clase in sorted(results['curvas'].keys()):
        alphas_c = [results['curvas'][clase][i]['alpha_c'] 
                   for i in range(args.images_per_class)]
        
        print(f"\nClase {clase}:")
        print(f"  α_c medio: {np.mean(alphas_c):.4f} ± {np.std(alphas_c):.4f}")
        print(f"  α_c min:   {np.min(alphas_c):.4f}")
        print(f"  α_c max:   {np.max(alphas_c):.4f}")
        print(f"  Valores individuales: {[f'{a:.3f}' for a in alphas_c]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Genera curvas R(α) con alta resolución para detectar α_c'
    )
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--output', default='curvas_finas_R_alpha.pt', 
                       help='Archivo de salida')
    parser.add_argument('--alpha_min', type=float, default=0.0)
    parser.add_argument('--alpha_max', type=float, default=2.0)
    parser.add_argument('--alpha_step', type=float, default=0.01,
                       help='Paso de alpha (default: 0.01 para alta resolución)')
    parser.add_argument('--T_steps', type=int, default=80)
    parser.add_argument('--images_per_class', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--del_t', type=float, default=0.9)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--n', type=int, default=4)
    parser.add_argument('--ch', type=int, default=4)
    parser.add_argument('--T_model', type=int, default=4)
    parser.add_argument('--ksize', type=int, default=7)
    parser.add_argument('--init_omg', type=float, default=0.1)
    
    args = parser.parse_args()
    main(args)
