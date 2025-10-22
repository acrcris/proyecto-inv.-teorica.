#!/usr/bin/env python3
"""
analizar_distribuciones_alpha_c_gpu.py

Script optimizado para calcular distribuciones de α_c en GPU (MPS/CUDA).
Soporta datasets grandes de MNIST con procesamiento eficiente.

Uso:
    python analizar_distribuciones_alpha_c_gpu.py --images_per_class 100 --device mps
"""
import os
import sys
import torch
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
import torchvision.transforms.functional as TF

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
        print(f"✅ Usando dispositivo especificado: {device_arg}")
    
    return device


def calcular_curva_imagen(kblock, img, alphas, ch=4, h=64, w=64, T=100, device='cpu'):
    """
    Calcula curva R(α) para una imagen con soporte GPU.
    
    Args:
        kblock: Modelo Kuramoto en GPU
        img: Imagen de entrada
        alphas: Array de valores alpha
        ch, h, w: Parámetros de dimensión
        T: Pasos temporales
        device: Dispositivo (mps/cuda/cpu)
    
    Returns:
        order_params: Array con valores de R para cada alpha
    """
    # Preparar imagen
    if img.ndim == 3 and img.shape[0] == 1:
        img_single = img[0]
    else:
        img_single = img
    
    # Resize a 64x64
    img_resized = TF.resize(img_single.unsqueeze(0), [h, w])[0]
    
    # Repetir canales y mover a GPU
    img_channels = img_resized.repeat(ch, 1, 1).to(device)
    
    order_params = []
    
    for alpha in alphas:
        with torch.no_grad():
            # Escalar c con alpha
            c_scaled = img_channels.unsqueeze(0) * alpha
            
            # Estado inicial aleatorio en GPU
            x0 = torch.randn(1, ch, h, w, device=device)
            
            # Ejecutar dinámica
            x_final, xs, es = kblock(
                x0, c_scaled,
                T=T,
                gamma=0.7,
                del_t=0.9,
                return_xs=True,
                return_es=True
            )
            
            # Calcular R promediando últimos 5 pasos
            R_t = []
            for x_t in xs[-5:]:
                # Tomamos los dos primeros canales como parte real e imaginaria
                x_complex = torch.view_as_complex(
                    x_t[0, 0:2].permute(1, 2, 0).contiguous()
                )
                phases = torch.angle(x_complex)
                
                # Calcular coherencia global (mover a CPU solo el resultado)
                R = np.abs(np.mean(np.exp(1j * phases.cpu().detach().numpy())))
                R_t.append(R)
            
            R = np.mean(R_t)
            order_params.append(R)
    
    # Limpiar memoria GPU periódicamente
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return np.array(order_params)


def main(args):
    start_time = datetime.now()
    
    print("="*70)
    print("ANÁLISIS DE DISTRIBUCIONES DE α_c CON GPU")
    print("="*70)
    print(f"Inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Detectar dispositivo
    device = get_device(args.device)
    
    # Parámetros
    ch = 4
    n = 4
    h, w = 64, 64
    T = args.T
    n_imgs_per_class = args.images_per_class
    
    print(f"\nParámetros:")
    print(f"  Imágenes por clase: {n_imgs_per_class}")
    print(f"  Total de imágenes: {n_imgs_per_class * 10}")
    print(f"  T (pasos temporales): {T}")
    print(f"  gamma: {args.gamma}")
    print(f"  del_t: {args.del_t}")
    print(f"  ksize: {args.ksize}")
    print(f"  init_omg: {args.init_omg}")
    
    # Cargar dataset
    print(f"\nCargando dataset MNIST desde {args.data_root}...")
    mn = MNISTLoader(root=args.data_root, batch_size=1, img_size=64)
    _, test_loader = mn.get_mnist(batch_size=1, train_split=False)
    
    # Modelo en GPU
    print(f"Inicializando modelo Kuramoto en {device}...")
    kblock = KBlock(
        n=n, ch=ch, T=4, 
        ksize=args.ksize, 
        init_omg=args.init_omg
    ).to(device)
    
    # Recolectar imágenes
    images_by_class = {i: [] for i in range(10)}
    
    print(f"\nRecolectando {n_imgs_per_class} imágenes por clase...")
    for img_batch, label_batch in test_loader:
        label = int(label_batch[0].item())
        if len(images_by_class[label]) < n_imgs_per_class:
            images_by_class[label].append(img_batch[0])
        
        if all(len(imgs) >= n_imgs_per_class for imgs in images_by_class.values()):
            break
    
    total_collected = sum(len(imgs) for imgs in images_by_class.values())
    print(f"✅ {total_collected} imágenes recolectadas")
    
    for clase, imgs in images_by_class.items():
        print(f"   Clase {clase}: {len(imgs)} imágenes")
    
    # Array de alphas
    alphas = np.arange(args.alpha_min, args.alpha_max + args.alpha_step/2, args.alpha_step)
    print(f"\nRango de α:")
    print(f"  Valores: {len(alphas)} puntos")
    print(f"  Rango: [{alphas[0]:.6f}, {alphas[-1]:.6f}]")
    print(f"  Paso: {alphas[1] - alphas[0]:.8f}")
    
    # Calcular curvas
    results = {
        'alphas': alphas,
        'clases': {},
        'params': {
            'T': T,
            'gamma': args.gamma,
            'del_t': args.del_t,
            'ksize': args.ksize,
            'init_omg': args.init_omg,
            'n_imgs_per_class': n_imgs_per_class,
            'alpha_min': args.alpha_min,
            'alpha_max': args.alpha_max,
            'alpha_step': args.alpha_step
        },
        'metadata': {
            'device': str(device),
            'timestamp': start_time.isoformat(),
            'pytorch_version': torch.__version__
        }
    }
    
    print(f"\n{'='*70}")
    print("CALCULANDO CURVAS R(α) Y PUNTOS CRÍTICOS α_c")
    print(f"{'='*70}")
    print(f"Total de simulaciones: {total_collected * len(alphas):,}")
    print()
    
    total_imgs = total_collected
    with tqdm(total=total_imgs, desc="Progreso total") as pbar_total:
        for clase in sorted(images_by_class.keys()):
            imgs = images_by_class[clase]
            results['clases'][clase] = {'imgs': []}
            
            for img_idx, img in enumerate(imgs):
                # Calcular curva
                order_params = calcular_curva_imagen(
                    kblock, img, alphas,
                    ch=ch, h=h, w=w, T=T, device=device
                )
                
                # Calcular α_c
                dR = np.gradient(order_params, alphas)
                idx_c = np.argmax(dR)
                alpha_c = alphas[idx_c]
                
                results['clases'][clase]['imgs'].append({
                    'img_idx': img_idx,
                    'R': order_params,
                    'dR': dR,
                    'alpha_c': alpha_c,
                    'R_at_alpha_c': order_params[idx_c],
                    'dR_max': np.max(dR),
                    'R_inicial': order_params[0],
                    'R_final': order_params[-1]
                })
                
                pbar_total.update(1)
                pbar_total.set_postfix({
                    'Clase': clase,
                    'Img': f'{img_idx+1}/{len(imgs)}',
                    'α_c': f'{alpha_c:.5f}'
                })
    
    # Calcular estadísticas por clase
    print(f"\n{'='*70}")
    print("CALCULANDO ESTADÍSTICAS POR CLASE")
    print(f"{'='*70}")
    
    for clase in results['clases'].keys():
        alphas_c = [img_data['alpha_c'] for img_data in results['clases'][clase]['imgs']]
        R_iniciales = [img_data['R_inicial'] for img_data in results['clases'][clase]['imgs']]
        R_finales = [img_data['R_final'] for img_data in results['clases'][clase]['imgs']]
        
        results['clases'][clase]['estadisticas'] = {
            'alpha_c_mean': np.mean(alphas_c),
            'alpha_c_std': np.std(alphas_c),
            'alpha_c_median': np.median(alphas_c),
            'alpha_c_min': np.min(alphas_c),
            'alpha_c_max': np.max(alphas_c),
            'R_inicial_mean': np.mean(R_iniciales),
            'R_final_mean': np.mean(R_finales),
            'R_final_std': np.std(R_finales)
        }
        
        stats = results['clases'][clase]['estadisticas']
        print(f"\nClase {clase}:")
        print(f"  α_c: {stats['alpha_c_mean']:.6f} ± {stats['alpha_c_std']:.6f}")
        print(f"  Rango α_c: [{stats['alpha_c_min']:.6f}, {stats['alpha_c_max']:.6f}]")
        print(f"  R(α=0): {stats['R_inicial_mean']:.4f}")
        print(f"  R(α={args.alpha_max}): {stats['R_final_mean']:.4f} ± {stats['R_final_std']:.4f}")
    
    # Guardar resultados
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    results['metadata']['end_timestamp'] = end_time.isoformat()
    results['metadata']['duration_seconds'] = duration
    
    output_path = args.output
    torch.save(results, output_path)
    
    print(f"\n{'='*70}")
    print("ANÁLISIS COMPLETADO")
    print(f"{'='*70}")
    print(f"Archivo guardado: {output_path}")
    print(f"Tamaño: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print(f"Duración total: {duration / 60:.1f} minutos ({duration:.1f} segundos)")
    print(f"Tiempo por imagen: {duration / total_imgs:.2f} segundos")
    print(f"Tiempo por simulación: {duration / (total_imgs * len(alphas)):.3f} segundos")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Análisis de distribuciones de α_c optimizado para GPU',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Parámetros principales
    parser.add_argument('--images_per_class', type=int, default=10,
                       help='Número de imágenes por clase (0-9)')
    parser.add_argument('--output', default='distribuciones_alpha_c.pt',
                       help='Archivo de salida (.pt)')
    
    # Rango de alpha
    parser.add_argument('--alpha_min', type=float, default=0.0,
                       help='Valor mínimo de α')
    parser.add_argument('--alpha_max', type=float, default=0.1,
                       help='Valor máximo de α')
    parser.add_argument('--alpha_step', type=float, default=0.0005,
                       help='Paso de α (menor = más resolución)')
    
    # Parámetros del modelo
    parser.add_argument('--T', type=int, default=100,
                       help='Pasos temporales')
    parser.add_argument('--gamma', type=float, default=0.7,
                       help='Fuerza de acoplamiento')
    parser.add_argument('--del_t', type=float, default=0.9,
                       help='Paso de integración')
    parser.add_argument('--ksize', type=int, default=7,
                       help='Tamaño de kernel convolucional')
    parser.add_argument('--init_omg', type=float, default=0.1,
                       help='Escala de frecuencias naturales')
    
    # Configuración
    parser.add_argument('--data_root', default='./data',
                       help='Ruta del dataset MNIST')
    parser.add_argument('--device', default='auto',
                       choices=['auto', 'mps', 'cuda', 'cpu'],
                       help='Dispositivo a usar (auto detecta el mejor)')
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
