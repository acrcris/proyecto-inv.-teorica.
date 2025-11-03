#!/usr/bin/env python3
"""
test_optimizado_rapido.py

Test rápido que solo verifica que la versión optimizada funciona correctamente.
"""
import sys
import time
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kuramoto.modelo import KBlock
from datasets.loader import MNISTLoader
import torchvision.transforms.functional as TF


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def calcular_order_parameter_gpu(xs_batch, device):
    """Versión optimizada GPU."""
    if isinstance(xs_batch, list):
        xs_batch = torch.stack(xs_batch, dim=1)
    
    if xs_batch.ndim == 4:
        xs_batch = xs_batch.unsqueeze(0)
    
    batch_size, T, ch, h, w = xs_batch.shape
    xs_tail = xs_batch[:, -5:, :, :, :]
    
    cos_component = xs_tail[:, :, 0, :, :]
    sin_component = xs_tail[:, :, 1, :, :]
    phases = torch.atan2(sin_component, cos_component)
    
    cos_phases = torch.cos(phases)
    sin_phases = torch.sin(phases)
    mean_cos = cos_phases.mean(dim=(-2, -1))
    mean_sin = sin_phases.mean(dim=(-2, -1))
    R_per_timestep = torch.sqrt(mean_cos**2 + mean_sin**2)
    R_values = R_per_timestep.mean(dim=1)
    
    return R_values


def calcular_alpha_c_optimizado(kblock, img, alphas, ch, h, w, T, device, batch_size=50):
    """Versión optimizada."""
    if img.ndim == 3 and img.shape[0] == 1:
        img_single = img[0]
    else:
        img_single = img
    
    img_resized = TF.resize(img_single.unsqueeze(0), [h, w])[0]
    img_channels = img_resized.repeat(ch, 1, 1).to(device)
    
    alphas_tensor = torch.tensor(alphas, dtype=torch.float32, device=device)
    num_alphas = len(alphas)
    order_params = []
    
    with torch.no_grad():
        for batch_start in range(0, num_alphas, batch_size):
            batch_end = min(batch_start + batch_size, num_alphas)
            batch_alphas = alphas_tensor[batch_start:batch_end]
            batch_len = len(batch_alphas)
            
            c_batch = img_channels.unsqueeze(0) * batch_alphas.view(-1, 1, 1, 1)
            x0_batch = torch.randn(batch_len, ch, h, w, device=device)
            
            _, xs_batch = kblock(
                x0_batch, c_batch, T=T, gamma=0.7, del_t=0.9, return_xs=True
            )
            
            R_batch = calcular_order_parameter_gpu(xs_batch, device)
            order_params.append(R_batch)
    
    order_curve = torch.cat(order_params, dim=0)
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    
    order_curve_np = order_curve.cpu().numpy()
    gradient = np.gradient(order_curve_np)
    idx_max = np.argmax(gradient)
    alpha_c = alphas[idx_max]
    
    return float(alpha_c)


def main():
    print("="*70)
    print("TEST RÁPIDO - VERSIÓN OPTIMIZADA GPU")
    print("="*70)
    
    device = get_device()
    print(f"\n✅ Dispositivo: {device}")
    
    # Configuración
    ch, n, h, w, T = 4, 4, 64, 64, 100
    alphas = np.arange(0.0, 0.1 + 0.0005/2, 0.0005)
    
    print(f"\nConfiguración:")
    print(f"  Canales={ch}, n={n}, tamaño={h}×{w}, timesteps={T}")
    print(f"  Alphas: {len(alphas)} puntos en [{alphas[0]}, {alphas[-1]}]")
    
    # Modelo
    print(f"\nInicializando modelo...")
    kblock = KBlock(n=n, ch=ch, T=T, ksize=7, init_omg=0.1).to(device)
    kblock.eval()
    
    # Dataset
    print("Cargando MNIST...")
    mn = MNISTLoader(root='./data', batch_size=1, img_size=64)
    train_loader, _ = mn.get_mnist(batch_size=1, train_split=True)
    
    # Probar con 5 imágenes
    n_test = 5
    test_images = []
    test_labels = []
    for i, (img, label) in enumerate(train_loader):
        if i >= n_test:
            break
        test_images.append(img[0])
        test_labels.append(int(label[0].item()))
    
    print(f"✅ {len(test_images)} imágenes de prueba")
    print(f"   Clases: {test_labels}\n")
    
    # Test
    print("="*70)
    print("PROCESAMIENTO OPTIMIZADO")
    print("="*70)
    
    tiempos = []
    alphas_c = []
    
    for idx, (img, clase) in enumerate(zip(test_images, test_labels)):
        print(f"\nImagen {idx+1}/{n_test} (clase {clase}):")
        
        start = time.time()
        alpha_c = calcular_alpha_c_optimizado(
            kblock, img, alphas, ch, h, w, T, device, batch_size=50
        )
        elapsed = time.time() - start
        
        tiempos.append(elapsed)
        alphas_c.append(alpha_c)
        
        print(f"  α_c = {alpha_c:.6f}")
        print(f"  Tiempo: {elapsed:.2f} segundos")
        print(f"  Velocidad: {3600/elapsed:.0f} imgs/hora")
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)
    
    tiempo_promedio = np.mean(tiempos)
    imgs_por_hora = 3600 / tiempo_promedio
    
    print(f"\n📊 Estadísticas:")
    print(f"  Tiempo promedio: {tiempo_promedio:.2f} ± {np.std(tiempos):.2f} segundos")
    print(f"  Velocidad: {imgs_por_hora:.0f} imágenes/hora")
    print(f"  α_c promedio: {np.mean(alphas_c):.6f}")
    print(f"  α_c rango: [{np.min(alphas_c):.6f}, {np.max(alphas_c):.6f}]")
    
    # Comparación con versión original
    print(f"\n💡 Comparación con versión original:")
    print(f"  Original: ~151 imgs/hora (23.9 sec/img)")
    print(f"  Refactorizada: ~244 imgs/hora (14.7 sec/img)")
    print(f"  Optimizada: ~{imgs_por_hora:.0f} imgs/hora ({tiempo_promedio:.1f} sec/img)")
    
    if imgs_por_hora > 244:
        mejora = imgs_por_hora / 244
        print(f"\n🚀 ACELERACIÓN: {mejora:.1f}x más rápido que la refactorizada")
        print(f"🚀 ACELERACIÓN: {imgs_por_hora/151:.1f}x más rápido que la original")
        
        # Estimación de tiempo total
        total_imgs = 60000
        tiempo_estimado_horas = total_imgs / imgs_por_hora
        print(f"\n⏱️  Tiempo estimado para 60,000 imágenes:")
        print(f"  {tiempo_estimado_horas:.1f} horas ({tiempo_estimado_horas/24:.1f} días)")
    else:
        print(f"\n⚠️  No hubo mejora significativa. Ajustar batch_size o revisar código.")
    
    print(f"\n✅ Test completado exitosamente")


if __name__ == '__main__':
    main()
