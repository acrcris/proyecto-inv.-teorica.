#!/usr/bin/env python3
"""
validar_equivalencia.py

Valida que la versión optimizada produce EXACTAMENTE los mismos valores de alpha_c
que la versión refactorizada actual.
"""
import sys
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


# ============================================================================
# VERSIÓN REFACTORIZADA (ACTUAL)
# ============================================================================

def calcular_order_parameter_refactorizado(xs):
    """Versión actual (refactorizada) - secuencial con NumPy."""
    order_params = []
    for x_t in xs[-5:]:
        x_complex = x_t[0] + 1j * x_t[1]
        phases = torch.angle(x_complex.cpu())
        phases_np = phases.numpy()
        R = np.abs(np.exp(1j * phases_np).mean())
        order_params.append(R)
    return np.mean(order_params)


def calcular_alpha_c_refactorizado(kblock, img, alphas, ch, h, w, T, device):
    """Versión refactorizada (actual) - la que está corriendo ahora."""
    if img.ndim == 3 and img.shape[0] == 1:
        img_single = img[0]
    else:
        img_single = img
    
    img_resized = TF.resize(img_single.unsqueeze(0), [h, w])[0]
    img_channels = img_resized.repeat(ch, 1, 1).to(device)
    
    order_curve = []
    
    with torch.no_grad():
        for alpha in alphas:
            c_scaled = img_channels * alpha
            x0 = torch.randn(ch, h, w, device=device)
            
            _, xs = kblock(
                x0.unsqueeze(0), 
                c_scaled.unsqueeze(0), 
                T=T,
                gamma=0.7,
                del_t=0.9,
                return_xs=True,
            )
            
            R = calcular_order_parameter_refactorizado(xs[0])
            order_curve.append(R)
    
    order_curve = np.array(order_curve)
    gradient = np.gradient(order_curve)
    idx_max = np.argmax(gradient)
    alpha_c = alphas[idx_max]
    
    return float(alpha_c), order_curve


# ============================================================================
# VERSIÓN OPTIMIZADA (NUEVA)
# ============================================================================

def calcular_order_parameter_optimizado(xs_batch, device):
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
    """Versión optimizada - la nueva."""
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
            
            R_batch = calcular_order_parameter_optimizado(xs_batch, device)
            order_params.append(R_batch)
    
    order_curve_tensor = torch.cat(order_params, dim=0)
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    
    order_curve = order_curve_tensor.cpu().numpy()
    gradient = np.gradient(order_curve)
    idx_max = np.argmax(gradient)
    alpha_c = alphas[idx_max]
    
    return float(alpha_c), order_curve


# ============================================================================
# TEST DE EQUIVALENCIA
# ============================================================================

def main():
    print("="*80)
    print("VALIDACIÓN DE EQUIVALENCIA: Refactorizada vs Optimizada")
    print("="*80)
    print()
    
    device = get_device()
    print(f"✅ Dispositivo: {device}\n")
    
    # Configuración IDÉNTICA a la usada en producción
    ch, n, h, w, T = 4, 4, 64, 64, 100
    alphas = np.arange(0.0, 0.1 + 0.0005/2, 0.0005)
    
    print(f"Configuración:")
    print(f"  Canales={ch}, n={n}, tamaño={h}×{w}, timesteps={T}")
    print(f"  Alphas: {len(alphas)} puntos en [{alphas[0]}, {alphas[-1]}]")
    print()
    
    # Cargar modelo
    print("Inicializando modelo...")
    kblock = KBlock(n=n, ch=ch, T=T, ksize=7, init_omg=0.1).to(device)
    kblock.eval()
    
    # Cargar dataset
    print("Cargando MNIST...")
    mn = MNISTLoader(root='./data', batch_size=1, img_size=64)
    train_loader, _ = mn.get_mnist(batch_size=1, train_split=True)
    
    # Tomar 10 imágenes de prueba
    n_test = 10
    test_images = []
    test_labels = []
    for i, (img, label) in enumerate(train_loader):
        if i >= n_test:
            break
        test_images.append(img[0])
        test_labels.append(int(label[0].item()))
    
    print(f"✅ {len(test_images)} imágenes de prueba")
    print(f"   Clases: {test_labels}\n")
    
    # Comparar versiones
    print("="*80)
    print("COMPARACIÓN DETALLADA")
    print("="*80)
    
    diferencias = []
    max_diff_curva = []
    
    for idx, (img, clase) in enumerate(zip(test_images, test_labels)):
        print(f"\n{'='*80}")
        print(f"Imagen {idx+1}/{n_test} (clase {clase})")
        print(f"{'='*80}")
        
        # IMPORTANTE: Usar la MISMA semilla para ambas versiones
        torch.manual_seed(42 + idx)
        alpha_c_ref, curve_ref = calcular_alpha_c_refactorizado(
            kblock, img, alphas, ch, h, w, T, device
        )
        
        torch.manual_seed(42 + idx)
        alpha_c_opt, curve_opt = calcular_alpha_c_optimizado(
            kblock, img, alphas, ch, h, w, T, device, batch_size=50
        )
        
        # Análisis
        diff_alpha = abs(alpha_c_ref - alpha_c_opt)
        diff_alpha_pct = 100 * diff_alpha / alpha_c_ref if alpha_c_ref != 0 else 0
        
        diff_curva = np.abs(curve_ref - curve_opt)
        max_diff = np.max(diff_curva)
        mean_diff = np.mean(diff_curva)
        
        print(f"  Refactorizada: α_c = {alpha_c_ref:.8f}")
        print(f"  Optimizada:    α_c = {alpha_c_opt:.8f}")
        print(f"  Diferencia:    Δα_c = {diff_alpha:.8f} ({diff_alpha_pct:.4f}%)")
        print(f"\n  Curva de orden:")
        print(f"    Diferencia máxima:   {max_diff:.8f}")
        print(f"    Diferencia promedio: {mean_diff:.8f}")
        
        # Validación
        if diff_alpha < 1e-6:
            print(f"\n  ✅ IDÉNTICO: α_c exactamente igual")
        elif diff_alpha_pct < 0.01:
            print(f"\n  ✅ EQUIVALENTE: Diferencia < 0.01%")
        elif diff_alpha_pct < 0.1:
            print(f"\n  ⚠️  ACEPTABLE: Diferencia < 0.1%")
        else:
            print(f"\n  ❌ DIFERENTE: Diferencia significativa")
        
        diferencias.append(diff_alpha)
        max_diff_curva.append(max_diff)
    
    # Resumen final
    print("\n" + "="*80)
    print("RESUMEN DE EQUIVALENCIA")
    print("="*80)
    
    diff_promedio = np.mean(diferencias)
    diff_max = np.max(diferencias)
    diff_std = np.std(diferencias)
    
    print(f"\nDiferencias en α_c:")
    print(f"  Promedio: {diff_promedio:.8f}")
    print(f"  Máxima:   {diff_max:.8f}")
    print(f"  Std Dev:  {diff_std:.8f}")
    
    print(f"\nDiferencias en curvas de orden:")
    print(f"  Max diff promedio: {np.mean(max_diff_curva):.8f}")
    print(f"  Max diff máxima:   {np.max(max_diff_curva):.8f}")
    
    # Veredicto final
    print(f"\n{'='*80}")
    print("VEREDICTO")
    print(f"{'='*80}\n")
    
    imagenes_identicas = sum(1 for d in diferencias if d < 1e-6)
    imagenes_equivalentes = sum(1 for d in diferencias if d < 1e-4)
    
    print(f"De {n_test} imágenes:")
    print(f"  {imagenes_identicas} ({100*imagenes_identicas/n_test:.0f}%) son IDÉNTICAS (diff < 1e-6)")
    print(f"  {imagenes_equivalentes} ({100*imagenes_equivalentes/n_test:.0f}%) son EQUIVALENTES (diff < 1e-4)")
    
    if imagenes_equivalentes == n_test:
        print(f"\n✅ VALIDACIÓN EXITOSA")
        print(f"   La versión optimizada produce resultados equivalentes.")
        print(f"   Es SEGURO reemplazar el proceso actual.")
        print(f"\n💡 Nota sobre las diferencias:")
        print(f"   Las pequeñas diferencias se deben a:")
        print(f"   1. Orden de operaciones en punto flotante")
        print(f"   2. torch.atan2 vs torch.angle (implementación)")
        print(f"   3. Redondeo acumulado en sumas GPU vs CPU")
        print(f"   Estas diferencias son DESPRECIABLES para el análisis.")
        return 0
    else:
        print(f"\n⚠️  ADVERTENCIA")
        print(f"   Hay diferencias significativas en algunas imágenes.")
        print(f"   Revisar antes de reemplazar el proceso.")
        return 1


if __name__ == '__main__':
    exit(main())
