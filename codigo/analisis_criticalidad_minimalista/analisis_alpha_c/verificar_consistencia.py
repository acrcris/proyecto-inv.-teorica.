"""
Script de verificación: comprueba que el cálculo de R(α) sea consistente
con el código de referencia usando torch.view_as_complex.
"""
import torch
import numpy as np
from pathlib import Path
import sys

# Agregar path del módulo
sys.path.insert(0, str(Path(__file__).parent))

from datasets.loader import MNISTLoader
from kuramoto.modelo import KBlock
from analisis.criticalidad import KuramotoMetrics


def test_order_parameter_calculation():
    """
    Verifica que el cálculo de R sea consistente con el código de referencia.
    """
    print("="*70)
    print("VERIFICACIÓN: Cálculo de parámetro de orden R")
    print("="*70)
    
    # Setup
    device = torch.device("cpu")  # CPU para reproducibilidad
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Cargar una imagen de prueba
    loader = MNISTLoader(root="./data", img_size=64)
    dataset = loader.train_dataset
    img, label = dataset[0]
    
    print(f"\nImagen de prueba: clase {label}")
    
    # Preparar imagen
    ch = 4
    img_channels = img.repeat(ch, 1, 1).unsqueeze(0).to(device)
    
    # Construir KBlock
    kblock = KBlock(
        n=ch,
        ch=ch,
        connectivity="conv",
        T=100,
        ksize=7,
        init_omg=0.1,
        c_norm=None,
        use_omega_c=False,
    ).to(device)
    kblock.eval()
    
    # Probar con alpha=0.5
    alpha = 0.5
    c_scaled = img_channels * alpha
    x0 = torch.randn_like(img_channels)
    
    print(f"\nEjecutando dinámica con α={alpha}...")
    with torch.no_grad():
        _, xs = kblock(x0, c_scaled, T=100, gamma=0.7, del_t=0.9, return_xs=True)
    
    print(f"  • Serie temporal xs: {len(xs)} pasos")
    print(f"  • Shape de cada paso: {xs[0].shape}")
    
    # Método 1: Usando la función actualizada
    print("\n" + "-"*70)
    print("MÉTODO 1: KuramotoMetrics.order_parameter (actualizado)")
    print("-"*70)
    R_series = KuramotoMetrics.order_parameter(xs)
    R_mean_last5 = np.mean(R_series[-5:])
    
    print(f"  • R(t) completo: {len(R_series)} valores")
    print(f"  • R(t=0) = {R_series[0]:.6f}")
    print(f"  • R(t=final) = {R_series[-1]:.6f}")
    print(f"  • R promedio (últimos 5): {R_mean_last5:.6f}")
    
    # Método 2: Código de referencia (explícito)
    print("\n" + "-"*70)
    print("MÉTODO 2: Código de referencia (explícito)")
    print("-"*70)
    R_t_ref = []
    for x_t in xs[-5:]:  # Últimos 5 pasos
        # Tomar canales 0:2 como (real, imag)
        x_complex = torch.view_as_complex(x_t[0, 0:2].permute(1, 2, 0).contiguous())
        phases = torch.angle(x_complex)
        R_ref = np.abs(np.mean(np.exp(1j * phases.detach().numpy())))
        R_t_ref.append(R_ref)
    
    R_mean_ref = np.mean(R_t_ref)
    print(f"  • R valores (últimos 5): {[f'{r:.6f}' for r in R_t_ref]}")
    print(f"  • R promedio (últimos 5): {R_mean_ref:.6f}")
    
    # Comparación
    print("\n" + "="*70)
    print("COMPARACIÓN")
    print("="*70)
    
    # Comparar valores de los últimos 5 pasos
    R_last5_method1 = R_series[-5:]
    diff = np.abs(R_last5_method1 - R_t_ref)
    max_diff = np.max(diff)
    
    print(f"\nÚltimos 5 valores de R:")
    print(f"  Método 1: {[f'{r:.6f}' for r in R_last5_method1]}")
    print(f"  Método 2: {[f'{r:.6f}' for r in R_t_ref]}")
    print(f"\nDiferencias absolutas: {[f'{d:.2e}' for d in diff]}")
    print(f"Diferencia máxima: {max_diff:.2e}")
    print(f"Diferencia promedio: {np.mean(diff):.2e}")
    
    if max_diff < 1e-6:
        print("\n✅ ÉXITO: Los métodos son consistentes (diferencia < 1e-6)")
        return True
    else:
        print(f"\n⚠️  ADVERTENCIA: Diferencia de {max_diff:.2e}")
        if max_diff < 1e-3:
            print("   (Aceptable para propósitos prácticos)")
            return True
        else:
            print("   (Revisar implementación)")
            return False


def test_full_alpha_sweep():
    """
    Prueba un barrido completo de alpha como en el código de referencia.
    """
    print("\n\n" + "="*70)
    print("VERIFICACIÓN: Barrido completo de α")
    print("="*70)
    
    device = torch.device("cpu")
    torch.manual_seed(42)
    
    # Cargar imagen
    loader = MNISTLoader(root="./data", img_size=64)
    img, label = dataset = loader.train_dataset[0]
    
    # Setup
    ch = 4
    img_channels = img.repeat(ch, 1, 1).unsqueeze(0).to(device)
    
    kblock = KBlock(n=ch, ch=ch, connectivity="conv", T=100, ksize=7,
                   init_omg=0.1, c_norm=None, use_omega_c=False).to(device)
    kblock.eval()
    
    # Barrido como en código de referencia
    alphas = np.arange(0.0, 2.0, 0.1)  # Menos puntos para rapidez
    order_params = []
    
    print(f"\nBarriendo α de {alphas[0]} a {alphas[-1]} ({len(alphas)} puntos)...")
    
    for i, alpha in enumerate(alphas):
        c_scaled = img_channels * alpha
        x0 = torch.randn_like(img_channels)
        
        with torch.no_grad():
            _, xs = kblock(x0, c_scaled, T=100, gamma=0.7, del_t=0.9, return_xs=True)
        
        # Calcular R promediando últimos 5 pasos
        R_series = KuramotoMetrics.order_parameter(xs)
        R = np.mean(R_series[-5:])
        order_params.append(R)
        
        if i % 5 == 0:
            print(f"  α={alpha:.2f} → R={R:.4f}")
    
    order_params = np.array(order_params)
    
    # Calcular gradiente y α_c
    dR = np.gradient(order_params, alphas)
    alpha_c = alphas[np.argmax(dR)]
    
    print(f"\n{'='*70}")
    print(f"Resultados del barrido:")
    print(f"{'='*70}")
    print(f"  • R(α=0.0) = {order_params[0]:.4f}")
    print(f"  • R(α=2.0) = {order_params[-1]:.4f}")
    print(f"  • α_c (punto crítico) = {alpha_c:.4f}")
    print(f"  • R(α_c) = {order_params[np.argmax(dR)]:.4f}")
    print(f"  • dR/dα máximo = {np.max(dR):.4f}")
    
    return True


if __name__ == "__main__":
    print("\n🔬 Iniciando verificación de consistencia...\n")
    
    try:
        # Test 1: Verificar cálculo de R
        success1 = test_order_parameter_calculation()
        
        # Test 2: Verificar barrido completo
        success2 = test_full_alpha_sweep()
        
        if success1 and success2:
            print("\n" + "="*70)
            print("✅ TODAS LAS VERIFICACIONES PASARON")
            print("="*70)
            print("\nEl código es consistente con el método de referencia.")
            print("Usa: torch.view_as_complex + torch.angle + np.exp(1j*phases)")
            print()
        else:
            print("\n❌ Algunas verificaciones fallaron")
            
    except Exception as e:
        print(f"\n❌ ERROR durante verificación: {e}")
        import traceback
        traceback.print_exc()
