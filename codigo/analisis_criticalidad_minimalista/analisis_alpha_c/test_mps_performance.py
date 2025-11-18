#!/usr/bin/env python3
"""
Script de prueba para verificar el rendimiento de MPS (Metal) en Mac.

Compara la velocidad de:
- CPU
- MPS (Metal Performance Shaders - GPU de Apple)

Y determina cuál es más rápido para este tipo de cálculos.
"""

import time
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np

# Añadir path del módulo kuramoto
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'akorn'))

from source.models.kuramoto import KBlock


def benchmark_device(device_name, n_iterations=10):
    """
    Realiza un benchmark en el dispositivo especificado.
    
    Args:
        device_name: 'cpu' o 'mps'
        n_iterations: Número de iteraciones para promediar
    
    Returns:
        float: Tiempo promedio por iteración en segundos
    """
    device = torch.device(device_name)
    
    # Parámetros del modelo
    PARAMS = {
        'ch': 3,
        'n': 3,
        'h': 64,
        'w': 64,
        'T': 30,
        'gamma': 0.7,
        'del_t': 0.9,
        'ksize': 3,
        'init_omg': 0.1,
    }
    
    # Crear modelo
    torch.manual_seed(1)
    kblock = KBlock(
        n=PARAMS['n'], 
        ch=PARAMS['ch'], 
        connectivity='conv', 
        T=PARAMS['T'], 
        ksize=PARAMS['ksize'],
        init_omg=PARAMS['init_omg'], 
        c_norm=None, 
        use_omega_c=False
    ).to(device)
    
    # Estado inicial y acoplamiento
    x_init = torch.randn(1, PARAMS['ch'], PARAMS['h'], PARAMS['w']).to(device)
    c = torch.randn(1, PARAMS['ch'], PARAMS['h'], PARAMS['w']).to(device) * 0.5
    
    # Warm-up (importante para MPS)
    if device_name == 'mps':
        print(f"  Calentando GPU Metal...")
        for _ in range(3):
            with torch.no_grad():
                _ = kblock(x_init, c, T=PARAMS['T'], 
                          gamma=PARAMS['gamma'], del_t=PARAMS['del_t'])
        torch.mps.synchronize()
    
    # Benchmark
    times = []
    
    for i in range(n_iterations):
        x = x_init.clone()
        
        start = time.time()
        
        with torch.no_grad():
            _, xs, es = kblock(x, c, T=PARAMS['T'], 
                              gamma=PARAMS['gamma'], del_t=PARAMS['del_t'],
                              return_xs=True, return_es=True)
        
        # Sincronizar para MPS
        if device_name == 'mps':
            torch.mps.synchronize()
        
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"  Iteración {i+1}/{n_iterations}: {elapsed:.4f}s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time


def main():
    print("="*70)
    print("TEST DE RENDIMIENTO: CPU vs MPS (Metal)")
    print("="*70)
    print(f"\nPyTorch version: {torch.__version__}")
    
    # Verificar disponibilidad de MPS
    mps_available = torch.backends.mps.is_available()
    print(f"MPS (Metal) disponible: {mps_available}")
    
    if not mps_available:
        print("\n❌ MPS no está disponible en este sistema")
        print("   Este script solo funciona en Mac con Apple Silicon (M1/M2/M3)")
        print("   y PyTorch >= 1.12 con soporte MPS")
        return
    
    print(f"\n✅ Este Mac tiene GPU Metal (Apple Silicon)")
    print(f"   Puedes usar aceleración GPU para cálculos de Kuramoto\n")
    
    # Benchmark CPU
    print("="*70)
    print("BENCHMARK EN CPU")
    print("="*70)
    
    try:
        cpu_time, cpu_std = benchmark_device('cpu', n_iterations=5)
        print(f"\n📊 Resultado CPU:")
        print(f"   Tiempo promedio: {cpu_time:.4f} ± {cpu_std:.4f} segundos")
    except Exception as e:
        print(f"\n❌ Error en CPU: {e}")
        cpu_time = None
    
    # Benchmark MPS
    print("\n" + "="*70)
    print("BENCHMARK EN MPS (Metal GPU)")
    print("="*70)
    
    try:
        mps_time, mps_std = benchmark_device('mps', n_iterations=5)
        print(f"\n📊 Resultado MPS:")
        print(f"   Tiempo promedio: {mps_time:.4f} ± {mps_std:.4f} segundos")
    except Exception as e:
        print(f"\n❌ Error en MPS: {e}")
        print(f"   Esto puede ocurrir si:")
        print(f"   1. PyTorch no tiene soporte MPS completo en tu versión")
        print(f"   2. Hay operaciones no soportadas en Metal")
        print(f"   3. Problemas de memoria GPU")
        mps_time = None
    
    # Comparación
    print("\n" + "="*70)
    print("COMPARACIÓN Y RECOMENDACIÓN")
    print("="*70)
    
    if cpu_time and mps_time:
        speedup = cpu_time / mps_time
        
        print(f"\n⚡ RESULTADOS:")
        print(f"   CPU:  {cpu_time:.4f}s")
        print(f"   MPS:  {mps_time:.4f}s")
        print(f"   Speedup: {speedup:.2f}x")
        
        if speedup > 1.2:
            print(f"\n✅ RECOMENDACIÓN: Usa MPS (--device mps)")
            print(f"   MPS es {speedup:.2f}x más rápido que CPU")
            print(f"   Comando sugerido:")
            print(f"   python calcular_c_critico_local.py --clases 0 --device mps")
        elif speedup < 0.8:
            print(f"\n⚠️  RECOMENDACIÓN: Usa CPU (--device cpu)")
            print(f"   CPU es {1/speedup:.2f}x más rápido que MPS")
            print(f"   Esto puede ocurrir si el modelo es pequeño y el overhead de GPU")
            print(f"   supera los beneficios de paralelización")
            print(f"   Comando sugerido:")
            print(f"   python calcular_c_critico_local.py --clases 0 --device cpu")
        else:
            print(f"\n➖ RECOMENDACIÓN: Rendimiento similar")
            print(f"   Usa detección automática (--device auto)")
            print(f"   Comando sugerido:")
            print(f"   python calcular_c_critico_local.py --clases 0")
    elif cpu_time:
        print(f"\n⚠️  Solo CPU funcionó correctamente")
        print(f"   Usa: --device cpu")
    elif mps_time:
        print(f"\n✅ Solo MPS funcionó correctamente")
        print(f"   Usa: --device mps")
    else:
        print(f"\n❌ Ningún dispositivo funcionó correctamente")
        print(f"   Revisa la instalación de PyTorch")
    
    print("\n" + "="*70)
    print("INFORMACIÓN ADICIONAL")
    print("="*70)
    print(f"\n💡 Tips para mejor rendimiento:")
    print(f"   1. Actualiza PyTorch a la última versión: pip install --upgrade torch")
    print(f"   2. Procesa múltiples clases en paralelo si tienes RAM suficiente")
    print(f"   3. Usa --limite para pruebas rápidas antes de procesar todo")
    print(f"   4. El script tiene auto-restart: puedes detenerlo y continuar después")
    
    print(f"\n📊 Estimación de tiempo para MNIST completo (60,000 imágenes):")
    if cpu_time:
        total_time_cpu = cpu_time * 200 * 60000 / 3600  # 200 valores de C por imagen
        print(f"   CPU:  ~{total_time_cpu:.1f} horas")
    if mps_time:
        total_time_mps = mps_time * 200 * 60000 / 3600
        print(f"   MPS:  ~{total_time_mps:.1f} horas")
    
    print("\n✅ Test completado\n")


if __name__ == '__main__':
    main()
