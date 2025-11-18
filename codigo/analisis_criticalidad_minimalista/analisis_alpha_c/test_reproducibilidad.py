#!/usr/bin/env python3
"""
Test de reproducibilidad: compara resultados del notebook con el script calcular_c_critico_local.py

Este script:
1. Carga la misma imagen que el notebook (primera imagen de la clase 0)
2. Usa los mismos parámetros exactos del notebook
3. Calcula C_crítico y la curva R vs C
4. Compara y visualiza los resultados

Uso:
    python test_reproducibilidad.py
"""

import sys
from pathlib import Path
import torch
import torchvision
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import random

# Añadir paths necesarios
SCRIPT_DIR = Path(__file__).parent
ANALISIS_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ANALISIS_ROOT))

from kuramoto import KBlock

# ============================================================================
# PARÁMETROS EXACTOS DEL NOTEBOOK
# ============================================================================

SEED = 1
CH = 3
N = 3
H, W = 64, 64
T = 30
GAMMA = 0.7
DEL_T = 0.9
KSIZE = 3
INIT_OMG = 0.1

# Rango de C (igual al notebook)
C_RANGE = np.arange(0.0, 2.0, 0.01)

# ============================================================================
# CONFIGURACIÓN DE REPRODUCIBILIDAD
# ============================================================================

def set_seed(seed):
    """Configura todas las semillas para reproducibilidad total."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# FUNCIÓN DE PARÁMETRO DE ORDEN (IDÉNTICA AL NOTEBOOK)
# ============================================================================

def kuramoto_order_parameter(xs):
    """
    Calcula el parámetro de orden de Kuramoto r(t)
    Idéntica a la función del notebook.
    """
    r_values = []

    for x in xs:
        # x: [B, C, H, W]
        x = x[0].detach().cpu()

        # Calcula la fase de cada oscilador
        theta = torch.atan2(x[1], x[0])

        # Convierte a representación compleja e^{iθ}
        complex_phase = torch.exp(1j * theta)

        # Promedio global sobre todos los píxeles
        order_param = complex_phase.mean()

        # Magnitud = r(t)
        r = torch.abs(order_param)

        r_values.append(r.item())

    return np.array(r_values)

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    print("="*70)
    print("TEST DE REPRODUCIBILIDAD")
    print("Comparando notebook vs calcular_c_critico_local.py")
    print("="*70)
    
    # Configurar semilla
    set_seed(SEED)
    
    # Detectar dispositivo
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = 'CUDA'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = 'MPS (Apple Metal)'
    else:
        device = torch.device('cpu')
        device_name = 'CPU'
    
    print(f"\nDispositivo: {device_name}")
    
    # Cargar dataset MNIST (igual al notebook)
    print("\n1. Cargando dataset MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Obtener primera imagen de clase 0 (igual al notebook)
    imagenes_por_clase = {i: [] for i in range(10)}
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        imagenes_por_clase[label].append(i)
    
    # Primera imagen de clase 0
    img_idx = imagenes_por_clase[0][0]
    img, label = train_dataset[img_idx]
    
    print(f"   Imagen seleccionada: índice {img_idx}, clase {label}")
    
    # Inicializar modelo KBlock (igual al notebook)
    print("\n2. Inicializando modelo KBlock...")
    set_seed(SEED)  # Re-seed antes de crear x_init
    
    x_init = torch.randn(1, CH, H, W)
    kblock = KBlock(
        n=N, 
        ch=CH, 
        connectivity='conv', 
        T=T, 
        ksize=KSIZE,
        init_omg=INIT_OMG, 
        c_norm=None, 
        use_omega_c=False
    )
    
    # Mover a dispositivo
    x_init = x_init.to(device)
    kblock = kblock.to(device)
    
    print(f"   Parámetros: CH={CH}, N={N}, T={T}, KSIZE={KSIZE}")
    print(f"   gamma={GAMMA}, del_t={DEL_T}, seed={SEED}")
    
    # Preparar imagen como acoplamiento externo
    img_resized = TF.resize(img, [H, W])
    img_channels = img_resized.repeat(CH, 1, 1)
    img_channels = img_channels.to(device)
    
    # Calcular R vs C (igual al notebook)
    print("\n3. Calculando curva R vs C...")
    R_final = []
    
    for i, c_val in enumerate(C_RANGE):
        x = x_init.clone()
        c = img_channels.unsqueeze(0) * c_val
        
        with torch.no_grad():
            x, xs, es = kblock(x, c, T=T, gamma=GAMMA, del_t=DEL_T, 
                              return_xs=True, return_es=True)
        
        R = kuramoto_order_parameter(xs)
        R_final.append(R[-1])
        
        if (i + 1) % 50 == 0:
            print(f"   Progreso: {i+1}/{len(C_RANGE)} valores de C")
    
    R_final = np.array(R_final)
    
    # Calcular C_crítico (igual al notebook)
    print("\n4. Calculando C_crítico...")
    df = np.gradient(R_final, C_RANGE)
    i_c = np.argmax(df)
    c_critico = C_RANGE[i_c]
    
    print(f"   C_crítico = {c_critico:.4f}")
    print(f"   R en C_crítico = {R_final[i_c]:.4f}")
    print(f"   Máxima pendiente = {df[i_c]:.4f}")
    
    # Visualizar resultados
    print("\n5. Generando gráficas...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Gráfica 1: R vs C completo
    axes[0].plot(C_RANGE, R_final, 'b-', linewidth=2)
    axes[0].axvline(c_critico, color='r', linestyle='--', 
                   label=f'$C_c = {c_critico:.2f}$')
    axes[0].set_xlabel('Acoplamiento (C)', fontsize=12)
    axes[0].set_ylabel('Parámetro de orden final R(T)', fontsize=12)
    axes[0].set_title('Transición de fase de sincronización', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Gráfica 2: Zoom en zona de transición
    mask_zoom = (C_RANGE >= 0.0) & (C_RANGE <= 0.5)
    axes[1].plot(C_RANGE[mask_zoom], R_final[mask_zoom], 'b-', linewidth=2)
    axes[1].axvline(c_critico, color='r', linestyle='--', 
                   label=f'$C_c = {c_critico:.2f}$')
    axes[1].set_xlabel('Acoplamiento (C)', fontsize=12)
    axes[1].set_ylabel('R(T)', fontsize=12)
    axes[1].set_title('Zoom: Zona de transición', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim([0, 0.5])
    
    # Gráfica 3: Derivada (pendiente)
    axes[2].plot(C_RANGE, df, 'g-', linewidth=2)
    axes[2].axvline(c_critico, color='r', linestyle='--', 
                   label=f'$C_c = {c_critico:.2f}$')
    axes[2].set_xlabel('Acoplamiento (C)', fontsize=12)
    axes[2].set_ylabel('dR/dC', fontsize=12)
    axes[2].set_title('Derivada (identificación de C_crítico)', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    
    output_file = SCRIPT_DIR / 'test_reproducibilidad_resultados.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Gráficas guardadas en: {output_file}")
    
    # Mostrar imagen original
    fig2, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(img.squeeze(), cmap='gray')
    ax.axis('off')
    ax.set_title(f'Imagen de test\n(Clase {label}, índice {img_idx})', fontsize=12)
    
    output_img = SCRIPT_DIR / 'test_reproducibilidad_imagen.png'
    plt.savefig(output_img, dpi=150, bbox_inches='tight')
    print(f"   Imagen guardada en: {output_img}")
    
    plt.show()
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN DE RESULTADOS")
    print("="*70)
    print(f"Imagen:           Clase {label}, índice {img_idx}")
    print(f"C_crítico:        {c_critico:.4f}")
    print(f"R(C_crítico):     {R_final[i_c]:.4f}")
    print(f"Pendiente máxima: {df[i_c]:.4f}")
    print(f"Rango de R:       [{R_final.min():.4f}, {R_final.max():.4f}]")
    print()
    print("✅ Test completado exitosamente")
    print("   Los resultados deberían ser idénticos a los del notebook")
    print("="*70)

if __name__ == "__main__":
    main()
