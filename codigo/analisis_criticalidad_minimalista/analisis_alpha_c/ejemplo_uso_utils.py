"""
Ejemplo de uso de las utilidades refactorizadas.

Este script demuestra cómo usar los módulos de utils/ para análisis de alpha crítico.
"""

import sys
import os
from pathlib import Path

# Agregar path del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

# Importar utilidades refactorizadas
from utils import (
    get_device,
    generate_alphas,
    prepare_image,
    setup_matplotlib,
    save_figure,
    ModelConfig,
    AlphaAnalysisConfig,
    get_config_summary
)

# Importar módulos del proyecto
from kuramoto.modelo import KBlock
from datasets.loader import MNISTLoader
from analisis.criticalidad import KuramotoMetrics


def ejemplo_basico():
    """Ejemplo básico de uso de utilidades."""
    
    print("="*70)
    print("EJEMPLO DE USO DE UTILIDADES REFACTORIZADAS")
    print("="*70)
    
    # 1. Configurar dispositivo
    print("\n1. Configuración de dispositivo:")
    device = get_device('auto')
    print(f"   Dispositivo seleccionado: {device}")
    
    # 2. Generar valores de alpha
    print("\n2. Generación de alphas:")
    alphas = generate_alphas(0.0, 0.1, 0.005)
    print(f"   Generados {len(alphas)} valores de alpha")
    print(f"   Rango: [{alphas[0]:.6f}, {alphas[-1]:.6f}]")
    
    # 3. Configurar matplotlib
    print("\n3. Configuración de matplotlib:")
    setup_matplotlib(backend='Agg', cache_dir='_mplconfig')
    print("   ✓ Matplotlib configurado (backend Agg)")
    
    # 4. Usar configuración del modelo
    print("\n4. Configuración del modelo:")
    config = ModelConfig()
    print(f"   γ (gamma): {config.gamma}")
    print(f"   Δt (delta_t): {config.delta_t}")
    print(f"   T (timesteps): {config.timesteps}")
    print(f"   Canales: {config.ch}")
    
    # 5. Cargar y preparar imagen
    print("\n5. Preparación de imagen:")
    loader = MNISTLoader(root="./data", img_size=28)
    train_loader, _ = loader.get_mnist(batch_size=1)
    
    img, label = next(iter(train_loader))
    print(f"   Imagen original: {img.shape}, Etiqueta: {label.item()}")
    
    img_prepared = prepare_image(
        img[0],
        img_size=config.img_size,
        ch=config.ch,
        device=device
    )
    print(f"   Imagen preparada: {img_prepared.shape}")
    
    # 6. Crear modelo KBlock
    print("\n6. Creación de modelo Kuramoto:")
    kblock = KBlock(
        ch=config.ch,
        ksize=config.ksize,
        init_omg=config.init_omg
    ).to(device)
    print(f"   ✓ KBlock creado en {device}")
    
    # 7. Ejecutar dinámica para un alpha
    print("\n7. Ejecución de dinámica Kuramoto:")
    alpha_test = 0.01
    c_scaled = img_prepared.unsqueeze(0) * alpha_test
    x0 = torch.randn_like(c_scaled, device=device)
    
    with torch.no_grad():
        x_final, xs = kblock(
            x0,
            c_scaled,
            T=config.timesteps,
            gamma=config.gamma,
            delta_t=config.delta_t,
            return_xs=True
        )
    
    print(f"   α = {alpha_test}")
    print(f"   Estados guardados: {len(xs)}")
    print(f"   Shape de cada estado: {xs[0].shape}")
    
    # 8. Calcular parámetro de orden
    print("\n8. Cálculo de parámetro de orden:")
    metrics = KuramotoMetrics()
    R_values = []
    for x in xs:
        R = metrics.order_parameter(x)
        R_values.append(R.item())
    
    R_final = np.mean(R_values[-5:])  # Promedio últimos 5 pasos
    print(f"   R final (promedio últimos 5 pasos): {R_final:.4f}")
    
    # 9. Mostrar resumen de configuración
    print("\n9. Resumen de configuración:")
    print(get_config_summary())
    
    print("\n" + "="*70)
    print("✓ Ejemplo completado exitosamente")
    print("="*70)


def ejemplo_analisis_completo():
    """Ejemplo de análisis completo de alpha crítico para una imagen."""
    
    print("\n" + "="*70)
    print("EJEMPLO: ANÁLISIS COMPLETO DE ALPHA CRÍTICO")
    print("="*70)
    
    # Configuración
    device = get_device('auto')
    model_config = ModelConfig(timesteps=50)
    alpha_config = AlphaAnalysisConfig(
        alpha_start=0.0,
        alpha_end=0.05,
        alpha_step=0.001
    )
    
    print(f"\nDispositivo: {device}")
    print(f"Rango de alpha: [{alpha_config.alpha_start}, {alpha_config.alpha_end}]")
    print(f"Número de puntos: {alpha_config.num_points}")
    
    # Cargar imagen
    loader = MNISTLoader(root="./data", img_size=28)
    train_loader, _ = loader.get_mnist(batch_size=1)
    img, label = next(iter(train_loader))
    
    print(f"\nImagen: clase {label.item()}")
    
    # Preparar imagen
    img_prepared = prepare_image(
        img[0],
        img_size=model_config.img_size,
        ch=model_config.ch,
        device=device
    )
    
    # Crear modelo
    kblock = KBlock(
        ch=model_config.ch,
        ksize=model_config.ksize,
        init_omg=model_config.init_omg
    ).to(device)
    
    # Generar alphas
    alphas = generate_alphas(
        alpha_config.alpha_start,
        alpha_config.alpha_end,
        alpha_config.alpha_step
    )
    
    # Barrido de alpha
    print(f"\nEjecutando barrido de {len(alphas)} valores de alpha...")
    metrics = KuramotoMetrics()
    order_params = []
    
    for alpha in alphas[:10]:  # Solo primeros 10 para el ejemplo
        c_scaled = img_prepared.unsqueeze(0) * float(alpha)
        x0 = torch.randn_like(c_scaled, device=device)
        
        with torch.no_grad():
            _, xs = kblock(
                x0,
                c_scaled,
                T=model_config.timesteps,
                gamma=model_config.gamma,
                delta_t=model_config.delta_t,
                return_xs=True
            )
        
        # Calcular R promedio de últimos pasos
        R_values = [metrics.order_parameter(x).item() for x in xs[-alpha_config.window:]]
        R_avg = np.mean(R_values)
        order_params.append(R_avg)
    
    print(f"✓ Barrido completado (primeros 10 puntos)")
    print(f"\nResultados:")
    for i, (a, r) in enumerate(zip(alphas[:10], order_params)):
        print(f"  α = {a:.4f}  →  R = {r:.4f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Ejecutar ejemplos
    ejemplo_basico()
    
    # Descomentar para ejecutar análisis completo
    # ejemplo_analisis_completo()
