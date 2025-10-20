"""
Script de ejemplo que replica el análisis del notebook kuramoto_pruebas_basico.ipynb
usando el módulo analisis_criticalidad_minimalista.

Este script demuestra:
1. Carga y preparación de datos MNIST
2. Inicialización del modelo Kuramoto (KBlock)
3. Evolución temporal de la dinámica
4. Cálculo de métricas de criticalidad
5. Visualizaciones y animaciones
"""

import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
import random
import matplotlib.pyplot as plt

# Importar el módulo local
import sys
sys.path.insert(0, '/home/crperezp/proyectos/ProyectoInvTeorica/Proyecto-Inv.-teorica./codigo')
from analisis_criticalidad_minimalista import (
    KBlock,
    KuramotoMetrics,
    Entropia,
    DFA,
    PSD,
    MutualInformation,
    Correlacion,
    SeriesAnalysis,
    Visualizador,
    Animaciones
)


def main():
    """Función principal del ejemplo."""
    
    # =============== 1. CARGA DE DATOS ===============
    print("=" * 60)
    print("1. CARGANDO DATOS MNIST")
    print("=" * 60)
    
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
    
    # Obtener una imagen de ejemplo
    img, label = train_dataset[0]
    print(f"Imagen de ejemplo: Label = {label}")
    print(f"Shape original: {img.shape}")
    
    # =============== 2. PREPARACIÓN DE INPUT ===============
    print("\n" + "=" * 60)
    print("2. PREPARANDO INPUT PARA KURAMOTO")
    print("=" * 60)
    
    # Redimensionar y replicar en canales
    img_resized = TF.resize(img, [64, 64])
    img_channels = img_resized.repeat(4, 1, 1)
    c = img_channels.unsqueeze(0) * 1  # Campo de acoplamiento externo
    
    print(f"Shape del campo c: {c.shape}")
    
    # =============== 3. CONFIGURACIÓN DEL MODELO ===============
    print("\n" + "=" * 60)
    print("3. INICIALIZANDO MODELO KURAMOTO")
    print("=" * 60)
    
    # Parámetros
    ch = 4      # Canales
    n = 4       # Dimensión de osciladores (ch/n = número de osciladores por pixel)
    h, w = 64, 64
    T = 100     # Pasos de integración
    
    # Reproducibilidad
    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Crear modelo
    kblock = KBlock(
        n=n, 
        ch=ch, 
        connectivity='conv', 
        T=T, 
        ksize=3, 
        init_omg=0.1, 
        c_norm=None, 
        use_omega_c=False
    )
    
    print(f"Modelo creado: {kblock.__class__.__name__}")
    print(f"Parámetros: ch={ch}, n={n}, T={T}")
    
    # =============== 4. EVOLUCIÓN TEMPORAL ===============
    print("\n" + "=" * 60)
    print("4. EJECUTANDO DINÁMICA DE KURAMOTO")
    print("=" * 60)
    
    # Estados iniciales aleatorios
    x = torch.randn(1, ch, h, w)
    
    # Evolucionar
    x_final, xs, es = kblock(
        x, c, 
        T=T, 
        gamma=0.7, 
        del_t=0.9, 
        return_xs=True, 
        return_es=True
    )
    
    print(f"Evolución completa: {len(xs)} estados guardados")
    print(f"Energías: {len(es)} valores")
    
    # =============== 5. ANÁLISIS DE CRITICALIDAD ===============
    print("\n" + "=" * 60)
    print("5. CALCULANDO MÉTRICAS DE CRITICALIDAD")
    print("=" * 60)
    
    # 5.1 Parámetro de orden de Kuramoto
    R = KuramotoMetrics.order_parameter(xs, ch_pair=(0, 1))
    print(f"Parámetro de orden R(t): min={R.min():.4f}, max={R.max():.4f}, final={R[-1]:.4f}")
    
    # 5.2 Series temporales de magnitud
    series = KuramotoMetrics.magnitudes_mean_series(xs)
    print(f"Series de magnitud: shape={series.shape}")
    
    # 5.3 Varianza temporal
    var_per_channel = SeriesAnalysis.compute_variance_temporal(series)
    print(f"Varianza temporal por canal: {var_per_channel}")
    
    # 5.4 PSD y pendiente
    global_series = series.mean(axis=1)
    f, P, slope = PSD.psd_slope(global_series)
    print(f"PSD slope: {slope:.4f}")
    
    # 5.5 DFA
    scales, F, alpha = DFA.dfa(global_series)
    print(f"DFA alpha: {alpha:.4f}")
    
    # 5.6 Entropía
    if isinstance(xs, list):
        xs_tensor = torch.stack(xs)
    else:
        xs_tensor = xs
    entropy_results = Entropia.entropy_analysis(xs_tensor)
    print("Entropía por canal:")
    for canal, metrics in entropy_results.items():
        print(f"  {canal}: {metrics['Entropía de Shannon']:.4f}")
    
    # 5.7 Información mutua
    ch_count = series.shape[1]
    MI = np.zeros((ch_count, ch_count))
    for i in range(ch_count):
        for j in range(ch_count):
            MI[i, j] = MutualInformation.mutual_info(series[:, i], series[:, j])
    print(f"Información mutua (diagonal promedio): {np.diag(MI).mean():.4f}")
    
    # 5.8 Correlación
    corr_matrix = Correlacion.pearson_matrix(series)
    print(f"Correlación entre canales: min={corr_matrix.min():.4f}, max={corr_matrix.max():.4f}")
    
    # 5.9 Estadísticas detalladas
    stats = SeriesAnalysis.compute_channel_statistics(xs)
    print(f"Cambio relativo promedio: {stats['rel_change'].mean():.4f}")
    
    # =============== 6. VISUALIZACIONES ===============
    print("\n" + "=" * 60)
    print("6. GENERANDO VISUALIZACIONES")
    print("=" * 60)
    
    # 6.1 Parámetro de orden
    plt.figure(figsize=(8, 4))
    plt.plot(R)
    plt.xlabel('t')
    plt.ylabel('R(t)')
    plt.title('Orden de Kuramoto (global)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('kuramoto_order.png', dpi=150)
    print("✅ Guardado: kuramoto_order.png")
    plt.close()
    
    # 6.2 Series temporales
    Visualizador.plot_series(
        series, 
        title='Evolución temporal de la magnitud media por canal',
        ylabel='mean(|x|)'
    )
    
    # 6.3 Matriz de correlación
    Visualizador.plot_matrix(
        corr_matrix,
        title='Correlación entre series de magnitud media (canales)',
        cmap='coolwarm'
    )
    
    # 6.4 Energía
    Visualizador.plot_energy(es, normalize=True)
    
    # 6.5 PSD
    plt.figure(figsize=(8, 4))
    plt.loglog(f, P)
    plt.title(f"PSD del series global (slope={slope:.3f})")
    plt.xlabel('Frecuencia')
    plt.ylabel('Potencia')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('psd_analysis.png', dpi=150)
    print("✅ Guardado: psd_analysis.png")
    plt.close()
    
    # 6.6 DFA
    plt.figure(figsize=(8, 4))
    plt.loglog(scales, F, '.-')
    plt.title(f"DFA (alpha={alpha:.3f})")
    plt.xlabel('Scale')
    plt.ylabel('F(scale)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('dfa_analysis.png', dpi=150)
    print("✅ Guardado: dfa_analysis.png")
    plt.close()
    
    # =============== 7. ANIMACIONES ===============
    print("\n" + "=" * 60)
    print("7. GENERANDO ANIMACIONES")
    print("=" * 60)
    
    # 7.1 Dinámica de un canal
    Animaciones.animate_dynamics(xs, channel=0, interval=100, filename='animacion_canal0.gif')
    
    # 7.2 Magnitud total
    Animaciones.animate_magnitude(xs, interval=100, filename='animacion_magnitud.gif')
    
    # 7.3 Campo vectorial
    Animaciones.animate_vector_field(xs, step=4, interval=100, filename='animacion_campo.gif')
    
    # 7.4 Evolución de fases
    Animaciones.animate_phase_evolution(xs, interval=100, filename='animacion_fases.gif')
    
    # 7.5 Evolución de magnitudes por canal
    Animaciones.animate_magnitude_evolution(xs, interval=100, filename='animacion_magnitudes_canales.gif')
    
    # =============== 8. RESUMEN FINAL ===============
    print("\n" + "=" * 60)
    print("RESUMEN DE ANÁLISIS")
    print("=" * 60)
    print(f"✅ Modelo ejecutado: {T} pasos de integración")
    print(f"✅ Estados guardados: {len(xs)}")
    print(f"✅ Parámetro de orden final R: {R[-1]:.4f}")
    print(f"✅ DFA alpha: {alpha:.4f}")
    print(f"✅ PSD slope: {slope:.4f}")
    print(f"✅ Entropía promedio: {np.mean([m['Entropía de Shannon'] for m in entropy_results.values()]):.4f}")
    print("\n🎉 Análisis completado exitosamente!")
    

if __name__ == "__main__":
    main()
