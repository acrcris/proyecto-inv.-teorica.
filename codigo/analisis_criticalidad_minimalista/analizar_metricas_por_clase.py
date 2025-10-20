"""
Script: analizar_metricas_por_clase.py

Analiza las métricas de criticalidad para cada clase de MNIST.
Lee los estados guardados por run_kuramoto_por_clase.py y calcula:
- Parámetro de orden de Kuramoto R(t)
- DFA (exponente alpha)
- PSD (pendiente espectral)
- Entropía de Shannon
- Información mutua
- Correlación entre canales
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Agregar el path del módulo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analisis.criticalidad import (
    KuramotoMetrics,
    DFA,
    PSD,
    Entropia,
    MutualInformation,
    Correlacion
)
from analisis.series_temporales import SeriesAnalysis

# Carpetas
RESULTS_DIR = "resultados_kuramoto_por_clase"
ANALYSIS_DIR = "analisis_metricas"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

def cargar_resultados(clase):
    """Carga los resultados guardados para una clase."""
    path = os.path.join(RESULTS_DIR, f"clase_{clase}.pt")
    data = torch.load(path)
    return data['xs'], data['es']

def calcular_metricas(xs, es, clase):
    """Calcula todas las métricas de criticalidad para una clase."""
    print(f"\n{'='*60}")
    print(f"Analizando Clase {clase}")
    print(f"{'='*60}")
    
    resultados = {}
    
    # 1. Parámetro de orden de Kuramoto
    R = KuramotoMetrics.order_parameter(xs, ch_pair=(0, 1))
    resultados['R_mean'] = R.mean()
    resultados['R_final'] = R[-1]
    resultados['R_std'] = R.std()
    print(f"Parámetro de orden R(t):")
    print(f"  - Media: {resultados['R_mean']:.4f}")
    print(f"  - Final: {resultados['R_final']:.4f}")
    print(f"  - Desv. Est.: {resultados['R_std']:.4f}")
    
    # 2. Series temporales de magnitud
    series = KuramotoMetrics.magnitudes_mean_series(xs)
    global_series = series.mean(axis=1)
    
    # 3. DFA
    try:
        scales, F, alpha = DFA.dfa(global_series)
        resultados['DFA_alpha'] = alpha
        print(f"DFA:")
        print(f"  - Alpha: {alpha:.4f} (crítico si ≈ 1.0)")
    except Exception as e:
        resultados['DFA_alpha'] = np.nan
        print(f"DFA: Error al calcular ({e})")
    
    # 4. PSD
    try:
        f, Pxx, slope = PSD.psd_slope(global_series)
        resultados['PSD_slope'] = slope
        print(f"PSD:")
        print(f"  - Pendiente: {slope:.4f} (crítico si ≈ -1.0)")
    except Exception as e:
        resultados['PSD_slope'] = np.nan
        print(f"PSD: Error al calcular ({e})")
    
    # 5. Entropía de Shannon
    if isinstance(xs, list):
        xs_tensor = torch.stack(xs)
    else:
        xs_tensor = xs
    entropy_results = Entropia.entropy_analysis(xs_tensor)
    entropias = [v['Entropía de Shannon'] for v in entropy_results.values()]
    resultados['Entropia_mean'] = np.mean(entropias)
    resultados['Entropia_std'] = np.std(entropias)
    print(f"Entropía de Shannon:")
    print(f"  - Media: {resultados['Entropia_mean']:.4f}")
    print(f"  - Desv. Est.: {resultados['Entropia_std']:.4f}")
    
    # 6. Correlación
    corr_matrix = Correlacion.pearson_matrix(series)
    # Correlación media (excluyendo diagonal)
    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
    resultados['Corr_mean'] = corr_matrix[mask].mean()
    resultados['Corr_std'] = corr_matrix[mask].std()
    print(f"Correlación entre canales:")
    print(f"  - Media: {resultados['Corr_mean']:.4f}")
    print(f"  - Desv. Est.: {resultados['Corr_std']:.4f}")
    
    # 7. Información Mutua
    ch_count = series.shape[1]
    MI = np.zeros((ch_count, ch_count))
    for i in range(ch_count):
        for j in range(ch_count):
            MI[i, j] = MutualInformation.mutual_info(series[:, i], series[:, j])
    resultados['MI_mean'] = MI[mask].mean()
    resultados['MI_std'] = MI[mask].std()
    print(f"Información Mutua:")
    print(f"  - Media: {resultados['MI_mean']:.4f}")
    print(f"  - Desv. Est.: {resultados['MI_std']:.4f}")
    
    # 8. Estadísticas de series temporales
    stats = SeriesAnalysis.compute_channel_statistics(xs)
    resultados['Variance_mean'] = stats['serie_channel_std'].mean()
    
    return resultados, R, series, corr_matrix

def generar_graficos(clase, R, series, corr_matrix):
    """Genera gráficos para una clase."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Parámetro de orden R(t)
    axes[0, 0].plot(R, linewidth=2)
    axes[0, 0].set_xlabel('Tiempo t')
    axes[0, 0].set_ylabel('R(t)')
    axes[0, 0].set_title(f'Clase {clase}: Parámetro de Orden de Kuramoto')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Umbral crítico')
    axes[0, 0].legend()
    
    # 2. Series temporales de magnitud
    for i in range(series.shape[1]):
        axes[0, 1].plot(series[:, i], label=f'Canal {i}', alpha=0.7)
    axes[0, 1].set_xlabel('Tiempo t')
    axes[0, 1].set_ylabel('Magnitud media')
    axes[0, 1].set_title(f'Clase {clase}: Evolución de Magnitudes')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Matriz de correlación
    im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 0].set_title(f'Clase {clase}: Correlación entre Canales')
    axes[1, 0].set_xlabel('Canal')
    axes[1, 0].set_ylabel('Canal')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 4. Distribución de R(t)
    axes[1, 1].hist(R, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(R.mean(), color='r', linestyle='--', linewidth=2, label=f'Media: {R.mean():.3f}')
    axes[1, 1].set_xlabel('R(t)')
    axes[1, 1].set_ylabel('Frecuencia')
    axes[1, 1].set_title(f'Clase {clase}: Distribución de R(t)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(ANALYSIS_DIR, f'clase_{clase}_metricas.pdf')
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico guardado: {out_path}")

def main():
    """Función principal."""
    print("="*60)
    print("ANÁLISIS DE MÉTRICAS DE CRITICALIDAD POR CLASE")
    print("="*60)
    
    # Almacenar resultados de todas las clases
    todas_las_metricas = []
    
    for clase in range(10):
        # Cargar resultados
        xs, es = cargar_resultados(clase)
        
        # Calcular métricas
        resultados, R, series, corr_matrix = calcular_metricas(xs, es, clase)
        resultados['Clase'] = clase
        todas_las_metricas.append(resultados)
        
        # Generar gráficos
        generar_graficos(clase, R, series, corr_matrix)
    
    # Crear DataFrame con todas las métricas
    df = pd.DataFrame(todas_las_metricas)
    df = df[['Clase', 'R_mean', 'R_final', 'DFA_alpha', 'PSD_slope', 
             'Entropia_mean', 'Corr_mean', 'MI_mean']]
    
    # Guardar CSV
    csv_path = os.path.join(ANALYSIS_DIR, 'resumen_metricas_por_clase.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Resumen guardado: {csv_path}")
    
    # Mostrar tabla resumen
    print("\n" + "="*60)
    print("RESUMEN DE MÉTRICAS POR CLASE")
    print("="*60)
    print(df.to_string(index=False))
    
    # Generar gráfico comparativo
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metricas = [
        ('R_mean', 'Parámetro de Orden R (media)', 0.5),
        ('DFA_alpha', 'DFA Alpha', 1.0),
        ('PSD_slope', 'PSD Pendiente', -1.0),
        ('Entropia_mean', 'Entropía de Shannon (media)', None),
        ('Corr_mean', 'Correlación (media)', None),
        ('MI_mean', 'Información Mutua (media)', None)
    ]
    
    for idx, (metrica, titulo, linea_critica) in enumerate(metricas):
        ax = axes[idx // 3, idx % 3]
        valores = df[metrica].values
        ax.bar(range(10), valores, alpha=0.7, edgecolor='black')
        if linea_critica is not None:
            ax.axhline(y=linea_critica, color='r', linestyle='--', 
                      linewidth=2, label=f'Crítico: {linea_critica}')
            ax.legend()
        ax.set_xlabel('Clase')
        ax.set_ylabel(metrica)
        ax.set_title(titulo)
        ax.set_xticks(range(10))
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    comp_path = os.path.join(ANALYSIS_DIR, 'comparacion_metricas_clases.pdf')
    plt.savefig(comp_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico comparativo guardado: {comp_path}")
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
    print(f"Resultados guardados en: {ANALYSIS_DIR}/")

if __name__ == "__main__":
    main()
