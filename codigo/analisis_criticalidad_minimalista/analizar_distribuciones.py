"""
Script: analizar_distribuciones_CORRECTED.py

ANÁLISIS CORRECTO DE DISTRIBUCIONES

Trabaja con datos SIN promediar para:
1. Analizar distribución de R(t) en cada paso temporal
2. Verificar normalidad ANTES de decidir usar media/mediana
3. Detectar heterocedasticidad temporal
4. Identificar momentos críticos en la evolución

"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, shapiro, kstest, anderson
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = "resultados_kuramoto_full_dataset_CORRECTED"
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analisis_distribuciones")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

def cargar_datos():
    """Carga métricas completas SIN promedios."""
    data_path = os.path.join(RESULTS_DIR, 'metricas_completas_CORRECTED.pt')
    
    if not os.path.exists(data_path):
        print(f"❌ Error: No se encontró {data_path}")
        print("   Primero ejecuta: python run_kuramoto_full_dataset_CORRECTED.py")
        sys.exit(1)
    
    print(f"📂 Cargando datos desde: {data_path}")
    data = torch.load(data_path, weights_only=False)
    
    metricas = data['metricas']
    T_steps = data['T_steps']
    
    print(f"✅ Datos cargados: {len(metricas):,} imágenes")
    print(f"   Pasos temporales: {T_steps+1}")
    print(f"   Total de valores R: {len(metricas) * (T_steps+1):,}")
    print()
    
    return metricas, T_steps

def construir_dataframe_completo(metricas, T_steps):
    """
    Construye DataFrame con TODAS las observaciones temporales.
    
    En lugar de:
      Imagen 1 → R_mean = 0.543 (1 fila)
    
    Tenemos:
      Imagen 1, t=0   → R = 0.123
      Imagen 1, t=1   → R = 0.234
      ...
      Imagen 1, t=100 → R = 0.543
      (101 filas por imagen)
    """
    print("🔄 Construyendo DataFrame completo...")
    print("   (Esto puede tomar unos minutos)")
    
    registros = []
    
    for m in metricas:
        if not m.get('success', False):
            continue
        
        idx = m['idx']
        label = m['label']
        R_series = m['R_series']
        
        # Crear una fila por cada paso temporal
        for t in range(len(R_series)):
            registros.append({
                'imagen_idx': idx,
                'clase': label,
                'tiempo_t': t,
                'R': R_series[t],
            })
    
    df = pd.DataFrame(registros)
    
    print(f"✅ DataFrame creado:")
    print(f"   Filas totales: {len(df):,}")
    print(f"   Imágenes únicas: {df['imagen_idx'].nunique():,}")
    print(f"   Pasos temporales: {df['tiempo_t'].nunique()}")
    print()
    
    return df

def analizar_normalidad_temporal(df, T_steps):
    """
    Analiza normalidad de R en CADA momento temporal.
    
    Pregunta: ¿En qué momentos t la distribución de R es gaussiana?
    """
    print("📊 Analizando normalidad temporal...")
    print()
    
    resultados = []
    
    for t in range(T_steps + 1):
        df_t = df[df['tiempo_t'] == t]
        
        # Todos los valores de R en el tiempo t (todas las imágenes, todas las clases)
        R_values = df_t['R'].values
        
        if len(R_values) < 3:
            continue
        
        # Tests de normalidad
        if len(R_values) <= 5000:
            stat_shapiro, p_shapiro = shapiro(R_values)
        else:
            muestra = np.random.choice(R_values, size=5000, replace=False)
            stat_shapiro, p_shapiro = shapiro(muestra)
        
        R_norm = (R_values - R_values.mean()) / R_values.std()
        stat_ks, p_ks = kstest(R_norm, 'norm')
        
        result_anderson = anderson(R_values, dist='norm')
        anderson_critical_5 = result_anderson.critical_values[2]
        anderson_rechaza = result_anderson.statistic > anderson_critical_5
        
        skew = stats.skew(R_values)
        kurt = stats.kurtosis(R_values, fisher=True)
        
        resultados.append({
            'tiempo_t': t,
            'N': len(R_values),
            'media': R_values.mean(),
            'mediana': np.median(R_values),
            'std': R_values.std(),
            'Shapiro_p': p_shapiro,
            'Shapiro_normal': 'Sí' if p_shapiro > 0.05 else 'No',
            'KS_p': p_ks,
            'KS_normal': 'Sí' if p_ks > 0.05 else 'No',
            'Anderson_normal': 'Sí' if not anderson_rechaza else 'No',
            'Skewness': skew,
            'Kurtosis': kurt
        })
    
    df_normalidad_temporal = pd.DataFrame(resultados)
    
    # Guardar
    csv_path = os.path.join(ANALYSIS_DIR, 'normalidad_temporal.csv')
    df_normalidad_temporal.to_csv(csv_path, index=False)
    print(f"✅ Normalidad temporal guardada: {csv_path}")
    
    # Resumen
    n_normal_shapiro = (df_normalidad_temporal['Shapiro_normal'] == 'Sí').sum()
    total_t = len(df_normalidad_temporal)
    
    print(f"\n📊 RESUMEN TEMPORAL:")
    print(f"   Momentos temporales analizados: {total_t}")
    print(f"   Shapiro-Wilk: {n_normal_shapiro}/{total_t} momentos son normales ({100*n_normal_shapiro/total_t:.0f}%)")
    print()
    
    return df_normalidad_temporal

def analizar_normalidad_por_clase(df, T_steps):
    """
    Analiza normalidad de R por clase (colapsando tiempo).
    
    Para cada clase, tenemos ~1000 imágenes × 101 pasos = ~101,000 valores de R
    """
    print("📊 Analizando normalidad por clase...")
    print()
    
    resultados = []
    
    for clase in range(10):
        df_clase = df[df['clase'] == clase]
        R_values = df_clase['R'].values
        
        if len(R_values) < 3:
            continue
        
        # Tests
        muestra = np.random.choice(R_values, size=min(5000, len(R_values)), replace=False)
        stat_shapiro, p_shapiro = shapiro(muestra)
        
        R_norm = (R_values - R_values.mean()) / R_values.std()
        stat_ks, p_ks = kstest(R_norm, 'norm')
        
        result_anderson = anderson(muestra, dist='norm')
        anderson_critical_5 = result_anderson.critical_values[2]
        anderson_rechaza = result_anderson.statistic > anderson_critical_5
        
        skew = stats.skew(R_values)
        kurt = stats.kurtosis(R_values, fisher=True)
        
        resultados.append({
            'clase': clase,
            'N': len(R_values),
            'imagenes': df_clase['imagen_idx'].nunique(),
            'media': R_values.mean(),
            'mediana': np.median(R_values),
            'std': R_values.std(),
            'Q1': np.percentile(R_values, 25),
            'Q3': np.percentile(R_values, 75),
            'Shapiro_p': p_shapiro,
            'Shapiro_normal': 'Sí' if p_shapiro > 0.05 else 'No',
            'KS_p': p_ks,
            'KS_normal': 'Sí' if p_ks > 0.05 else 'No',
            'Anderson_normal': 'Sí' if not anderson_rechaza else 'No',
            'Skewness': skew,
            'Kurtosis': kurt
        })
    
    df_normalidad_clase = pd.DataFrame(resultados)
    
    # Guardar
    csv_path = os.path.join(ANALYSIS_DIR, 'normalidad_por_clase.csv')
    df_normalidad_clase.to_csv(csv_path, index=False)
    print(f"✅ Normalidad por clase guardada: {csv_path}")
    
    # Mostrar resultados
    print("\n📊 RESULTADOS POR CLASE:")
    print("─" * 80)
    for _, row in df_normalidad_clase.iterrows():
        clase = int(row['clase'])
        normal = row['Shapiro_normal']
        simbolo = '✓' if normal == 'Sí' else '✗'
        
        print(f"Clase {clase}: N={row['N']:>7,} | "
              f"μ={row['media']:.4f}, σ={row['std']:.4f} | "
              f"Skew={row['Skewness']:+.3f}, Kurt={row['Kurtosis']:+.3f} | "
              f"{simbolo} {normal}")
    
    print()
    return df_normalidad_clase

def visualizar_distribucion_temporal(df, T_steps, df_normalidad_temporal):
    """
    Visualiza cómo evoluciona la distribución de R en el tiempo.
    """
    print("📊 Generando visualizaciones temporales...")
    
    # 1. Histogramas en momentos clave
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Distribución de R en Diferentes Momentos Temporales', 
                 fontsize=16, fontweight='bold')
    
    momentos = [0, 20, 40, 60, 80, 100]
    axes = axes.flatten()
    
    for idx, t in enumerate(momentos):
        ax = axes[idx]
        
        df_t = df[df['tiempo_t'] == t]
        R_values = df_t['R'].values
        
        # Histograma
        n, bins, patches = ax.hist(R_values, bins=50, density=True, 
                                   alpha=0.7, color='steelblue', 
                                   edgecolor='black', linewidth=0.5)
        
        # Ajuste gaussiano
        mu, sigma = R_values.mean(), R_values.std()
        x = np.linspace(R_values.min(), R_values.max(), 100)
        gauss_fit = norm.pdf(x, mu, sigma)
        ax.plot(x, gauss_fit, 'r-', linewidth=2, label=f'N({mu:.3f}, {sigma:.3f}²)')
        
        # Info de normalidad
        row_norm = df_normalidad_temporal[df_normalidad_temporal['tiempo_t'] == t].iloc[0]
        p_val = row_norm['Shapiro_p']
        is_normal = row_norm['Shapiro_normal']
        
        color = 'green' if is_normal == 'Sí' else 'red'
        ax.set_title(f't = {t}\np-value = {p_val:.4f} ({is_normal})', 
                    color=color, fontsize=11, fontweight='bold')
        
        ax.set_xlabel('R (Parámetro de Orden)', fontsize=9)
        ax.set_ylabel('Densidad', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf_path = os.path.join(ANALYSIS_DIR, 'distribucion_temporal_R.pdf')
    plt.savefig(pdf_path, format='pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {pdf_path}")
    
    # 2. Evolución de estadísticos en el tiempo
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Evolución Temporal de Estadísticos de R', fontsize=16, fontweight='bold')
    
    tiempos = df_normalidad_temporal['tiempo_t'].values
    medias = df_normalidad_temporal['media'].values
    medianas = df_normalidad_temporal['mediana'].values
    stds = df_normalidad_temporal['std'].values
    
    # Media vs Mediana
    ax = axes[0, 0]
    ax.plot(tiempos, medias, 'b-', label='Media', linewidth=2)
    ax.plot(tiempos, medianas, 'r--', label='Mediana', linewidth=2)
    ax.set_xlabel('Tiempo t', fontsize=10)
    ax.set_ylabel('R', fontsize=10)
    ax.set_title('Media vs Mediana', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Desviación estándar
    ax = axes[0, 1]
    ax.plot(tiempos, stds, 'g-', linewidth=2)
    ax.set_xlabel('Tiempo t', fontsize=10)
    ax.set_ylabel('σ', fontsize=10)
    ax.set_title('Desviación Estándar', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Skewness
    ax = axes[1, 0]
    skews = df_normalidad_temporal['Skewness'].values
    ax.plot(tiempos, skews, 'orange', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Tiempo t', fontsize=10)
    ax.set_ylabel('Asimetría', fontsize=10)
    ax.set_title('Skewness (0 = simétrica)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Kurtosis
    ax = axes[1, 1]
    kurts = df_normalidad_temporal['Kurtosis'].values
    ax.plot(tiempos, kurts, 'purple', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Tiempo t', fontsize=10)
    ax.set_ylabel('Curtosis', fontsize=10)
    ax.set_title('Kurtosis (0 = gaussiana)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf_path = os.path.join(ANALYSIS_DIR, 'evolucion_estadisticos_R.pdf')
    plt.savefig(pdf_path, format='pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {pdf_path}")
    
    print()

def visualizar_distribucion_por_clase(df, df_normalidad_clase):
    """Visualiza distribuciones por clase."""
    print("📊 Generando visualizaciones por clase...")
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Distribución de R por Clase (Todos los Tiempos)', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for clase in range(10):
        ax = axes[clase]
        
        df_clase = df[df['clase'] == clase]
        R_values = df_clase['R'].values
        
        # Histograma
        n, bins, patches = ax.hist(R_values, bins=50, density=True, 
                                   alpha=0.7, color='steelblue', 
                                   edgecolor='black', linewidth=0.5)
        
        # Ajuste gaussiano
        mu, sigma = R_values.mean(), R_values.std()
        x = np.linspace(R_values.min(), R_values.max(), 100)
        gauss_fit = norm.pdf(x, mu, sigma)
        ax.plot(x, gauss_fit, 'r-', linewidth=2)
        
        # Info
        row = df_normalidad_clase[df_normalidad_clase['clase'] == clase].iloc[0]
        is_normal = row['Shapiro_normal']
        p_val = row['Shapiro_p']
        
        color = 'green' if is_normal == 'Sí' else 'red'
        ax.set_title(f'Clase {clase} (N={row["N"]:,})\n'
                    f'p={p_val:.4f} ({is_normal})',
                    color=color, fontsize=9, fontweight='bold')
        
        ax.set_xlabel('R', fontsize=8)
        ax.set_ylabel('Densidad', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf_path = os.path.join(ANALYSIS_DIR, 'distribucion_por_clase_R.pdf')
    plt.savefig(pdf_path, format='pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {pdf_path}")
    print()

def generar_recomendaciones(df_normalidad_temporal, df_normalidad_clase):
    """Genera recomendaciones sobre qué estadístico usar."""
    print("📝 Generando recomendaciones...")
    
    recomendaciones = f"""
═══════════════════════════════════════════════════════════════════════════════
                    RECOMENDACIONES ESTADÍSTICAS
═══════════════════════════════════════════════════════════════════════════════

Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. ANÁLISIS TEMPORAL
─────────────────────────────────────────────────────────────────────────────
"""
    
    n_normal_temporal = (df_normalidad_temporal['Shapiro_normal'] == 'Sí').sum()
    total_temporal = len(df_normalidad_temporal)
    
    recomendaciones += f"\nMomentos normales: {n_normal_temporal}/{total_temporal} ({100*n_normal_temporal/total_temporal:.0f}%)\n\n"
    
    if n_normal_temporal / total_temporal > 0.7:
        recomendaciones += "✓ RECOMENDACIÓN: Usar MEDIA para describir evolución temporal de R\n"
        recomendaciones += "  La mayoría de momentos temporales presentan distribución normal.\n"
    else:
        recomendaciones += "⚠ RECOMENDACIÓN: Usar MEDIANA para describir evolución temporal de R\n"
        recomendaciones += "  Muchos momentos temporales NO presentan distribución normal.\n"
    
    recomendaciones += f"\n2. ANÁLISIS POR CLASE\n"
    recomendaciones += "─" * 80 + "\n\n"
    
    for _, row in df_normalidad_clase.iterrows():
        clase = int(row['clase'])
        is_normal = row['Shapiro_normal']
        
        recomendaciones += f"Clase {clase}: "
        
        if is_normal == 'Sí':
            recomendaciones += f"✓ Usar MEDIA (μ={row['media']:.4f} ± {row['std']:.4f})\n"
        else:
            recomendaciones += f"⚠ Usar MEDIANA (Md={row['mediana']:.4f} [{row['Q1']:.4f}, {row['Q3']:.4f}])\n"
    
    recomendaciones += "\n" + "=" * 80 + "\n"
    
    # Guardar
    txt_path = os.path.join(ANALYSIS_DIR, 'recomendaciones_estadisticas.txt')
    with open(txt_path, 'w') as f:
        f.write(recomendaciones)
    
    print(recomendaciones)
    print(f"✅ Recomendaciones guardadas: {txt_path}")
    print()

def main():
    """Función principal."""
    print("="*70)
    print("ANÁLISIS CORRECTO DE DISTRIBUCIONES")
    print("="*70)
    print()
    
    # Cargar datos
    metricas, T_steps = cargar_datos()
    
    # Construir DataFrame completo
    df = construir_dataframe_completo(metricas, T_steps)
    
    # Análisis de normalidad temporal
    df_normalidad_temporal = analizar_normalidad_temporal(df, T_steps)
    
    # Análisis de normalidad por clase
    df_normalidad_clase = analizar_normalidad_por_clase(df, T_steps)
    
    # Visualizaciones
    visualizar_distribucion_temporal(df, T_steps, df_normalidad_temporal)
    visualizar_distribucion_por_clase(df, df_normalidad_clase)
    
    # Recomendaciones
    generar_recomendaciones(df_normalidad_temporal, df_normalidad_clase)
    
    print("="*70)
    print("ANÁLISIS COMPLETADO ✅")
    print("="*70)
    print()
    print(f"📁 Resultados en: {ANALYSIS_DIR}/")
    print()

if __name__ == "__main__":
    main()
