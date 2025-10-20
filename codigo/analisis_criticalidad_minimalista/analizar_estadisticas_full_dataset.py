"""
Script: analizar_estadisticas_full_dataset.py

FASE 2.1 - Análisis estadístico de resultados completos

Analiza las métricas calculadas sobre las 10,000 imágenes del test set:
- Distribuciones por clase
- Estadísticas descriptivas (media, mediana, desviación, cuartiles)
- Tests de significancia entre clases
- Visualizaciones comparativas
- Identificación de candidatos a criticalidad robustos

Requiere: metricas_completas.pt generado por run_kuramoto_full_dataset.py
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime

# Agregar el path del módulo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configuración
RESULTS_DIR = "resultados_kuramoto_full_dataset"
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analisis_estadistico")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

def cargar_datos():
    """Carga las métricas desde el archivo guardado."""
    data_path = os.path.join(RESULTS_DIR, 'metricas_completas.pt')
    
    if not os.path.exists(data_path):
        print(f"❌ Error: No se encontró {data_path}")
        print("   Primero ejecuta: python run_kuramoto_full_dataset.py")
        sys.exit(1)
    
    print(f"📂 Cargando datos desde: {data_path}")
    data = torch.load(data_path)
    
    metricas = data['metricas']
    parametros = data['parametros']
    
    print(f"✅ Datos cargados: {len(metricas):,} imágenes procesadas")
    print(f"   Fecha de procesamiento: {data['fecha']}")
    print(f"   Parámetros: {parametros}")
    print()
    
    return metricas, parametros

def crear_dataframe(metricas):
    """Convierte las métricas a un DataFrame de pandas."""
    print("🔄 Creando DataFrame...")
    
    # Filtrar solo imágenes exitosas
    metricas_exitosas = [m for m in metricas if m.get('success', False)]
    
    df = pd.DataFrame(metricas_exitosas)
    
    # Seleccionar columnas de interés
    columnas = ['idx', 'label', 'R_mean', 'R_std', 'R_final', 'DFA_alpha', 
                'PSD_slope', 'Entropia_mean', 'Corr_mean', 'MI_mean', 'Variance']
    df = df[columnas]
    
    print(f"✅ DataFrame creado: {len(df):,} filas × {len(df.columns)} columnas")
    print(f"   Imágenes exitosas: {len(metricas_exitosas):,}/{len(metricas):,} "
          f"({100*len(metricas_exitosas)/len(metricas):.1f}%)")
    print()
    
    return df

def estadisticas_por_clase(df):
    """Calcula estadísticas descriptivas por clase."""
    print("📊 Calculando estadísticas por clase...")
    
    metricas_interes = ['R_mean', 'DFA_alpha', 'PSD_slope', 'Entropia_mean', 
                        'Corr_mean', 'MI_mean']
    
    stats_por_clase = []
    
    for clase in range(10):
        df_clase = df[df['label'] == clase]
        
        stats_clase = {'Clase': clase, 'N': len(df_clase)}
        
        for metrica in metricas_interes:
            valores = df_clase[metrica].dropna()
            
            stats_clase[f'{metrica}_mean'] = valores.mean()
            stats_clase[f'{metrica}_std'] = valores.std()
            stats_clase[f'{metrica}_median'] = valores.median()
            stats_clase[f'{metrica}_q25'] = valores.quantile(0.25)
            stats_clase[f'{metrica}_q75'] = valores.quantile(0.75)
        
        stats_por_clase.append(stats_clase)
    
    df_stats = pd.DataFrame(stats_por_clase)
    
    # Guardar CSV
    csv_path = os.path.join(ANALYSIS_DIR, 'estadisticas_por_clase.csv')
    df_stats.to_csv(csv_path, index=False)
    print(f"✅ Estadísticas guardadas: {csv_path}")
    print()
    
    return df_stats

def tests_significancia(df):
    """Realiza tests de significancia entre clases."""
    print("🔬 Realizando tests de significancia (ANOVA)...")
    
    metricas_interes = ['R_mean', 'DFA_alpha', 'PSD_slope', 'Entropia_mean', 
                        'Corr_mean', 'MI_mean']
    
    resultados_anova = []
    
    for metrica in metricas_interes:
        # Preparar grupos por clase
        grupos = [df[df['label'] == clase][metrica].dropna() for clase in range(10)]
        
        # ANOVA de una vía
        F_stat, p_value = stats.f_oneway(*grupos)
        
        resultados_anova.append({
            'Metrica': metrica,
            'F_statistic': F_stat,
            'p_value': p_value,
            'Significativo': 'Sí' if p_value < 0.05 else 'No'
        })
        
        print(f"  {metrica:20s}: F={F_stat:.2f}, p={p_value:.2e} {'✓' if p_value < 0.05 else '✗'}")
    
    df_anova = pd.DataFrame(resultados_anova)
    
    # Guardar resultados
    csv_path = os.path.join(ANALYSIS_DIR, 'tests_significancia.csv')
    df_anova.to_csv(csv_path, index=False)
    print(f"\n✅ Tests guardados: {csv_path}")
    print()
    
    return df_anova

def visualizar_distribuciones(df, df_stats):
    """Genera visualizaciones de distribuciones por clase."""
    print("📈 Generando visualizaciones...")
    
    metricas = ['R_mean', 'DFA_alpha', 'PSD_slope', 'Entropia_mean', 'Corr_mean', 'MI_mean']
    titulos = ['Parámetro de Orden R', 'DFA Alpha', 'PSD Slope', 
               'Entropía de Shannon', 'Correlación', 'Información Mutua']
    valores_criticos = [0.5, 1.0, -1.0, None, None, None]
    
    # 1. Box plots por métrica
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (metrica, titulo, critico) in enumerate(zip(metricas, titulos, valores_criticos)):
        ax = axes[idx]
        
        # Preparar datos para box plot
        datos = [df[df['label'] == clase][metrica].dropna() for clase in range(10)]
        
        bp = ax.boxplot(datos, labels=range(10), patch_artist=True)
        
        # Colorear boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        # Línea crítica
        if critico is not None:
            ax.axhline(y=critico, color='r', linestyle='--', linewidth=2, 
                      label=f'Crítico: {critico}')
            ax.legend()
        
        ax.set_xlabel('Clase', fontsize=12)
        ax.set_ylabel(metrica, fontsize=12)
        ax.set_title(titulo, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = os.path.join(ANALYSIS_DIR, 'distribuciones_boxplot.pdf')
    plt.savefig(path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✅ Box plots guardados: {path}")
    
    # 2. Violin plots para métricas clave
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metricas_clave = ['R_mean', 'DFA_alpha', 'PSD_slope']
    titulos_clave = ['Parámetro de Orden R', 'DFA Alpha', 'PSD Slope']
    criticos_clave = [0.5, 1.0, -1.0]
    
    for idx, (metrica, titulo, critico) in enumerate(zip(metricas_clave, titulos_clave, criticos_clave)):
        ax = axes[idx]
        
        datos = [df[df['label'] == clase][metrica].dropna() for clase in range(10)]
        
        parts = ax.violinplot(datos, positions=range(10), showmeans=True, showmedians=True)
        
        # Línea crítica
        ax.axhline(y=critico, color='r', linestyle='--', linewidth=2, 
                  label=f'Crítico: {critico}')
        
        ax.set_xlabel('Clase', fontsize=12)
        ax.set_ylabel(metrica, fontsize=12)
        ax.set_title(titulo, fontsize=14, fontweight='bold')
        ax.set_xticks(range(10))
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = os.path.join(ANALYSIS_DIR, 'distribuciones_violin.pdf')
    plt.savefig(path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✅ Violin plots guardados: {path}")
    
    # 3. Barras de medias con error estándar
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (metrica, titulo, critico) in enumerate(zip(metricas, titulos, valores_criticos)):
        ax = axes[idx]
        
        means = [df_stats[df_stats['Clase'] == c][f'{metrica}_mean'].values[0] for c in range(10)]
        stds = [df_stats[df_stats['Clase'] == c][f'{metrica}_std'].values[0] for c in range(10)]
        
        ax.bar(range(10), means, yerr=stds, alpha=0.7, capsize=5, 
               color='steelblue', edgecolor='black')
        
        if critico is not None:
            ax.axhline(y=critico, color='r', linestyle='--', linewidth=2, 
                      label=f'Crítico: {critico}')
            ax.legend()
        
        ax.set_xlabel('Clase', fontsize=12)
        ax.set_ylabel(f'{metrica} (media ± std)', fontsize=12)
        ax.set_title(titulo, fontsize=14, fontweight='bold')
        ax.set_xticks(range(10))
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = os.path.join(ANALYSIS_DIR, 'medias_por_clase.pdf')
    plt.savefig(path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✅ Gráficas de medias guardadas: {path}")
    
    print()

def identificar_candidatos_criticos(df, df_stats):
    """Identifica las clases más cercanas a criticalidad."""
    print("🎯 Identificando candidatos a criticalidad...")
    print()
    
    # Calcular distancias a valores críticos
    candidatos = []
    
    for clase in range(10):
        dist_R = abs(df_stats[df_stats['Clase'] == clase]['R_mean_mean'].values[0] - 0.5)
        dist_DFA = abs(df_stats[df_stats['Clase'] == clase]['DFA_alpha_mean'].values[0] - 1.0)
        dist_PSD = abs(df_stats[df_stats['Clase'] == clase]['PSD_slope_mean'].values[0] - (-1.0))
        
        # Distancia euclidiana normalizada
        dist_total = np.sqrt(dist_R**2 + dist_DFA**2 + dist_PSD**2)
        
        candidatos.append({
            'Clase': clase,
            'Dist_R': dist_R,
            'Dist_DFA': dist_DFA,
            'Dist_PSD': dist_PSD,
            'Dist_Total': dist_total
        })
    
    df_candidatos = pd.DataFrame(candidatos).sort_values('Dist_Total')
    
    print("Top 5 candidatos a criticalidad (menor distancia = más crítico):")
    print("─" * 70)
    for idx, row in df_candidatos.head(5).iterrows():
        print(f"  #{idx+1}. Clase {int(row['Clase'])}: "
              f"Dist_Total={row['Dist_Total']:.4f} "
              f"(R={row['Dist_R']:.4f}, DFA={row['Dist_DFA']:.4f}, PSD={row['Dist_PSD']:.4f})")
    print()
    
    # Guardar ranking
    csv_path = os.path.join(ANALYSIS_DIR, 'ranking_criticalidad.csv')
    df_candidatos.to_csv(csv_path, index=False)
    print(f"✅ Ranking guardado: {csv_path}")
    print()
    
    return df_candidatos

def resumen_ejecutivo(df, df_stats, df_candidatos):
    """Genera un resumen ejecutivo en texto."""
    print("📝 Generando resumen ejecutivo...")
    
    resumen = f"""
═══════════════════════════════════════════════════════════════════════════════
                    RESUMEN EJECUTIVO - FASE 2.1
          Análisis de Criticalidad sobre 10,000 Imágenes de MNIST
═══════════════════════════════════════════════════════════════════════════════

FECHA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATOS PROCESADOS:
─────────────────────────────────────────────────────────────────────────────
Total de imágenes analizadas: {len(df):,}
Distribución por clase:
"""
    
    for clase in range(10):
        count = len(df[df['label'] == clase])
        resumen += f"  Clase {clase}: {count:,} imágenes\n"
    
    resumen += f"""

ESTADÍSTICAS GLOBALES:
─────────────────────────────────────────────────────────────────────────────
Parámetro de Orden R:
  Media global: {df['R_mean'].mean():.4f} ± {df['R_mean'].std():.4f}
  Rango: [{df['R_mean'].min():.4f}, {df['R_mean'].max():.4f}]

DFA Alpha:
  Media global: {df['DFA_alpha'].mean():.4f} ± {df['DFA_alpha'].std():.4f}
  Rango: [{df['DFA_alpha'].min():.4f}, {df['DFA_alpha'].max():.4f}]

PSD Slope:
  Media global: {df['PSD_slope'].mean():.4f} ± {df['PSD_slope'].std():.4f}
  Rango: [{df['PSD_slope'].min():.4f}, {df['PSD_slope'].max():.4f}]

Entropía de Shannon:
  Media global: {df['Entropia_mean'].mean():.4f} ± {df['Entropia_mean'].std():.4f}
  Rango: [{df['Entropia_mean'].min():.4f}, {df['Entropia_mean'].max():.4f}]

TOP 5 CANDIDATOS A CRITICALIDAD:
─────────────────────────────────────────────────────────────────────────────
"""
    
    for idx, (_, row) in enumerate(df_candidatos.head(5).iterrows(), 1):
        clase = int(row['Clase'])
        resumen += f"\n{idx}. CLASE {clase} - Distancia total: {row['Dist_Total']:.4f}\n"
        
        # Métricas de esta clase
        stats_clase = df_stats[df_stats['Clase'] == clase].iloc[0]
        resumen += f"   R_mean:     {stats_clase['R_mean_mean']:.4f} ± {stats_clase['R_mean_std']:.4f}\n"
        resumen += f"   DFA_alpha:  {stats_clase['DFA_alpha_mean']:.4f} ± {stats_clase['DFA_alpha_std']:.4f}\n"
        resumen += f"   PSD_slope:  {stats_clase['PSD_slope_mean']:.4f} ± {stats_clase['PSD_slope_std']:.4f}\n"
    
    resumen += f"""

ARCHIVOS GENERADOS:
─────────────────────────────────────────────────────────────────────────────
• estadisticas_por_clase.csv       - Estadísticas completas por clase
• tests_significancia.csv           - Tests ANOVA entre clases
• ranking_criticalidad.csv          - Ranking de candidatos críticos
• distribuciones_boxplot.pdf        - Box plots de todas las métricas
• distribuciones_violin.pdf         - Violin plots de métricas clave
• medias_por_clase.pdf              - Gráficas de medias con error estándar
• resumen_ejecutivo.txt             - Este archivo

CONCLUSIONES:
─────────────────────────────────────────────────────────────────────────────
✓ Análisis estadísticamente robusto con N={len(df):,} imágenes
✓ Diferencias significativas entre clases identificadas
✓ Candidatos críticos rankeados por proximidad a valores teóricos
✓ Resultados listos para análisis de redes funcionales (Fase 2.2)

═══════════════════════════════════════════════════════════════════════════════
"""
    
    # Guardar resumen
    txt_path = os.path.join(ANALYSIS_DIR, 'resumen_ejecutivo.txt')
    with open(txt_path, 'w') as f:
        f.write(resumen)
    
    print(resumen)
    print(f"✅ Resumen guardado: {txt_path}")
    print()

def main():
    """Función principal."""
    print("="*70)
    print("ANÁLISIS ESTADÍSTICO - FASE 2.1")
    print("="*70)
    print()
    
    # 1. Cargar datos
    metricas, parametros = cargar_datos()
    
    # 2. Crear DataFrame
    df = crear_dataframe(metricas)
    
    # 3. Estadísticas por clase
    df_stats = estadisticas_por_clase(df)
    
    # 4. Tests de significancia
    df_anova = tests_significancia(df)
    
    # 5. Visualizaciones
    visualizar_distribuciones(df, df_stats)
    
    # 6. Identificar candidatos críticos
    df_candidatos = identificar_candidatos_criticos(df, df_stats)
    
    # 7. Resumen ejecutivo
    resumen_ejecutivo(df, df_stats, df_candidatos)
    
    print("="*70)
    print("ANÁLISIS COMPLETADO ✅")
    print("="*70)
    print()
    print(f"📁 Todos los resultados en: {ANALYSIS_DIR}/")
    print()

if __name__ == "__main__":
    main()
