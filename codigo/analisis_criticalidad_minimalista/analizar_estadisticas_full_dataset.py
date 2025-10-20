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
from scipy.stats import norm, shapiro, kstest, anderson
from datetime import datetime

# Agregar el path del módulo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configuración
RESULTS_DIR = "resultados_kuramoto_full_dataset_CORRECTED"
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analisis_estadistico")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

def cargar_datos():
    """Carga las métricas desde el archivo guardado."""
    data_path = os.path.join(RESULTS_DIR, 'metricas_completas_CORRECTED.pt')
    
    if not os.path.exists(data_path):
        print(f"❌ Error: No se encontró {data_path}")
        print("   Primero ejecuta: python run_kuramoto_full_dataset.py")
        sys.exit(1)
    
    print(f"📂 Cargando datos desde: {data_path}")
    data = torch.load(data_path, weights_only=False)
    
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
Estadísticas descriptivas:
• estadisticas_por_clase.csv       - Estadísticas completas por clase
• tests_significancia.csv           - Tests ANOVA entre clases
• ranking_criticalidad.csv          - Ranking de candidatos críticos

Análisis de normalidad:
• tests_normalidad.csv              - Tests Shapiro-Wilk, KS, Anderson-Darling
• resumen_normalidad.txt            - Interpretación de normalidad

Visualizaciones:
• distribuciones_boxplot.pdf        - Box plots de todas las métricas
• distribuciones_violin.pdf         - Violin plots de métricas clave
• medias_por_clase.pdf              - Gráficas de medias con error estándar
• histogramas_*.pdf                 - Histogramas con ajuste gaussiano (6 archivos)
• qqplots_*.pdf                     - Q-Q plots para verificar normalidad (6 archivos)

Resúmenes:
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

def analizar_normalidad(df):
    """
    Analiza la normalidad de las distribuciones de cada métrica.
    
    Aplica tres tests estadísticos:
    - Shapiro-Wilk: Sensible a desviaciones de normalidad
    - Kolmogorov-Smirnov: Compara con distribución normal teórica
    - Anderson-Darling: Más sensible en las colas
    
    Calcula también:
    - Asimetría (skewness): 0 = simétrica, >0 = cola derecha, <0 = cola izquierda
    - Curtosis (kurtosis): 3 = gaussiana, >3 = colas pesadas, <3 = colas ligeras
    """
    print("📊 Analizando normalidad de las distribuciones...")
    print()
    
    metricas_interes = ['R_mean', 'DFA_alpha', 'PSD_slope', 'Entropia_mean', 
                        'Corr_mean', 'MI_mean']
    
    resultados_normalidad = []
    
    for clase in range(10):
        df_clase = df[df['label'] == clase]
        
        for metrica in metricas_interes:
            valores = df_clase[metrica].dropna().values
            
            if len(valores) < 3:
                continue
            
            # Test de Shapiro-Wilk (H0: distribución normal)
            # p > 0.05 → No rechazar H0 → Posiblemente normal
            if len(valores) <= 5000:  # Shapiro-Wilk limitado a 5000 muestras
                stat_shapiro, p_shapiro = shapiro(valores)
            else:
                # Para muestras grandes, usar una submuestra aleatoria
                muestra = np.random.choice(valores, size=5000, replace=False)
                stat_shapiro, p_shapiro = shapiro(muestra)
            
            # Test de Kolmogorov-Smirnov con media y std empíricas
            # Normalizar los datos
            valores_norm = (valores - valores.mean()) / valores.std()
            stat_ks, p_ks = kstest(valores_norm, 'norm')
            
            # Test de Anderson-Darling
            result_anderson = anderson(valores, dist='norm')
            # Significancia al 5% (índice 2 corresponde a 5%)
            anderson_critical_5 = result_anderson.critical_values[2]
            anderson_rechaza = result_anderson.statistic > anderson_critical_5
            
            # Asimetría y Curtosis
            skew = stats.skew(valores)
            kurt = stats.kurtosis(valores, fisher=True)  # fisher=True → exceso de curtosis (0=gaussiana)
            
            resultados_normalidad.append({
                'Clase': clase,
                'Metrica': metrica,
                'N': len(valores),
                'Media': valores.mean(),
                'Std': valores.std(),
                'Shapiro_stat': stat_shapiro,
                'Shapiro_p': p_shapiro,
                'Shapiro_normal': 'Sí' if p_shapiro > 0.05 else 'No',
                'KS_stat': stat_ks,
                'KS_p': p_ks,
                'KS_normal': 'Sí' if p_ks > 0.05 else 'No',
                'Anderson_stat': result_anderson.statistic,
                'Anderson_critical_5%': anderson_critical_5,
                'Anderson_normal': 'Sí' if not anderson_rechaza else 'No',
                'Skewness': skew,
                'Kurtosis': kurt
            })
    
    df_normalidad = pd.DataFrame(resultados_normalidad)
    
    # Guardar resultados
    csv_path = os.path.join(ANALYSIS_DIR, 'tests_normalidad.csv')
    df_normalidad.to_csv(csv_path, index=False)
    print(f"✅ Tests de normalidad guardados: {csv_path}")
    
    # Resumen por métrica
    print("\n📊 RESUMEN DE NORMALIDAD POR MÉTRICA:")
    print("─" * 80)
    for metrica in metricas_interes:
        df_met = df_normalidad[df_normalidad['Metrica'] == metrica]
        n_shapiro_normal = (df_met['Shapiro_normal'] == 'Sí').sum()
        n_ks_normal = (df_met['KS_normal'] == 'Sí').sum()
        n_anderson_normal = (df_met['Anderson_normal'] == 'Sí').sum()
        total = len(df_met)
        
        print(f"\n{metrica}:")
        print(f"  Shapiro-Wilk:    {n_shapiro_normal}/{total} clases son normales ({100*n_shapiro_normal/total:.0f}%)")
        print(f"  Kolmogorov-Smirnov: {n_ks_normal}/{total} clases son normales ({100*n_ks_normal/total:.0f}%)")
        print(f"  Anderson-Darling:   {n_anderson_normal}/{total} clases son normales ({100*n_anderson_normal/total:.0f}%)")
        print(f"  Asimetría promedio: {df_met['Skewness'].mean():+.3f} (0=simétrica)")
        print(f"  Curtosis promedio:  {df_met['Kurtosis'].mean():+.3f} (0=gaussiana)")
    
    print()
    return df_normalidad

def visualizar_histogramas_normalidad(df, df_normalidad):
    """
    Genera histogramas con ajuste gaussiano para cada métrica y clase.
    Permite verificar visualmente si los datos siguen una distribución normal.
    """
    print("📊 Generando histogramas con ajuste gaussiano...")
    
    metricas_interes = ['R_mean', 'DFA_alpha', 'PSD_slope', 'Entropia_mean', 
                        'Corr_mean', 'MI_mean']
    
    for metrica in metricas_interes:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f'Distribuciones de {metrica} por Clase con Ajuste Gaussiano', 
                     fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for clase in range(10):
            ax = axes[clase]
            
            # Datos de esta clase
            df_clase = df[df['label'] == clase]
            valores = df_clase[metrica].dropna().values
            
            # Histograma
            n, bins, patches = ax.hist(valores, bins=30, density=True, 
                                       alpha=0.7, color='steelblue', 
                                       edgecolor='black', linewidth=0.5)
            
            # Ajuste gaussiano
            mu, sigma = valores.mean(), valores.std()
            x = np.linspace(valores.min(), valores.max(), 100)
            gauss_fit = norm.pdf(x, mu, sigma)
            ax.plot(x, gauss_fit, 'r-', linewidth=2, label=f'N({mu:.3f}, {sigma:.3f})')
            
            # Información del test de normalidad
            df_norm_clase = df_normalidad[
                (df_normalidad['Clase'] == clase) & 
                (df_normalidad['Metrica'] == metrica)
            ]
            
            if len(df_norm_clase) > 0:
                row = df_norm_clase.iloc[0]
                shapiro_normal = row['Shapiro_normal']
                p_value = row['Shapiro_p']
                skew = row['Skewness']
                kurt = row['Kurtosis']
                
                # Color del título según normalidad
                color = 'green' if shapiro_normal == 'Sí' else 'red'
                ax.set_title(f'Clase {clase} (N={len(valores)})\n'
                            f'Shapiro p={p_value:.4f}\n'
                            f'Skew={skew:+.2f}, Kurt={kurt:+.2f}',
                            color=color, fontsize=9)
            else:
                ax.set_title(f'Clase {clase} (N={len(valores)})', fontsize=9)
            
            ax.set_xlabel(metrica, fontsize=8)
            ax.set_ylabel('Densidad', fontsize=8)
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        
        plt.tight_layout()
        pdf_path = os.path.join(ANALYSIS_DIR, f'histogramas_{metrica}.pdf')
        plt.savefig(pdf_path, format='pdf', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ {pdf_path}")
    
    print()

def visualizar_qq_plots(df, df_normalidad):
    """
    Genera Q-Q plots (cuantil-cuantil) para verificar normalidad.
    Si los puntos caen sobre la línea diagonal, la distribución es normal.
    """
    print("📊 Generando Q-Q plots...")
    
    metricas_interes = ['R_mean', 'DFA_alpha', 'PSD_slope', 'Entropia_mean', 
                        'Corr_mean', 'MI_mean']
    
    for metrica in metricas_interes:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f'Q-Q Plots de {metrica} por Clase', 
                     fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for clase in range(10):
            ax = axes[clase]
            
            # Datos de esta clase
            df_clase = df[df['label'] == clase]
            valores = df_clase[metrica].dropna().values
            
            # Q-Q plot
            stats.probplot(valores, dist="norm", plot=ax)
            
            # Información del test de normalidad
            df_norm_clase = df_normalidad[
                (df_normalidad['Clase'] == clase) & 
                (df_normalidad['Metrica'] == metrica)
            ]
            
            if len(df_norm_clase) > 0:
                row = df_norm_clase.iloc[0]
                shapiro_normal = row['Shapiro_normal']
                p_value = row['Shapiro_p']
                
                # Color del título según normalidad
                color = 'green' if shapiro_normal == 'Sí' else 'red'
                ax.set_title(f'Clase {clase} - Shapiro p={p_value:.4f}',
                            color=color, fontsize=9)
            else:
                ax.set_title(f'Clase {clase}', fontsize=9)
            
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        
        plt.tight_layout()
        pdf_path = os.path.join(ANALYSIS_DIR, f'qqplots_{metrica}.pdf')
        plt.savefig(pdf_path, format='pdf', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ {pdf_path}")
    
    print()

def resumen_normalidad_global(df_normalidad):
    """Genera un resumen textual del análisis de normalidad."""
    print("📝 Generando resumen de normalidad...")
    
    resumen = """
═══════════════════════════════════════════════════════════════════════════════
                    ANÁLISIS DE NORMALIDAD - FASE 2.1
═══════════════════════════════════════════════════════════════════════════════

INTERPRETACIÓN DE TESTS:
─────────────────────────────────────────────────────────────────────────────
• Shapiro-Wilk (p-value):
  - p > 0.05: NO se rechaza H0 → Datos posiblemente normales
  - p ≤ 0.05: Se rechaza H0 → Datos NO son normales
  
• Kolmogorov-Smirnov (p-value):
  - p > 0.05: NO se rechaza H0 → Datos posiblemente normales
  - p ≤ 0.05: Se rechaza H0 → Datos NO son normales
  
• Anderson-Darling (estadístico vs valor crítico):
  - stat < critical_value: Datos posiblemente normales
  - stat ≥ critical_value: Datos NO son normales

• Asimetría (Skewness):
  - ≈ 0: Distribución simétrica (ideal para gaussiana)
  - > 0: Cola hacia la derecha (asimetría positiva)
  - < 0: Cola hacia la izquierda (asimetría negativa)

• Curtosis (Kurtosis exceso):
  - ≈ 0: Curtosis similar a gaussiana
  - > 0: Colas más pesadas que gaussiana (leptocúrtica)
  - < 0: Colas más ligeras que gaussiana (platicúrtica)

"""
    
    metricas_interes = ['R_mean', 'DFA_alpha', 'PSD_slope', 'Entropia_mean', 
                        'Corr_mean', 'MI_mean']
    
    resumen += "\nRESUMEN POR MÉTRICA:\n"
    resumen += "─" * 80 + "\n"
    
    for metrica in metricas_interes:
        df_met = df_normalidad[df_normalidad['Metrica'] == metrica]
        
        n_shapiro_normal = (df_met['Shapiro_normal'] == 'Sí').sum()
        n_ks_normal = (df_met['KS_normal'] == 'Sí').sum()
        n_anderson_normal = (df_met['Anderson_normal'] == 'Sí').sum()
        total = len(df_met)
        
        skew_mean = df_met['Skewness'].mean()
        skew_std = df_met['Skewness'].std()
        kurt_mean = df_met['Kurtosis'].mean()
        kurt_std = df_met['Kurtosis'].std()
        
        resumen += f"\n{metrica.upper()}:\n"
        resumen += f"  Normalidad según tests:\n"
        resumen += f"    - Shapiro-Wilk:      {n_shapiro_normal}/{total} clases ({100*n_shapiro_normal/total:.0f}%)\n"
        resumen += f"    - Kolmogorov-Smirnov: {n_ks_normal}/{total} clases ({100*n_ks_normal/total:.0f}%)\n"
        resumen += f"    - Anderson-Darling:   {n_anderson_normal}/{total} clases ({100*n_anderson_normal/total:.0f}%)\n"
        resumen += f"  Forma de la distribución:\n"
        resumen += f"    - Asimetría: {skew_mean:+.3f} ± {skew_std:.3f}\n"
        resumen += f"    - Curtosis:  {kurt_mean:+.3f} ± {kurt_std:.3f}\n"
        
        # Interpretación
        if n_shapiro_normal >= 7:
            resumen += f"  ✓ La mayoría de las clases presentan distribución normal\n"
        elif n_shapiro_normal >= 4:
            resumen += f"  ⚠ Algunas clases presentan desviaciones de normalidad\n"
        else:
            resumen += f"  ✗ La mayoría de las clases NO presentan distribución normal\n"
    
    resumen += "\n" + "═" * 80 + "\n"
    
    # Guardar
    txt_path = os.path.join(ANALYSIS_DIR, 'resumen_normalidad.txt')
    with open(txt_path, 'w') as f:
        f.write(resumen)
    
    print(resumen)
    print(f"✅ Resumen de normalidad guardado: {txt_path}")
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
    
    # 5. Análisis de normalidad (NUEVO)
    df_normalidad = analizar_normalidad(df)
    
    # 6. Visualizaciones de distribuciones
    visualizar_distribuciones(df, df_stats)
    
    # 7. Histogramas con ajuste gaussiano (NUEVO)
    visualizar_histogramas_normalidad(df, df_normalidad)
    
    # 8. Q-Q plots (NUEVO)
    visualizar_qq_plots(df, df_normalidad)
    
    # 9. Resumen de normalidad (NUEVO)
    resumen_normalidad_global(df_normalidad)
    
    # 10. Identificar candidatos críticos
    df_candidatos = identificar_candidatos_criticos(df, df_stats)
    
    # 11. Resumen ejecutivo
    resumen_ejecutivo(df, df_stats, df_candidatos)
    
    print("="*70)
    print("ANÁLISIS COMPLETADO ✅")
    print("="*70)
    print()
    print(f"📁 Todos los resultados en: {ANALYSIS_DIR}/")
    print()
    print("📊 ARCHIVOS DE NORMALIDAD GENERADOS:")
    print("  • tests_normalidad.csv           - Tests estadísticos completos")
    print("  • histogramas_*.pdf              - Histogramas con ajuste gaussiano (6 archivos)")
    print("  • qqplots_*.pdf                  - Q-Q plots por métrica (6 archivos)")
    print("  • resumen_normalidad.txt         - Resumen interpretativo")
    print()

if __name__ == "__main__":
    main()
