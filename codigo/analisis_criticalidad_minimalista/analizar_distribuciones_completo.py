"""
Análisis completo de distribuciones de métricas de criticalidad por clase.
Genera un informe detallado con estadísticas, tests de normalidad y comparaciones.
"""

import torch
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURACIÓN
# ==========================================
RUTA_METRICAS = Path("resultados_kuramoto_TRAIN_MAC_60k/metricas_completas_TRAIN_MAC_60k.pt")
RUTA_SALIDA = Path("resultados_kuramoto_TRAIN_MAC_60k/analisis_distribuciones")
RUTA_SALIDA.mkdir(exist_ok=True)

# ==========================================
# CARGA DE DATOS
# ==========================================
print("=" * 80)
print("ANÁLISIS DE DISTRIBUCIONES DE MÉTRICAS DE CRITICALIDAD")
print("=" * 80)
print(f"\n📂 Cargando datos desde: {RUTA_METRICAS}")

datos_completos = torch.load(RUTA_METRICAS, weights_only=False)
datos = datos_completos['metricas']  # Extraer la lista de métricas
print(f"✅ Dataset cargado: {len(datos)} imágenes")

# ==========================================
# EXTRACCIÓN DE MÉTRICAS
# ==========================================
print("\n📊 Extrayendo métricas por clase...")

metricas_por_clase = {i: {
    'R_stationary': [],
    'mag_channel_0': [],
    'mag_channel_1': [],
    'mag_channel_2': [],
    'mag_channel_3': [],
    'mag_channel_mean': [],
    'PSD_slope': [],
    'DFA_alpha': [],
    'mutual_info': [],
    'shannon_entropy_0': [],
    'shannon_entropy_1': [],
    'shannon_entropy_2': [],
    'shannon_entropy_3': [],
    'shannon_entropy_mean': []
} for i in range(10)}

for img_data in datos:
    clase = img_data['label']  # La clave es 'label' no 'clase'
    
    # Función auxiliar para convertir a float
    def to_float(val):
        if isinstance(val, (int, float)):
            return float(val)
        return val.item() if hasattr(val, 'item') else float(val)
    
    # R estacionario (valor final)
    metricas_por_clase[clase]['R_stationary'].append(
        to_float(img_data['R_series'][-1])
    )
    
    # Magnitud por canal (valores finales) - ya viene como tensor de 4 valores
    mag_final = img_data['mag_channel_mean_final']
    for ch in range(4):
        metricas_por_clase[clase][f'mag_channel_{ch}'].append(to_float(mag_final[ch]))
    metricas_por_clase[clase]['mag_channel_mean'].append(to_float(mag_final.mean()))
    
    # PSD slope
    metricas_por_clase[clase]['PSD_slope'].append(
        to_float(img_data['PSD_slope'])
    )
    
    # DFA alpha
    metricas_por_clase[clase]['DFA_alpha'].append(
        to_float(img_data['DFA_alpha'])
    )
    
    # Mutual Information
    metricas_por_clase[clase]['mutual_info'].append(
        to_float(img_data['mutual_info'])
    )
    
    # Entropía de Shannon por canal (valores finales) - ya viene como tensor de 4 valores
    entropy_final = img_data['shannon_entropy_by_channel']
    for ch in range(4):
        metricas_por_clase[clase][f'shannon_entropy_{ch}'].append(to_float(entropy_final[ch]))
    metricas_por_clase[clase]['shannon_entropy_mean'].append(to_float(entropy_final.mean()))

# Convertir a arrays numpy
for clase in range(10):
    for metrica in metricas_por_clase[clase]:
        metricas_por_clase[clase][metrica] = np.array(metricas_por_clase[clase][metrica])

print(f"✅ Métricas extraídas para {len(metricas_por_clase)} clases")

# ==========================================
# FUNCIONES DE ANÁLISIS
# ==========================================

def calcular_estadisticas_descriptivas(datos):
    """Calcula estadísticas descriptivas completas."""
    return {
        'n': len(datos),
        'media': np.mean(datos),
        'mediana': np.median(datos),
        'std': np.std(datos, ddof=1),
        'var': np.var(datos, ddof=1),
        'min': np.min(datos),
        'max': np.max(datos),
        'rango': np.max(datos) - np.min(datos),
        'q1': np.percentile(datos, 25),
        'q3': np.percentile(datos, 75),
        'iqr': np.percentile(datos, 75) - np.percentile(datos, 25),
        'cv': np.std(datos, ddof=1) / np.mean(datos) if np.mean(datos) != 0 else np.nan,
        'skewness': stats.skew(datos),
        'kurtosis': stats.kurtosis(datos)
    }

def test_normalidad(datos):
    """Realiza tests de normalidad."""
    resultados = {}
    
    # Shapiro-Wilk (max 5000 muestras por limitaciones del test)
    if len(datos) <= 5000:
        stat_sw, p_sw = stats.shapiro(datos)
        resultados['shapiro_wilk'] = {'statistic': stat_sw, 'p_value': p_sw}
    else:
        # Usar muestra aleatoria de 5000
        muestra = np.random.choice(datos, 5000, replace=False)
        stat_sw, p_sw = stats.shapiro(muestra)
        resultados['shapiro_wilk'] = {'statistic': stat_sw, 'p_value': p_sw, 'nota': 'muestra de 5000'}
    
    # Anderson-Darling
    result_ad = stats.anderson(datos, dist='norm')
    resultados['anderson_darling'] = {
        'statistic': result_ad.statistic,
        'critical_values': result_ad.critical_values,
        'significance_levels': result_ad.significance_level
    }
    
    # Kolmogorov-Smirnov
    stat_ks, p_ks = stats.kstest(datos, 'norm', args=(np.mean(datos), np.std(datos, ddof=1)))
    resultados['kolmogorov_smirnov'] = {'statistic': stat_ks, 'p_value': p_ks}
    
    # D'Agostino-Pearson
    stat_dp, p_dp = stats.normaltest(datos)
    resultados['dagostino_pearson'] = {'statistic': stat_dp, 'p_value': p_dp}
    
    return resultados

def interpretar_normalidad(tests):
    """Interpreta los resultados de los tests de normalidad."""
    alpha = 0.05
    resultados_interpretados = []
    
    # Shapiro-Wilk
    if 'shapiro_wilk' in tests:
        es_normal = tests['shapiro_wilk']['p_value'] > alpha
        resultados_interpretados.append(f"Shapiro-Wilk: {'NORMAL' if es_normal else 'NO NORMAL'} (p={tests['shapiro_wilk']['p_value']:.4f})")
    
    # Anderson-Darling
    if 'anderson_darling' in tests:
        stat = tests['anderson_darling']['statistic']
        crit_5 = tests['anderson_darling']['critical_values'][2]  # 5% significance
        es_normal = stat < crit_5
        resultados_interpretados.append(f"Anderson-Darling: {'NORMAL' if es_normal else 'NO NORMAL'} (stat={stat:.4f}, crit_5%={crit_5:.4f})")
    
    # Kolmogorov-Smirnov
    if 'kolmogorov_smirnov' in tests:
        es_normal = tests['kolmogorov_smirnov']['p_value'] > alpha
        resultados_interpretados.append(f"Kolmogorov-Smirnov: {'NORMAL' if es_normal else 'NO NORMAL'} (p={tests['kolmogorov_smirnov']['p_value']:.4f})")
    
    # D'Agostino-Pearson
    if 'dagostino_pearson' in tests:
        es_normal = tests['dagostino_pearson']['p_value'] > alpha
        resultados_interpretados.append(f"D'Agostino-Pearson: {'NORMAL' if es_normal else 'NO NORMAL'} (p={tests['dagostino_pearson']['p_value']:.4f})")
    
    return resultados_interpretados

def analizar_criticalidad(metrica_nombre, valor_media, valor_std):
    """Analiza si los valores están en rango de criticalidad."""
    interpretacion = ""
    
    if metrica_nombre == 'PSD_slope':
        # Criticalidad: pendiente cercana a -1
        if -1.2 <= valor_media <= -0.8:
            interpretacion = "✅ EN RANGO DE CRITICALIDAD (-1.2 a -0.8)"
        elif -1.5 <= valor_media <= -0.5:
            interpretacion = "⚠️  CERCANO A CRITICALIDAD"
        else:
            interpretacion = "❌ FUERA DE RANGO DE CRITICALIDAD"
    
    elif metrica_nombre == 'DFA_alpha':
        # Criticalidad: DFA entre 0.5 y 1.0
        if 0.5 <= valor_media <= 1.0:
            interpretacion = "✅ EN RANGO DE CRITICALIDAD (0.5 a 1.0)"
        elif 0.3 <= valor_media <= 1.2:
            interpretacion = "⚠️  CERCANO A CRITICALIDAD"
        else:
            interpretacion = "❌ FUERA DE RANGO DE CRITICALIDAD"
    
    elif metrica_nombre == 'R_stationary':
        # Criticalidad: sincronización intermedia (0.3-0.7)
        if 0.3 <= valor_media <= 0.7:
            interpretacion = "✅ SINCRONIZACIÓN INTERMEDIA (criticalidad)"
        elif valor_media < 0.3:
            interpretacion = "❌ SINCRONIZACIÓN BAJA (fase desordenada)"
        else:
            interpretacion = "❌ SINCRONIZACIÓN ALTA (fase ordenada)"
    
    return interpretacion

# ==========================================
# ANÁLISIS POR MÉTRICA Y CLASE
# ==========================================

# Definir métricas principales para análisis detallado
metricas_principales = {
    'R_stationary': 'R (Parámetro de Orden) - Valor Estacionario',
    'mag_channel_mean': 'Magnitud Media por Canal',
    'PSD_slope': 'PSD Slope (Pendiente del Espectro de Potencia)',
    'DFA_alpha': 'DFA Alpha (Exponente de Fluctuaciones)',
    'mutual_info': 'Mutual Information (Información Mutua)',
    'shannon_entropy_mean': 'Entropía de Shannon (Media por Canal)'
}

# También analizar canales individuales
metricas_canales = {
    'mag_channel_0': 'Magnitud Canal 0',
    'mag_channel_1': 'Magnitud Canal 1',
    'mag_channel_2': 'Magnitud Canal 2',
    'mag_channel_3': 'Magnitud Canal 3',
    'shannon_entropy_0': 'Entropía Shannon Canal 0',
    'shannon_entropy_1': 'Entropía Shannon Canal 1',
    'shannon_entropy_2': 'Entropía Shannon Canal 2',
    'shannon_entropy_3': 'Entropía Shannon Canal 3'
}

print("\n" + "=" * 80)
print("ANÁLISIS DETALLADO POR MÉTRICA Y CLASE")
print("=" * 80)

# ==========================================
# GENERAR INFORME COMPLETO
# ==========================================

informe_completo = []
informe_completo.append("=" * 100)
informe_completo.append("INFORME COMPLETO: ANÁLISIS DE DISTRIBUCIONES DE MÉTRICAS DE CRITICALIDAD")
informe_completo.append("=" * 100)
informe_completo.append(f"\nDataset: {len(datos)} imágenes MNIST (Training Set)")
informe_completo.append(f"Clases analizadas: 0-9 (10 dígitos)")
informe_completo.append(f"Métricas principales: {len(metricas_principales)}")
informe_completo.append(f"Métricas por canal: {len(metricas_canales)}")
informe_completo.append("")

# ==========================================
# ANÁLISIS DE MÉTRICAS PRINCIPALES
# ==========================================

for metrica_key, metrica_nombre in metricas_principales.items():
    informe_completo.append("\n" + "=" * 100)
    informe_completo.append(f"MÉTRICA: {metrica_nombre}")
    informe_completo.append("=" * 100)
    
    # Tabla resumen por clase
    tabla_resumen = []
    tabla_resumen.append("\n" + "-" * 100)
    tabla_resumen.append(f"{'Clase':<8} {'N':<8} {'Media':<12} {'Mediana':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Skew':<12} {'Kurt':<12}")
    tabla_resumen.append("-" * 100)
    
    for clase in range(10):
        datos_clase = metricas_por_clase[clase][metrica_key]
        stats_desc = calcular_estadisticas_descriptivas(datos_clase)
        
        tabla_resumen.append(
            f"{clase:<8} {stats_desc['n']:<8} {stats_desc['media']:<12.4f} {stats_desc['mediana']:<12.4f} "
            f"{stats_desc['std']:<12.4f} {stats_desc['min']:<12.4f} {stats_desc['max']:<12.4f} "
            f"{stats_desc['skewness']:<12.4f} {stats_desc['kurtosis']:<12.4f}"
        )
    
    tabla_resumen.append("-" * 100)
    informe_completo.extend(tabla_resumen)
    
    # Análisis detallado por clase
    for clase in range(10):
        datos_clase = metricas_por_clase[clase][metrica_key]
        stats_desc = calcular_estadisticas_descriptivas(datos_clase)
        tests_norm = test_normalidad(datos_clase)
        interpretaciones_norm = interpretar_normalidad(tests_norm)
        
        informe_completo.append(f"\n{'─' * 100}")
        informe_completo.append(f"CLASE {clase} - {metrica_nombre}")
        informe_completo.append(f"{'─' * 100}")
        
        # Estadísticas descriptivas
        informe_completo.append("\n📊 ESTADÍSTICAS DESCRIPTIVAS:")
        informe_completo.append(f"  • Tamaño muestral: {stats_desc['n']}")
        informe_completo.append(f"  • Media: {stats_desc['media']:.6f}")
        informe_completo.append(f"  • Mediana: {stats_desc['mediana']:.6f}")
        informe_completo.append(f"  • Desviación estándar: {stats_desc['std']:.6f}")
        informe_completo.append(f"  • Varianza: {stats_desc['var']:.6f}")
        informe_completo.append(f"  • Coeficiente de variación: {stats_desc['cv']:.4f}")
        informe_completo.append(f"  • Rango: [{stats_desc['min']:.6f}, {stats_desc['max']:.6f}]")
        informe_completo.append(f"  • Amplitud: {stats_desc['rango']:.6f}")
        informe_completo.append(f"  • Cuartiles: Q1={stats_desc['q1']:.6f}, Q2={stats_desc['mediana']:.6f}, Q3={stats_desc['q3']:.6f}")
        informe_completo.append(f"  • Rango intercuartílico (IQR): {stats_desc['iqr']:.6f}")
        informe_completo.append(f"  • Asimetría (Skewness): {stats_desc['skewness']:.6f}")
        informe_completo.append(f"  • Curtosis (Kurtosis): {stats_desc['kurtosis']:.6f}")
        
        # Tests de normalidad
        informe_completo.append("\n🔬 TESTS DE NORMALIDAD:")
        for interpretacion in interpretaciones_norm:
            informe_completo.append(f"  • {interpretacion}")
        
        # Análisis de criticalidad
        interpretacion_crit = analizar_criticalidad(metrica_key, stats_desc['media'], stats_desc['std'])
        if interpretacion_crit:
            informe_completo.append(f"\n🎯 ANÁLISIS DE CRITICALIDAD:")
            informe_completo.append(f"  {interpretacion_crit}")

# ==========================================
# ANÁLISIS DE CANALES INDIVIDUALES
# ==========================================

informe_completo.append("\n\n" + "=" * 100)
informe_completo.append("ANÁLISIS DE CANALES INDIVIDUALES")
informe_completo.append("=" * 100)

for metrica_key, metrica_nombre in metricas_canales.items():
    informe_completo.append(f"\n{'─' * 100}")
    informe_completo.append(f"MÉTRICA: {metrica_nombre}")
    informe_completo.append(f"{'─' * 100}")
    
    # Tabla resumen por clase
    tabla_resumen = []
    tabla_resumen.append(f"\n{'Clase':<8} {'Media':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    tabla_resumen.append("-" * 60)
    
    for clase in range(10):
        datos_clase = metricas_por_clase[clase][metrica_key]
        stats_desc = calcular_estadisticas_descriptivas(datos_clase)
        
        tabla_resumen.append(
            f"{clase:<8} {stats_desc['media']:<12.4f} {stats_desc['std']:<12.4f} "
            f"{stats_desc['min']:<12.4f} {stats_desc['max']:<12.4f}"
        )
    
    tabla_resumen.append("-" * 60)
    informe_completo.extend(tabla_resumen)

# ==========================================
# COMPARACIONES ENTRE CLASES
# ==========================================

informe_completo.append("\n\n" + "=" * 100)
informe_completo.append("COMPARACIONES ENTRE CLASES (ANOVA y Kruskal-Wallis)")
informe_completo.append("=" * 100)

for metrica_key, metrica_nombre in metricas_principales.items():
    informe_completo.append(f"\n{'─' * 100}")
    informe_completo.append(f"MÉTRICA: {metrica_nombre}")
    informe_completo.append(f"{'─' * 100}")
    
    # Recopilar datos de todas las clases
    datos_todas_clases = [metricas_por_clase[clase][metrica_key] for clase in range(10)]
    
    # ANOVA (asume normalidad)
    f_stat, p_anova = stats.f_oneway(*datos_todas_clases)
    informe_completo.append(f"\n  ANOVA (paramétrico):")
    informe_completo.append(f"    • F-statistic: {f_stat:.4f}")
    informe_completo.append(f"    • p-value: {p_anova:.6f}")
    informe_completo.append(f"    • Resultado: {'Las medias son SIGNIFICATIVAMENTE DIFERENTES' if p_anova < 0.05 else 'No hay diferencia significativa'} (α=0.05)")
    
    # Kruskal-Wallis (no paramétrico, no asume normalidad)
    h_stat, p_kw = stats.kruskal(*datos_todas_clases)
    informe_completo.append(f"\n  Kruskal-Wallis (no paramétrico):")
    informe_completo.append(f"    • H-statistic: {h_stat:.4f}")
    informe_completo.append(f"    • p-value: {p_kw:.6f}")
    informe_completo.append(f"    • Resultado: {'Las distribuciones son SIGNIFICATIVAMENTE DIFERENTES' if p_kw < 0.05 else 'No hay diferencia significativa'} (α=0.05)")

# ==========================================
# RESUMEN EJECUTIVO
# ==========================================

informe_completo.insert(7, "\n" + "=" * 100)
informe_completo.insert(8, "RESUMEN EJECUTIVO")
informe_completo.insert(9, "=" * 100)

resumen_lineas = []

for metrica_key, metrica_nombre in metricas_principales.items():
    # Calcular estadísticas globales
    todos_valores = np.concatenate([metricas_por_clase[clase][metrica_key] for clase in range(10)])
    media_global = np.mean(todos_valores)
    std_global = np.std(todos_valores, ddof=1)
    
    # Encontrar clase con mayor y menor media
    medias_por_clase = [np.mean(metricas_por_clase[clase][metrica_key]) for clase in range(10)]
    clase_max = np.argmax(medias_por_clase)
    clase_min = np.argmin(medias_por_clase)
    
    resumen_lineas.append(f"\n📊 {metrica_nombre}:")
    resumen_lineas.append(f"  • Media global: {media_global:.4f} ± {std_global:.4f}")
    resumen_lineas.append(f"  • Clase con mayor media: {clase_max} ({medias_por_clase[clase_max]:.4f})")
    resumen_lineas.append(f"  • Clase con menor media: {clase_min} ({medias_por_clase[clase_min]:.4f})")
    
    # Análisis de criticalidad global
    interpretacion_crit = analizar_criticalidad(metrica_key, media_global, std_global)
    if interpretacion_crit:
        resumen_lineas.append(f"  • {interpretacion_crit}")

# Insertar resumen después del header
for i, linea in enumerate(resumen_lineas):
    informe_completo.insert(10 + i, linea)

# ==========================================
# GUARDAR INFORME
# ==========================================

print("\n" + "=" * 80)
print("GUARDANDO RESULTADOS")
print("=" * 80)

# Guardar informe de texto
archivo_informe = RUTA_SALIDA / "informe_completo_distribuciones.txt"
with open(archivo_informe, 'w', encoding='utf-8') as f:
    f.write('\n'.join(informe_completo))
print(f"✅ Informe completo guardado: {archivo_informe}")

# Guardar tablas en CSV para análisis adicional
print("\n📄 Generando archivos CSV...")

# CSV 1: Estadísticas descriptivas por clase y métrica
datos_csv_desc = []
for metrica_key, metrica_nombre in {**metricas_principales, **metricas_canales}.items():
    for clase in range(10):
        datos_clase = metricas_por_clase[clase][metrica_key]
        stats_desc = calcular_estadisticas_descriptivas(datos_clase)
        datos_csv_desc.append({
            'metrica': metrica_key,
            'clase': clase,
            'n': stats_desc['n'],
            'media': stats_desc['media'],
            'mediana': stats_desc['mediana'],
            'std': stats_desc['std'],
            'var': stats_desc['var'],
            'cv': stats_desc['cv'],
            'min': stats_desc['min'],
            'max': stats_desc['max'],
            'rango': stats_desc['rango'],
            'q1': stats_desc['q1'],
            'q3': stats_desc['q3'],
            'iqr': stats_desc['iqr'],
            'skewness': stats_desc['skewness'],
            'kurtosis': stats_desc['kurtosis']
        })

df_desc = pd.DataFrame(datos_csv_desc)
archivo_csv_desc = RUTA_SALIDA / "estadisticas_descriptivas_por_clase.csv"
df_desc.to_csv(archivo_csv_desc, index=False)
print(f"✅ Estadísticas descriptivas guardadas: {archivo_csv_desc}")

# CSV 2: Tests de normalidad
datos_csv_norm = []
for metrica_key, metrica_nombre in metricas_principales.items():
    for clase in range(10):
        datos_clase = metricas_por_clase[clase][metrica_key]
        tests_norm = test_normalidad(datos_clase)
        
        datos_csv_norm.append({
            'metrica': metrica_key,
            'clase': clase,
            'shapiro_wilk_stat': tests_norm['shapiro_wilk']['statistic'],
            'shapiro_wilk_p': tests_norm['shapiro_wilk']['p_value'],
            'anderson_darling_stat': tests_norm['anderson_darling']['statistic'],
            'anderson_darling_crit_5pct': tests_norm['anderson_darling']['critical_values'][2],
            'kolmogorov_smirnov_stat': tests_norm['kolmogorov_smirnov']['statistic'],
            'kolmogorov_smirnov_p': tests_norm['kolmogorov_smirnov']['p_value'],
            'dagostino_pearson_stat': tests_norm['dagostino_pearson']['statistic'],
            'dagostino_pearson_p': tests_norm['dagostino_pearson']['p_value']
        })

df_norm = pd.DataFrame(datos_csv_norm)
archivo_csv_norm = RUTA_SALIDA / "tests_normalidad_por_clase.csv"
df_norm.to_csv(archivo_csv_norm, index=False)
print(f"✅ Tests de normalidad guardados: {archivo_csv_norm}")

# CSV 3: Comparaciones entre clases (ANOVA y Kruskal-Wallis)
datos_csv_comp = []
for metrica_key, metrica_nombre in metricas_principales.items():
    datos_todas_clases = [metricas_por_clase[clase][metrica_key] for clase in range(10)]
    f_stat, p_anova = stats.f_oneway(*datos_todas_clases)
    h_stat, p_kw = stats.kruskal(*datos_todas_clases)
    
    datos_csv_comp.append({
        'metrica': metrica_key,
        'anova_f_stat': f_stat,
        'anova_p_value': p_anova,
        'anova_significativo': p_anova < 0.05,
        'kruskal_wallis_h_stat': h_stat,
        'kruskal_wallis_p_value': p_kw,
        'kruskal_wallis_significativo': p_kw < 0.05
    })

df_comp = pd.DataFrame(datos_csv_comp)
archivo_csv_comp = RUTA_SALIDA / "comparaciones_entre_clases.csv"
df_comp.to_csv(archivo_csv_comp, index=False)
print(f"✅ Comparaciones entre clases guardadas: {archivo_csv_comp}")

# ==========================================
# RESUMEN FINAL
# ==========================================

print("\n" + "=" * 80)
print("✅ ANÁLISIS COMPLETADO")
print("=" * 80)
print(f"\n📂 Directorio de salida: {RUTA_SALIDA}")
print(f"\n📄 Archivos generados:")
print(f"  1. {archivo_informe.name} - Informe completo en texto")
print(f"  2. {archivo_csv_desc.name} - Estadísticas descriptivas (CSV)")
print(f"  3. {archivo_csv_norm.name} - Tests de normalidad (CSV)")
print(f"  4. {archivo_csv_comp.name} - Comparaciones entre clases (CSV)")
print(f"\n📊 Métricas analizadas:")
print(f"  • {len(metricas_principales)} métricas principales")
print(f"  • {len(metricas_canales)} métricas por canal")
print(f"  • 10 clases (dígitos 0-9)")
print(f"  • {len(datos)} imágenes totales")
print("\n✨ ¡Análisis completo finalizado!")
