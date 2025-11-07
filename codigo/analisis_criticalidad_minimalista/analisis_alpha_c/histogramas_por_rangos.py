#!/usr/bin/env python3
"""
Análisis detallado de histogramas por deciles y rangos
"""

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Rutas de bases de datos
db_original = Path("resultados_criticalidad.db")
db_refactorizado = Path("resultados_criticalidad_refactorizado.db")

def cargar_datos(db_path):
    """Carga α_c de la base de datos."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT alpha_c FROM resultados WHERE alpha_c IS NOT NULL")
    datos = [row[0] for row in cursor.fetchall()]
    conn.close()
    return datos

def analizar_por_rangos(datos, nombre):
    """Analiza distribución por rangos de valores."""
    rangos = [
        (0.0, 0.001, "0.000 - 0.001"),
        (0.001, 0.002, "0.001 - 0.002"),
        (0.002, 0.005, "0.002 - 0.005"),
        (0.005, 0.01, "0.005 - 0.010"),
        (0.01, 0.02, "0.010 - 0.020"),
        (0.02, 0.05, "0.020 - 0.050"),
        (0.05, 0.08, "0.050 - 0.080"),
        (0.08, 0.10, "0.080 - 0.100"),
    ]
    
    print(f"\n{'='*70}")
    print(f"ANÁLISIS POR RANGOS - {nombre}")
    print(f"{'='*70}")
    print(f"{'Rango':<20} {'Cantidad':<15} {'%':<15} {'%Acum':<15}")
    print("-" * 70)
    
    total = len(datos)
    acumulado = 0
    
    for min_val, max_val, label in rangos:
        count = sum(1 for d in datos if min_val <= d < max_val)
        porcentaje = 100 * count / total
        acumulado += porcentaje
        print(f"{label:<20} {count:<15} {porcentaje:>6.2f}%{'':<8} {acumulado:>6.2f}%")
    
    # Valores exactamente 0.1
    count_max = sum(1 for d in datos if d == 0.1)
    if count_max > 0:
        porcentaje = 100 * count_max / total
        acumulado += porcentaje
        print(f"{'0.100 (exacto)':<20} {count_max:<15} {porcentaje:>6.2f}%{'':<8} {acumulado:>6.2f}%")

def main():
    print("Cargando datos...")
    
    # Cargar datos
    datos_original = cargar_datos(db_original)
    datos_refactorizado = cargar_datos(db_refactorizado)
    
    print(f"✓ Original: {len(datos_original)} valores")
    print(f"✓ Refactorizado: {len(datos_refactorizado)} valores")
    
    # Análisis por rangos
    analizar_por_rangos(datos_original, "VERSIÓN ORIGINAL")
    analizar_por_rangos(datos_refactorizado, "VERSIÓN REFACTORIZADA")
    
    # Crear figura con histogramas por rango
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Análisis de Distribución de α_c por Rangos', fontsize=16, fontweight='bold')
    
    # Rango 1: 0 - 0.01 (zoom en valores bajos)
    ax = axes[0, 0]
    datos_orig_bajo = [d for d in datos_original if 0 <= d <= 0.01]
    datos_refact_bajo = [d for d in datos_refactorizado if 0 <= d <= 0.01]
    ax.hist(datos_orig_bajo, bins=100, alpha=0.6, label=f'Original (n={len(datos_orig_bajo)})', color='blue')
    ax.hist(datos_refact_bajo, bins=100, alpha=0.6, label=f'Refactorizado (n={len(datos_refact_bajo)})', color='red')
    ax.set_xlabel('α_c')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Rango 1: 0.000 - 0.010')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rango 2: 0.01 - 0.05
    ax = axes[0, 1]
    datos_orig_med = [d for d in datos_original if 0.01 <= d <= 0.05]
    datos_refact_med = [d for d in datos_refactorizado if 0.01 <= d <= 0.05]
    ax.hist(datos_orig_med, bins=100, alpha=0.6, label=f'Original (n={len(datos_orig_med)})', color='blue')
    ax.hist(datos_refact_med, bins=100, alpha=0.6, label=f'Refactorizado (n={len(datos_refact_med)})', color='red')
    ax.set_xlabel('α_c')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Rango 2: 0.010 - 0.050')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rango 3: 0.05 - 0.10
    ax = axes[1, 0]
    datos_orig_alto = [d for d in datos_original if 0.05 <= d <= 0.10]
    datos_refact_alto = [d for d in datos_refactorizado if 0.05 <= d <= 0.10]
    ax.hist(datos_orig_alto, bins=100, alpha=0.6, label=f'Original (n={len(datos_orig_alto)})', color='blue')
    ax.hist(datos_refact_alto, bins=100, alpha=0.6, label=f'Refactorizado (n={len(datos_refact_alto)})', color='red')
    ax.set_xlabel('α_c')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Rango 3: 0.050 - 0.100')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico de barras con conteos por rango
    ax = axes[1, 1]
    rangos = ['0-0.001', '0.001-0.002', '0.002-0.005', '0.005-0.01', 
              '0.01-0.02', '0.02-0.05', '0.05-0.08', '0.08-0.10']
    
    rangos_vals = [
        (0.0, 0.001), (0.001, 0.002), (0.002, 0.005), (0.005, 0.01),
        (0.01, 0.02), (0.02, 0.05), (0.05, 0.08), (0.08, 0.10)
    ]
    
    counts_orig = []
    counts_refact = []
    
    for min_val, max_val in rangos_vals:
        counts_orig.append(sum(1 for d in datos_original if min_val <= d < max_val))
        counts_refact.append(sum(1 for d in datos_refactorizado if min_val <= d < max_val))
    
    x = np.arange(len(rangos))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, counts_orig, width, label='Original', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, counts_refact, width, label='Refactorizado', color='red', alpha=0.7)
    
    ax.set_ylabel('Cantidad de muestras')
    ax.set_title('Distribución de muestras por rango')
    ax.set_xticks(x)
    ax.set_xticklabels(rangos, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Guardar figura
    output_path = Path("histogramas_por_rangos.png")
    print(f"\n📊 Guardando gráfico en: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado")
    
    print("\n✅ Análisis por rangos completado")

if __name__ == "__main__":
    main()
