#!/usr/bin/env python3
"""
Genera histogramas comparativos de α_c para ambas versiones (original y refactorizada)
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

def main():
    print("Cargando datos...")
    
    # Cargar datos
    datos_original = cargar_datos(db_original)
    datos_refactorizado = cargar_datos(db_refactorizado)
    
    print(f"✓ Original: {len(datos_original)} valores")
    print(f"✓ Refactorizado: {len(datos_refactorizado)} valores")
    
    # Crear figura con múltiples subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Histogramas superpuestos (escala lineal)
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(datos_original, bins=100, alpha=0.6, label='Original', color='blue', density=True)
    ax1.hist(datos_refactorizado, bins=100, alpha=0.6, label='Refactorizado', color='red', density=True)
    ax1.set_xlabel('α_c')
    ax1.set_ylabel('Densidad')
    ax1.set_title('Histogramas Superpuestos (Escala Lineal)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogramas lado a lado - Original
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(datos_original, bins=150, color='blue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('α_c')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title(f'Original (n={len(datos_original)})')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.text(0.98, 0.97, f"μ = {np.mean(datos_original):.6f}\nσ = {np.std(datos_original):.6f}\nmediana = {np.median(datos_original):.6f}",
             transform=ax2.transAxes, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
    
    # 3. Histogramas lado a lado - Refactorizado
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(datos_refactorizado, bins=150, color='red', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('α_c')
    ax3.set_ylabel('Frecuencia')
    ax3.set_title(f'Refactorizado (n={len(datos_refactorizado)})')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.text(0.98, 0.97, f"μ = {np.mean(datos_refactorizado):.6f}\nσ = {np.std(datos_refactorizado):.6f}\nmediana = {np.median(datos_refactorizado):.6f}",
             transform=ax3.transAxes, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
    
    # 4. Histogramas en escala logarítmica
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(datos_original, bins=100, alpha=0.6, label='Original', color='blue', density=True)
    ax4.hist(datos_refactorizado, bins=100, alpha=0.6, label='Refactorizado', color='red', density=True)
    ax4.set_xlabel('α_c')
    ax4.set_ylabel('Densidad (escala log)')
    ax4.set_title('Histogramas Superpuestos (Escala Log Y)')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')
    
    # 5. Cumulative Distribution Function (CDF)
    ax5 = plt.subplot(2, 3, 5)
    sorted_orig = np.sort(datos_original)
    sorted_refact = np.sort(datos_refactorizado)
    ax5.plot(sorted_orig, np.linspace(0, 1, len(sorted_orig)), label='Original', color='blue', linewidth=2)
    ax5.plot(sorted_refact, np.linspace(0, 1, len(sorted_refact)), label='Refactorizado', color='red', linewidth=2)
    ax5.set_xlabel('α_c')
    ax5.set_ylabel('CDF (Probabilidad acumulada)')
    ax5.set_title('Función de Distribución Acumulada')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Box plot comparativo
    ax6 = plt.subplot(2, 3, 6)
    bp = ax6.boxplot([datos_original, datos_refactorizado], 
                      labels=['Original', 'Refactorizado'],
                      patch_artist=True,
                      widths=0.6)
    
    # Colorear boxes
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax6.set_ylabel('α_c')
    ax6.set_title('Comparación de Distribuciones (Box Plot)')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Agregar estadísticas en el gráfico
    stats_text = f"""
    ORIGINAL:
    • Mín: {np.min(datos_original):.6f}
    • Q1: {np.percentile(datos_original, 25):.6f}
    • Mediana: {np.median(datos_original):.6f}
    • Q3: {np.percentile(datos_original, 75):.6f}
    • Máx: {np.max(datos_original):.6f}
    
    REFACTORIZADO:
    • Mín: {np.min(datos_refactorizado):.6f}
    • Q1: {np.percentile(datos_refactorizado, 25):.6f}
    • Mediana: {np.median(datos_refactorizado):.6f}
    • Q3: {np.percentile(datos_refactorizado, 75):.6f}
    • Máx: {np.max(datos_refactorizado):.6f}
    """
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    
    # Guardar figura
    output_path = Path("histogramas_comparativos.png")
    print(f"\n📊 Guardando gráfico en: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado")
    
    # Estadísticas detalladas
    print("\n" + "="*70)
    print("ESTADÍSTICAS DETALLADAS")
    print("="*70)
    
    print("\nVERSIÓN ORIGINAL:")
    print(f"  Muestras: {len(datos_original)}")
    print(f"  Promedio: {np.mean(datos_original):.6f}")
    print(f"  Desv. Std: {np.std(datos_original):.6f}")
    print(f"  Mínimo: {np.min(datos_original):.6f}")
    print(f"  Q1 (25%): {np.percentile(datos_original, 25):.6f}")
    print(f"  Mediana: {np.median(datos_original):.6f}")
    print(f"  Q3 (75%): {np.percentile(datos_original, 75):.6f}")
    print(f"  Máximo: {np.max(datos_original):.6f}")
    print(f"  IQR: {np.percentile(datos_original, 75) - np.percentile(datos_original, 25):.6f}")
    print(f"  Asimetría (skewness): {(np.mean(datos_original) - np.median(datos_original)) / np.std(datos_original):.4f}")
    
    print("\nVERSIÓN REFACTORIZADA:")
    print(f"  Muestras: {len(datos_refactorizado)}")
    print(f"  Promedio: {np.mean(datos_refactorizado):.6f}")
    print(f"  Desv. Std: {np.std(datos_refactorizado):.6f}")
    print(f"  Mínimo: {np.min(datos_refactorizado):.6f}")
    print(f"  Q1 (25%): {np.percentile(datos_refactorizado, 25):.6f}")
    print(f"  Mediana: {np.median(datos_refactorizado):.6f}")
    print(f"  Q3 (75%): {np.percentile(datos_refactorizado, 75):.6f}")
    print(f"  Máximo: {np.max(datos_refactorizado):.6f}")
    print(f"  IQR: {np.percentile(datos_refactorizado, 75) - np.percentile(datos_refactorizado, 25):.6f}")
    print(f"  Asimetría (skewness): {(np.mean(datos_refactorizado) - np.median(datos_refactorizado)) / np.std(datos_refactorizado):.4f}")
    
    print("\n" + "="*70)
    print("PERCENTILES")
    print("="*70)
    
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n{'Percentil':<12} {'Original':<15} {'Refactorizado':<15} {'Diferencia':<15}")
    print("-" * 57)
    for p in percentiles:
        val_orig = np.percentile(datos_original, p)
        val_refact = np.percentile(datos_refactorizado, p)
        diff = val_refact - val_orig
        print(f"{p}%{'':<9} {val_orig:<15.6f} {val_refact:<15.6f} {diff:+.6f}")
    
    print("\n✅ Análisis completado")

if __name__ == "__main__":
    main()
