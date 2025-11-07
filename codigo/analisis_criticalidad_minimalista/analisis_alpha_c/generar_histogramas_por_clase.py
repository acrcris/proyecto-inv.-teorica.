#!/usr/bin/env python3
"""
Genera histogramas individuales de α_c para cada clase (0-9)
en versiones separadas: normal y refactorizado
"""

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Rutas de bases de datos
db_original = Path("resultados_criticalidad.db")
db_refactorizado = Path("resultados_criticalidad_refactorizado.db")

# Rutas de salida
output_normal = Path("graficas/analisis_normal")
output_refactorizado = Path("graficas/analisis_refactorizado")

def cargar_datos_por_clase(db_path):
    """Carga α_c por clase de la base de datos."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    datos_por_clase = {i: [] for i in range(10)}
    
    cursor.execute("SELECT clase, alpha_c FROM resultados WHERE alpha_c IS NOT NULL")
    for clase, alpha_c in cursor.fetchall():
        if 0 <= clase < 10:
            datos_por_clase[clase].append(alpha_c)
    
    conn.close()
    return datos_por_clase

def generar_histogramas(datos_por_clase, output_dir, nombre_version):
    """Genera un histograma por clase."""
    
    print(f"\n📊 Generando histogramas para {nombre_version}...")
    
    for clase in range(10):
        datos = datos_por_clase[clase]
        
        if not datos:
            print(f"  ⚠️ Clase {clase}: Sin datos")
            continue
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Histograma
        n, bins, patches = ax.hist(datos, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
        
        # Estadísticas
        media = np.mean(datos)
        mediana = np.median(datos)
        std = np.std(datos)
        q1 = np.percentile(datos, 25)
        q3 = np.percentile(datos, 75)
        min_val = np.min(datos)
        max_val = np.max(datos)
        
        # Líneas verticales para media y mediana
        ax.axvline(media, color='red', linestyle='--', linewidth=2, label=f'Media: {media:.6f}')
        ax.axvline(mediana, color='green', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.6f}')
        
        # Configurar gráfico
        ax.set_xlabel('α_c (Punto Crítico)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
        ax.set_title(f'Clase {clase} - Distribución de α_c\n{nombre_version}', 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Caja de estadísticas
        stats_text = f"""Estadísticas:
n = {len(datos)}
μ = {media:.6f}
σ = {std:.6f}
Mín = {min_val:.6f}
Q1 = {q1:.6f}
Med = {mediana:.6f}
Q3 = {q3:.6f}
Máx = {max_val:.6f}"""
        
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                family='monospace')
        
        plt.tight_layout()
        
        # Guardar figura
        output_file = output_dir / f"clase_{clase:02d}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Clase {clase}: {output_file.name} (n={len(datos)})")

def main():
    print("="*70)
    print("GENERACIÓN DE HISTOGRAMAS POR CLASE")
    print("="*70)
    
    # Cargar datos
    print("\nCargando datos...")
    datos_normal = cargar_datos_por_clase(db_original)
    datos_refactorizado = cargar_datos_por_clase(db_refactorizado)
    
    print(f"✓ Análisis Normal: {sum(len(v) for v in datos_normal.values())} valores totales")
    print(f"✓ Análisis Refactorizado: {sum(len(v) for v in datos_refactorizado.values())} valores totales")
    
    # Generar histogramas
    generar_histogramas(datos_normal, output_normal, "Análisis Normal")
    generar_histogramas(datos_refactorizado, output_refactorizado, "Análisis Refactorizado")
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN POR CLASE")
    print("="*70)
    
    print("\nANÁLISIS NORMAL:")
    print(f"{'Clase':<8} {'n':<8} {'Media':<12} {'Desv. Std':<12} {'Mediana':<12}")
    print("-" * 52)
    for clase in range(10):
        datos = datos_normal[clase]
        if datos:
            print(f"{clase:<8} {len(datos):<8} {np.mean(datos):<12.6f} {np.std(datos):<12.6f} {np.median(datos):<12.6f}")
    
    print("\nANÁLISIS REFACTORIZADO:")
    print(f"{'Clase':<8} {'n':<8} {'Media':<12} {'Desv. Std':<12} {'Mediana':<12}")
    print("-" * 52)
    for clase in range(10):
        datos = datos_refactorizado[clase]
        if datos:
            print(f"{clase:<8} {len(datos):<8} {np.mean(datos):<12.6f} {np.std(datos):<12.6f} {np.median(datos):<12.6f}")
    
    # Comparación
    print("\n" + "="*70)
    print("DIFERENCIA (Refactorizado - Normal)")
    print("="*70)
    print(f"{'Clase':<8} {'Δ Media':<15} {'Δ Desv. Std':<15} {'Δ Mediana':<15}")
    print("-" * 53)
    for clase in range(10):
        datos_n = datos_normal[clase]
        datos_r = datos_refactorizado[clase]
        if datos_n and datos_r:
            delta_media = np.mean(datos_r) - np.mean(datos_n)
            delta_std = np.std(datos_r) - np.std(datos_n)
            delta_med = np.median(datos_r) - np.median(datos_n)
            print(f"{clase:<8} {delta_media:+15.6f} {delta_std:+15.6f} {delta_med:+15.6f}")
    
    print("\n✅ Histogramas generados exitosamente")
    print(f"\n📁 Ubicaciones:")
    print(f"   Normal: {output_normal}")
    print(f"   Refactorizado: {output_refactorizado}")

if __name__ == "__main__":
    main()
