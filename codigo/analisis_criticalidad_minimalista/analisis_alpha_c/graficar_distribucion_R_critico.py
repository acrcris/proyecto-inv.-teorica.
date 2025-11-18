#!/usr/bin/env python3
"""Script para graficar la distribución de R_critico desde la base de datos SQLite."""

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuración
DB_PATH = Path(__file__).parent / 'resultados_c_critical' / 'mnist_R_critico.db'
OUTPUT_DIR = Path(__file__).parent / 'resultados_c_critical'

def extraer_datos_R_critico(db_path):
    """Extrae todos los valores de R_critico de la base de datos."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    R_global = []
    R_por_clase = {}
    
    for clase in range(10):
        try:
            cursor.execute(f'SELECT R_critico FROM clase_{clase}')
            valores = [row[0] for row in cursor.fetchall()]
            R_por_clase[clase] = np.array(valores)
            R_global.extend(valores)
        except sqlite3.OperationalError:
            R_por_clase[clase] = np.array([])
    
    conn.close()
    return np.array(R_global), R_por_clase


def graficar_distribucion_total(R_global, output_path):
    """Genera histograma de la distribución total de R_critico."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histograma
    n, bins, patches = ax.hist(R_global, bins=50, alpha=0.7, color='steelblue', 
                                edgecolor='black', linewidth=0.5)
    
    # Estadísticas
    media = np.mean(R_global)
    mediana = np.median(R_global)
    std = np.std(R_global)
    
    # Líneas de referencia
    ax.axvline(media, color='red', linestyle='--', linewidth=2, 
               label=f'Media: {media:.4f}')
    ax.axvline(mediana, color='green', linestyle='--', linewidth=2, 
               label=f'Mediana: {mediana:.4f}')
    
    # Etiquetas y título
    ax.set_xlabel('R en criticalidad (C = 0.1769)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=14, fontweight='bold')
    ax.set_title(f'Distribución de R en punto de criticalidad\nMNIST completo (N = {len(R_global):,})', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Leyenda con estadísticas
    stats_text = f'μ = {media:.4f}\nσ = {std:.4f}\nmin = {R_global.min():.4f}\nmax = {R_global.max():.4f}'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=11, family='monospace')
    
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico guardado: {output_path}")
    plt.close()


def graficar_distribucion_por_clase(R_por_clase, output_path):
    """Genera histogramas por clase en una cuadrícula."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for clase in range(10):
        ax = axes[clase]
        valores = R_por_clase[clase]
        
        if len(valores) > 0:
            ax.hist(valores, bins=30, alpha=0.7, color=f'C{clase}', 
                   edgecolor='black', linewidth=0.5)
            
            media = np.mean(valores)
            ax.axvline(media, color='red', linestyle='--', linewidth=1.5)
            
            ax.set_title(f'Clase {clase}\nN={len(valores)}, μ={media:.4f}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('R_critico', fontsize=10)
            ax.set_ylabel('Frecuencia', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        else:
            ax.text(0.5, 0.5, f'Clase {clase}\nSin datos', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
    
    plt.suptitle('Distribución de R en criticalidad por clase de MNIST', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico guardado: {output_path}")
    plt.close()


def imprimir_estadisticas(R_global, R_por_clase):
    """Imprime estadísticas detalladas."""
    print("\n" + "="*60)
    print("ESTADÍSTICAS DE R_CRITICO")
    print("="*60)
    
    print(f"\n📊 DISTRIBUCIÓN GLOBAL:")
    print(f"  Total de imágenes: {len(R_global):,}")
    print(f"  Media: {np.mean(R_global):.6f}")
    print(f"  Mediana: {np.median(R_global):.6f}")
    print(f"  Desviación estándar: {np.std(R_global):.6f}")
    print(f"  Mínimo: {R_global.min():.6f}")
    print(f"  Máximo: {R_global.max():.6f}")
    print(f"  Rango: {R_global.max() - R_global.min():.6f}")
    
    print(f"\n📊 ESTADÍSTICAS POR CLASE:")
    print(f"{'Clase':<8} {'N':<8} {'Media':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 60)
    
    for clase in range(10):
        valores = R_por_clase[clase]
        if len(valores) > 0:
            print(f"{clase:<8} {len(valores):<8} {np.mean(valores):<12.6f} "
                  f"{np.std(valores):<12.6f} {valores.min():<12.6f} {valores.max():<12.6f}")


def main():
    print("📊 Generando visualizaciones de R_critico...")
    print(f"📂 Base de datos: {DB_PATH}")
    
    # Extraer datos
    R_global, R_por_clase = extraer_datos_R_critico(DB_PATH)
    
    if len(R_global) == 0:
        print("❌ No se encontraron datos en la base de datos.")
        return
    
    # Imprimir estadísticas
    imprimir_estadisticas(R_global, R_por_clase)
    
    # Generar gráficos
    print("\n📈 Generando gráficos...")
    
    output_total = OUTPUT_DIR / 'distribucion_R_critico_total.png'
    graficar_distribucion_total(R_global, output_total)
    
    output_clases = OUTPUT_DIR / 'distribucion_R_critico_por_clase.png'
    graficar_distribucion_por_clase(R_por_clase, output_clases)
    
    print("\n✨ ¡Visualizaciones completadas!")
    print(f"📁 Archivos guardados en: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
