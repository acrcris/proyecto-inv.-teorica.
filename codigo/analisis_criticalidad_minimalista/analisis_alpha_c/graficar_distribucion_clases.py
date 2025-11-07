#!/usr/bin/env python3
"""
Grafica la distribución de α_c para cada clase en ambas bases de datos
(original y refactorizada) lado a lado.
"""

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10

# Rutas de bases de datos
db_original = Path("resultados_criticalidad.db")
db_refactorizado = Path("resultados_criticalidad_refactorizado.db")

def cargar_datos(db_path):
    """Carga α_c y clase de la base de datos."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Obtener datos
    cursor.execute("SELECT clase, alpha_c FROM resultados")
    datos = cursor.fetchall()
    conn.close()
    
    return datos

def graficar_distribucion(ax, datos, titulo, num_clases=10):
    """Grafica boxplot y violinplot para cada clase."""
    
    # Organizar datos por clase
    clases_data = {i: [] for i in range(num_clases)}
    for clase, alpha_c in datos:
        if 0 <= clase < num_clases and alpha_c is not None:
            clases_data[clase].append(alpha_c)
    
    # Preparar datos para el gráfico
    plot_data = []
    labels = []
    for clase in range(num_clases):
        if clases_data[clase]:
            plot_data.append(clases_data[clase])
            labels.append(f"Clase {clase}")
    
    # Crear violinplot
    parts = ax.violinplot(plot_data, positions=range(len(plot_data)), 
                          showmeans=True, showmedians=True)
    
    # Agregar boxplot superpuesto
    bp = ax.boxplot(plot_data, positions=range(len(plot_data)), 
                     widths=0.3, patch_artist=True, 
                     boxprops=dict(facecolor='cyan', alpha=0.3),
                     medianprops=dict(color='red', linewidth=2))
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("α_c (Punto crítico)")
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Estadísticas
    stats_text = []
    for clase in range(num_clases):
        if clases_data[clase]:
            media = np.mean(clases_data[clase])
            std = np.std(clases_data[clase])
            stats_text.append(f"Clase {clase}: μ={media:.4f}, σ={std:.4f}, n={len(clases_data[clase])}")
    
    return clases_data, stats_text

def main():
    print("Cargando datos...")
    
    # Verificar que existan las bases de datos
    if not db_original.exists():
        print(f"❌ Base de datos no encontrada: {db_original}")
        return
    
    if not db_refactorizado.exists():
        print(f"⚠️ Base de datos refactorizada no encontrada: {db_refactorizado}")
        print("   Continuando solo con la base de datos original...")
        datos_refactorizado = None
    else:
        print(f"✓ Cargando {db_refactorizado}...")
        datos_refactorizado = cargar_datos(db_refactorizado)
    
    print(f"✓ Cargando {db_original}...")
    datos_original = cargar_datos(db_original)
    
    print(f"  - Original: {len(datos_original)} resultados")
    if datos_refactorizado:
        print(f"  - Refactorizado: {len(datos_refactorizado)} resultados")
    
    # Crear figura con dos subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Graficar original
    print("\nGraficando distribución original...")
    stats_original, stats_text_original = graficar_distribucion(
        axes[0], datos_original, 
        "Distribución α_c por Clase - ORIGINAL"
    )
    
    # Graficar refactorizado (si existe)
    if datos_refactorizado:
        print("Graficando distribución refactorizada...")
        stats_refactorizado, stats_text_refactorizado = graficar_distribucion(
            axes[1], datos_refactorizado,
            "Distribución α_c por Clase - REFACTORIZADO"
        )
    else:
        axes[1].text(0.5, 0.5, "Base de datos refactorizada\nno disponible",
                    ha='center', va='center', transform=axes[1].transAxes,
                    fontsize=14, color='red')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    
    plt.tight_layout()
    
    # Guardar figura
    output_path = Path("distribucion_clases.png")
    print(f"\n📊 Guardando gráfico en: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado")
    
    # Mostrar estadísticas
    print("\n" + "="*70)
    print("ESTADÍSTICAS - ORIGINAL")
    print("="*70)
    for stat in stats_text_original:
        print(stat)
    
    if datos_refactorizado:
        print("\n" + "="*70)
        print("ESTADÍSTICAS - REFACTORIZADO")
        print("="*70)
        for stat in stats_text_refactorizado:
            print(stat)
    
    # Comparación general
    print("\n" + "="*70)
    print("COMPARACIÓN GENERAL")
    print("="*70)
    
    alphas_original = [alpha_c for _, alpha_c in datos_original if alpha_c is not None]
    print(f"\nOriginal:")
    print(f"  Imágenes procesadas: {len(datos_original)}")
    print(f"  α_c promedio: {np.mean(alphas_original):.6f}")
    print(f"  α_c desv. estándar: {np.std(alphas_original):.6f}")
    print(f"  α_c mínimo: {np.min(alphas_original):.6f}")
    print(f"  α_c máximo: {np.max(alphas_original):.6f}")
    print(f"  α_c mediana: {np.median(alphas_original):.6f}")
    
    if datos_refactorizado:
        alphas_refactorizado = [alpha_c for _, alpha_c in datos_refactorizado if alpha_c is not None]
        print(f"\nRefactorizado:")
        print(f"  Imágenes procesadas: {len(datos_refactorizado)}")
        print(f"  α_c promedio: {np.mean(alphas_refactorizado):.6f}")
        print(f"  α_c desv. estándar: {np.std(alphas_refactorizado):.6f}")
        print(f"  α_c mínimo: {np.min(alphas_refactorizado):.6f}")
        print(f"  α_c máximo: {np.max(alphas_refactorizado):.6f}")
        print(f"  α_c mediana: {np.median(alphas_refactorizado):.6f}")
        
        # Diferencia
        print(f"\nDiferencia (Refactorizado - Original):")
        print(f"  Δ promedio: {np.mean(alphas_refactorizado) - np.mean(alphas_original):+.6f}")
        print(f"  Δ desv. estándar: {np.std(alphas_refactorizado) - np.std(alphas_original):+.6f}")
    
    print("\n✅ Análisis completado")
    
    # Mostrar la figura
    plt.show()

if __name__ == "__main__":
    main()
