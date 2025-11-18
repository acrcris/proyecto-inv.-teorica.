#!/usr/bin/env python3
"""
Script para visualizar la distribución de C_crítico en la base de datos combinada
"""
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuración
DB_PATH = Path(__file__).parent / "mnist_critical_tot.db"
OUTPUT_PATH = Path(__file__).parent / "distribucion_c_critico.png"

def obtener_todos_los_datos(db_path):
    """Extrae todos los valores de C_crítico de todas las clases"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    datos_por_clase = {}
    todos_los_datos = []
    
    # Obtener datos de cada clase
    for clase in range(10):
        tabla = f"clase_{clase}"
        try:
            cursor.execute(f"SELECT c_critical FROM {tabla}")
            valores = [row[0] for row in cursor.fetchall()]
            datos_por_clase[clase] = valores
            todos_los_datos.extend(valores)
            print(f"Clase {clase}: {len(valores)} registros")
        except sqlite3.OperationalError:
            print(f"Clase {clase}: No existe la tabla")
            datos_por_clase[clase] = []
    
    conn.close()
    return datos_por_clase, todos_los_datos

def crear_graficas(datos_por_clase, todos_los_datos):
    """Crea visualizaciones de la distribución"""
    
    # Crear dos figuras separadas
    # FIGURA 1: Resumen general
    fig1 = plt.figure(figsize=(16, 10))
    
    # 1. Histograma general de todos los datos
    ax1 = plt.subplot(2, 2, 1)
    if todos_los_datos:
        ax1.hist(todos_los_datos, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(todos_los_datos), color='red', linestyle='--', 
                   label=f'Media: {np.mean(todos_los_datos):.4f}')
        ax1.axvline(np.median(todos_los_datos), color='green', linestyle='--', 
                   label=f'Mediana: {np.median(todos_los_datos):.4f}')
        ax1.set_xlabel('C_crítico')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title(f'Distribución General (n={len(todos_los_datos)})', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Box plot por clase
    ax2 = plt.subplot(2, 2, 2)
    datos_para_boxplot = [datos_por_clase[i] for i in range(10) if datos_por_clase[i]]
    etiquetas = [f'Clase {i}' for i in range(10) if datos_por_clase[i]]
    if datos_para_boxplot:
        bp = ax2.boxplot(datos_para_boxplot, tick_labels=etiquetas, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax2.set_xlabel('Clase')
        ax2.set_ylabel('C_crítico')
        ax2.set_title('Distribución por Clase (Box Plot)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Violin plot
    ax3 = plt.subplot(2, 2, 3)
    if datos_para_boxplot:
        parts = ax3.violinplot(datos_para_boxplot, positions=range(len(datos_para_boxplot)),
                               showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('lightcoral')
            pc.set_alpha(0.7)
        ax3.set_xticks(range(len(etiquetas)))
        ax3.set_xticklabels(etiquetas, rotation=45)
        ax3.set_xlabel('Clase')
        ax3.set_ylabel('C_crítico')
        ax3.set_title('Distribución por Clase (Violin Plot)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Estadísticas descriptivas
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Calcular estadísticas
    if todos_los_datos:
        stats_text = "ESTADÍSTICAS GENERALES\n" + "="*40 + "\n\n"
        stats_text += f"Total de registros: {len(todos_los_datos)}\n"
        stats_text += f"Media: {np.mean(todos_los_datos):.6f}\n"
        stats_text += f"Mediana: {np.median(todos_los_datos):.6f}\n"
        stats_text += f"Desviación estándar: {np.std(todos_los_datos):.6f}\n"
        stats_text += f"Mínimo: {np.min(todos_los_datos):.6f}\n"
        stats_text += f"Máximo: {np.max(todos_los_datos):.6f}\n"
        stats_text += f"Rango: {np.max(todos_los_datos) - np.min(todos_los_datos):.6f}\n"
        stats_text += f"Q1 (25%): {np.percentile(todos_los_datos, 25):.6f}\n"
        stats_text += f"Q3 (75%): {np.percentile(todos_los_datos, 75):.6f}\n"
        stats_text += f"IQR: {np.percentile(todos_los_datos, 75) - np.percentile(todos_los_datos, 25):.6f}\n"
        
        ax4.text(0.1, 0.9, stats_text, 
                transform=ax4.transAxes,
                fontfamily='monospace',
                fontsize=11,
                verticalalignment='top')
    
    plt.tight_layout()
    
    # FIGURA 2: Histogramas por clase
    fig2 = plt.figure(figsize=(16, 8))
    colores = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i in range(10):
        ax = plt.subplot(2, 5, i + 1)
        if datos_por_clase[i]:
            ax.hist(datos_por_clase[i], bins=30, color=colores[i], 
                   alpha=0.7, edgecolor='black')
            media = np.mean(datos_por_clase[i])
            ax.axvline(media, color='red', linestyle='--', linewidth=2)
            ax.set_title(f'Clase {i} (n={len(datos_por_clase[i])})', fontsize=11, fontweight='bold')
            ax.set_xlabel('C_crítico', fontsize=9)
            ax.set_ylabel('Frecuencia', fontsize=9)
            ax.tick_params(labelsize=8)
            # Añadir texto con media
            ax.text(0.95, 0.95, f'μ={media:.3f}', 
                   transform=ax.transAxes, 
                   verticalalignment='top',
                   horizontalalignment='right',
                   fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Sin datos', 
                   transform=ax.transAxes,
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=12,
                   color='red')
            ax.set_title(f'Clase {i}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig1, fig2

def main():
    print("=" * 60)
    print("VISUALIZACIÓN DE LA DISTRIBUCIÓN DE C_CRÍTICO")
    print("=" * 60)
    print()
    
    # Verificar que existe la base de datos
    if not DB_PATH.exists():
        print(f"❌ Error: No se encontró la base de datos en {DB_PATH}")
        return
    
    print(f"📊 Leyendo datos de: {DB_PATH.name}")
    print()
    
    # Obtener datos
    datos_por_clase, todos_los_datos = obtener_todos_los_datos(DB_PATH)
    
    if not todos_los_datos:
        print("❌ No se encontraron datos en la base de datos")
        return
    
    print()
    print(f"✓ Total de registros: {len(todos_los_datos)}")
    print()
    
    # Crear gráficas
    print("📈 Generando visualizaciones...")
    fig1, fig2 = crear_graficas(datos_por_clase, todos_los_datos)
    
    # Guardar ambas figuras
    output_path1 = OUTPUT_PATH.parent / "distribucion_c_critico_resumen.png"
    output_path2 = OUTPUT_PATH.parent / "distribucion_c_critico_por_clase.png"
    
    fig1.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica resumen guardada en: {output_path1.name}")
    
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica por clase guardada en: {output_path2.name}")
    print()
    
    # Mostrar estadísticas por clase
    print("ESTADÍSTICAS POR CLASE:")
    print("-" * 60)
    print(f"{'Clase':<8} {'n':<8} {'Media':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 60)
    
    for clase in range(10):
        if datos_por_clase[clase]:
            valores = datos_por_clase[clase]
            print(f"{clase:<8} {len(valores):<8} "
                  f"{np.mean(valores):<10.4f} "
                  f"{np.std(valores):<10.4f} "
                  f"{np.min(valores):<10.4f} "
                  f"{np.max(valores):<10.4f}")
        else:
            print(f"{clase:<8} {'0':<8} {'-':<10} {'-':<10} {'-':<10} {'-':<10}")
    
    print("-" * 60)
    print()
    print("✅ Proceso completado")

if __name__ == "__main__":
    main()
