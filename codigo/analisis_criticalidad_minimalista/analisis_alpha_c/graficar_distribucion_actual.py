#!/usr/bin/env python3
"""
Script para visualizar la distribución de C_crítico en la base de datos actual
(Generará las mismas gráficas que en resultados_parciales pero desde resultados_c_critical)
"""
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import csv

# Configuración
SCRIPT_DIR = Path(__file__).parent
DB_PATH = SCRIPT_DIR / "resultados_c_critical" / "mnist_critical_tot.db"
OUTPUT_DIR = SCRIPT_DIR / "resultados_c_critical"

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


def obtener_registros_completos(db_path):
    """Devuelve una lista con (clase, image_idx, c_critical) para todas las tablas.

    Returns:
        list[dict]: cada elemento contiene claves: 'clase', 'image_idx', 'c_critical'
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    registros = []
    for clase in range(10):
        tabla = f"clase_{clase}"
        try:
            cursor.execute(f"SELECT image_idx, c_critical FROM {tabla}")
            for image_idx, c in cursor.fetchall():
                registros.append({
                    'clase': clase,
                    'image_idx': int(image_idx),
                    'c_critical': float(c)
                })
        except sqlite3.OperationalError:
            # La tabla puede no existir aún si no se ha procesado esa clase
            pass
    conn.close()
    return registros


def calcular_representantes_por_bin(registros, bins_edges):
    """Calcula un representante por bin.

    Criterio: el registro cuya `c_critical` está más cerca del centro del bin.

    Args:
        registros (list[dict]): salida de `obtener_registros_completos`.
        bins_edges (np.ndarray): bordes de bins (len = n_bins+1).

    Returns:
        list[dict]: uno por bin con: left, right, center, count, clase,
                    image_idx, c_critical, distance.
    """
    reps = []
    if len(bins_edges) < 2:
        return reps

    centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])

    # Pre-extraer valores para eficiencia
    valores = np.array([r['c_critical'] for r in registros], dtype=float)

    for i in range(len(bins_edges) - 1):
        left, right = bins_edges[i], bins_edges[i + 1]

        # Incluir el borde derecho solo en el último bin
        if i == len(bins_edges) - 2:
            mask = (valores >= left) & (valores <= right)
        else:
            mask = (valores >= left) & (valores < right)

        idxs = np.nonzero(mask)[0]

        if idxs.size == 0:
            reps.append({
                'bin_idx': i + 1,
                'left': float(left),
                'right': float(right),
                'center': float(centers[i]),
                'count': 0,
                'clase': None,
                'image_idx': None,
                'c_critical': None,
                'distance': None,
            })
            continue

        # Elegir el registro más cercano al centro del bin
        target = centers[i]
        # Distancias de los elementos del bin al centro
        dist = np.abs(valores[idxs] - target)
        best_local = idxs[np.argmin(dist)]
        best = registros[int(best_local)]
        reps.append({
            'bin_idx': i + 1,
            'left': float(left),
            'right': float(right),
            'center': float(centers[i]),
            'count': int(idxs.size),
            'clase': int(best['clase']),
            'image_idx': int(best['image_idx']),
            'c_critical': float(best['c_critical']),
            'distance': float(abs(best['c_critical'] - target)),
        })

    return reps


def guardar_representantes_csv(representantes, output_csv_path):
    """Guarda la tabla de representantes en CSV."""
    fieldnames = [
        'bin_idx', 'left', 'right', 'center', 'count',
        'clase', 'image_idx', 'c_critical', 'distance'
    ]
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in representantes:
            writer.writerow(row)


def calcular_centros_por_bin(bin_edges, counts):
    """Devuelve solo los puntos medios de cada bin y sus conteos.

    Args:
        bin_edges (np.ndarray): bordes de bins (n_bins+1)
        counts (np.ndarray): frecuencias por bin (n_bins)

    Returns:
        list[dict]: uno por bin con: bin_idx, left, right, center, count
    """
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    filas = []
    for i in range(len(centers)):
        filas.append({
            'bin_idx': i + 1,
            'left': float(bin_edges[i]),
            'right': float(bin_edges[i + 1]),
            'center': float(centers[i]),
            'count': int(counts[i]),
        })
    return filas


def guardar_centros_csv(filas, output_csv_path):
    """Guarda CSV con los puntos medios de cada bin."""
    fieldnames = ['bin_idx', 'left', 'right', 'center', 'count']
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in filas:
            writer.writerow(row)

def crear_graficas(datos_por_clase, todos_los_datos):
    """Crea visualizaciones de la distribución"""
    
    # Crear dos figuras separadas
    # FIGURA 1: Resumen general
    fig1 = plt.figure(figsize=(16, 10))
    
    # 1. Histograma general de todos los datos
    ax1 = plt.subplot(2, 2, 1)
    if todos_los_datos:
        # Usar bins explícitos para sincronizar con los representantes
        counts, bin_edges = np.histogram(todos_los_datos, bins=50)
        ax1.hist(todos_los_datos, bins=bin_edges, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(todos_los_datos), color='red', linestyle='--', 
                   label=f'Media: {np.mean(todos_los_datos):.4f}')
        ax1.axvline(np.median(todos_los_datos), color='green', linestyle='--', 
                   label=f'Mediana: {np.median(todos_los_datos):.4f}')
        ax1.set_xlabel('C_crítico', fontsize=12)
        ax1.set_ylabel('Frecuencia', fontsize=12)
        ax1.set_title(f'Distribución General (n={len(todos_los_datos)})', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
    else:
        bin_edges = None
        counts = None
    
    # 2. Box plot por clase
    ax2 = plt.subplot(2, 2, 2)
    datos_para_boxplot = [datos_por_clase[i] for i in range(10) if datos_por_clase[i]]
    etiquetas = [f'Clase {i}' for i in range(10) if datos_por_clase[i]]
    if datos_para_boxplot:
        bp = ax2.boxplot(datos_para_boxplot, tick_labels=etiquetas, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax2.set_xlabel('Clase', fontsize=12)
        ax2.set_ylabel('C_crítico', fontsize=12)
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
        ax3.set_xlabel('Clase', fontsize=12)
        ax3.set_ylabel('C_crítico', fontsize=12)
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
    
    plt.suptitle('Distribución de C_crítico por Clase - MNIST', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig1, fig2, bin_edges, counts

def main():
    print()
    print("=" * 70)
    print("GENERACIÓN DE GRÁFICAS DE DISTRIBUCIÓN DE C_CRÍTICO")
    print("=" * 70)
    print()
    
    # Verificar que existe la base de datos
    if not DB_PATH.exists():
        print(f"❌ Error: No se encontró la base de datos en {DB_PATH}")
        print(f"   Asegúrate de que el procesamiento haya comenzado.")
        return
    
    print(f"📊 Leyendo datos de: {DB_PATH.relative_to(SCRIPT_DIR)}")
    print(f"💾 Tamaño de DB: {DB_PATH.stat().st_size / (1024*1024):.2f} MB")
    print()
    
    # Obtener datos
    print("🔍 Extrayendo datos por clase:")
    print("-" * 70)
    datos_por_clase, todos_los_datos = obtener_todos_los_datos(DB_PATH)
    print("-" * 70)
    
    if not todos_los_datos:
        print()
        print("⚠️  No se encontraron datos en la base de datos")
        print("   El procesamiento puede estar iniciando. Espera unos minutos e intenta de nuevo.")
        return
    
    print()
    print(f"✅ Total de registros extraídos: {len(todos_los_datos):,}")
    print()
    
    # Crear gráficas
    print("📈 Generando visualizaciones...")
    fig1, fig2, bin_edges, counts = crear_graficas(datos_por_clase, todos_los_datos)
    
    # Guardar ambas figuras
    output_path1 = OUTPUT_DIR / "distribucion_c_critico_resumen.png"
    output_path2 = OUTPUT_DIR / "distribucion_c_critico_por_clase.png"
    
    print(f"💾 Guardando gráficas...")
    fig1.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"   ✅ {output_path1.relative_to(SCRIPT_DIR)}")
    
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"   ✅ {output_path2.relative_to(SCRIPT_DIR)}")
    print()
    
    # Mostrar estadísticas por clase
    print("=" * 70)
    print("ESTADÍSTICAS POR CLASE")
    print("=" * 70)
    print()
    print(f"{'Clase':<8} {'n':<10} {'Media':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 70)
    
    for clase in range(10):
        if datos_por_clase[clase]:
            valores = datos_por_clase[clase]
            print(f"{clase:<8} {len(valores):<10,} "
                  f"{np.mean(valores):<12.6f} "
                  f"{np.std(valores):<12.6f} "
                  f"{np.min(valores):<12.6f} "
                  f"{np.max(valores):<12.6f}")
        else:
            print(f"{clase:<8} {'0':<10} {'-':<12} {'-':<12} {'-':<12} {'-':<12}")

    # ------------------------------------------------------------------
    # Representantes por bin del histograma general
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("REPRESENTANTES POR BIN (HISTOGRAMA GENERAL)")
    print("=" * 70)

    if bin_edges is None or counts is None:
        print("⚠️  No fue posible calcular representantes (faltan bins)")
        return

    # 1) CSV preferido: puntos medios de cada bin
    centros = calcular_centros_por_bin(bin_edges, counts)
    centros_csv = OUTPUT_DIR / "representantes_por_bin_centros.csv"
    guardar_centros_csv(centros, centros_csv)
    print(f"💾 Centros de bins guardados en: {centros_csv.relative_to(SCRIPT_DIR)}")

    # 2) CSV opcional: muestra real más cercana al centro de cada bin
    registros = obtener_registros_completos(DB_PATH)
    representantes = calcular_representantes_por_bin(registros, bin_edges)

    # Guardar CSV
    reps_csv = OUTPUT_DIR / "representantes_por_bin_muestras.csv"
    guardar_representantes_csv(representantes, reps_csv)
    print(f"💾 Muestras cercanas al centro guardadas en: {reps_csv.relative_to(SCRIPT_DIR)}")

    # Mostrar un resumen compacto (primeros y últimos 3 bins con datos)
    def fmt_row(r):
        return (f"bin {r['bin_idx']:02d} | {r['left']:.4f}-{r['right']:.4f} | "
                f"n={r['count']:<5} -> "
                f"clase={r['clase']}, idx={r['image_idx']}, c={r['c_critical']}")

    print("\nResumen de centros (primeros 6):")
    for r in centros[:6]:
        print(f"  bin {r['bin_idx']:02d} | {r['left']:.4f}-{r['right']:.4f} | center={r['center']:.4f} | n={r['count']}")

    print("\nResumen de representantes por muestra:")
    mostrados = 0
    for r in representantes:
        if r['count'] > 0 and mostrados < 6:
            print("  ", fmt_row(r))
            mostrados += 1
    if len(representantes) > 6:
        print("  ...")
        for r in representantes[-3:]:
            if r['count'] > 0:
                print("  ", fmt_row(r))
    
    print("-" * 70)
    print()
    
    # Estadísticas globales
    if todos_los_datos:
        print("=" * 70)
        print("ESTADÍSTICAS GLOBALES")
        print("=" * 70)
        print(f"Total de imágenes procesadas: {len(todos_los_datos):,}")
        print(f"Media global:                  {np.mean(todos_los_datos):.6f}")
        print(f"Mediana global:                {np.median(todos_los_datos):.6f}")
        print(f"Desviación estándar:           {np.std(todos_los_datos):.6f}")
        print(f"Rango: [{np.min(todos_los_datos):.6f}, {np.max(todos_los_datos):.6f}]")
        print(f"IQR (Q3-Q1):                   {np.percentile(todos_los_datos, 75) - np.percentile(todos_los_datos, 25):.6f}")
        print("=" * 70)
    
    print()
    print("✅ Proceso completado exitosamente")
    print()
    print(f"📂 Las gráficas se encuentran en: {OUTPUT_DIR.relative_to(SCRIPT_DIR.parent)}")
    print()

if __name__ == "__main__":
    main()
