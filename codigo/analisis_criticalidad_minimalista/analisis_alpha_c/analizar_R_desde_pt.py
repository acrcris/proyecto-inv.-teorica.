#!/usr/bin/env python3
"""Script para extraer y graficar la distribución de R desde metricas_completas_TRAIN_MAC_60k.pt"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuración
PT_PATH = Path('/Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/analisis_criticalidad_minimalista/resultados_kuramoto_TRAIN_MAC_60k/metricas_completas_TRAIN_MAC_60k.pt')
OUTPUT_DIR = PT_PATH.parent / 'analisis_distribuciones'

def cargar_datos(pt_path):
    """Carga el archivo .pt y extrae valores de R."""
    print(f"📂 Cargando datos desde: {pt_path}")
    
    datos = torch.load(pt_path, map_location='cpu', weights_only=False)
    print(f"✅ Datos cargados. Claves disponibles: {list(datos.keys())}")
    
    return datos


def extraer_R_por_clase(datos):
    """Extrae valores de R organizados por clase."""
    R_global = []
    R_por_clase = {}
    
    # Explorar estructura del diccionario
    if 'metricas' in datos:
        metricas = datos['metricas']
        print(f"\n🔍 Estructura de 'metricas': {type(metricas)}")
        
        if isinstance(metricas, list):
            print(f"  Longitud de la lista: {len(metricas)}")
            if len(metricas) > 0:
                print(f"  Tipo del primer elemento: {type(metricas[0])}")
                print(f"  Claves del primer elemento: {list(metricas[0].keys()) if isinstance(metricas[0], dict) else 'No es dict'}")
            
            # Recorrer la lista de métricas
            for idx, item in enumerate(metricas):
                if isinstance(item, dict):
                    # Buscar R_final en el diccionario
                    if 'R_final' in item:
                        R_value = item['R_final']
                        if isinstance(R_value, (float, int)):
                            R_global.append(R_value)
                        elif isinstance(R_value, (torch.Tensor, np.ndarray)):
                            R_global.extend(R_value.flatten().tolist())
                    
                    # Organizar por clase si existe
                    if 'label' in item or 'clase' in item:
                        clase = item.get('label', item.get('clase'))
                        R_value = item.get('R_final')
                        
                        if clase not in R_por_clase:
                            R_por_clase[clase] = []
                        
                        if isinstance(R_value, (float, int)):
                            R_por_clase[clase].append(R_value)
                        elif isinstance(R_value, (torch.Tensor, np.ndarray)):
                            R_por_clase[clase].extend(R_value.flatten().tolist())
            
            print(f"  ✅ Extraídos {len(R_global)} valores de R")
            if R_por_clase:
                print(f"  ✅ Organizados en {len(R_por_clase)} clases")
        
        elif isinstance(metricas, dict):
            for clase, valores in metricas.items():
                if isinstance(valores, dict) and 'R_final' in valores:
                    R_values = valores['R_final']
                    if isinstance(R_values, torch.Tensor):
                        R_values = R_values.cpu().numpy()
                    elif isinstance(R_values, list):
                        R_values = np.array(R_values)
                    
                    R_por_clase[clase] = R_values
                    R_global.extend(R_values.flatten())
                    print(f"  Clase {clase}: {len(R_values)} valores")
    
    # Convertir R_por_clase a numpy arrays
    for clase in R_por_clase:
        R_por_clase[clase] = np.array(R_por_clase[clase])
    
    return np.array(R_global), R_por_clase


def graficar_distribucion_total(R_global, output_path):
    """Genera histograma de la distribución total de R."""
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
    ax.set_xlabel('Parámetro de orden R', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=14, fontweight='bold')
    ax.set_title(f'Distribución de R - TRAIN_MAC_60k\n(N = {len(R_global):,})', 
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
    """Genera histogramas por clase."""
    if not R_por_clase:
        print("⚠️  No hay datos por clase para graficar")
        return
    
    n_clases = len(R_por_clase)
    ncols = 5
    nrows = (n_clases + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4*nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, (clase, valores) in enumerate(sorted(R_por_clase.items())):
        ax = axes[idx]
        
        if len(valores) > 0:
            ax.hist(valores, bins=30, alpha=0.7, color=f'C{idx}', 
                   edgecolor='black', linewidth=0.5)
            
            media = np.mean(valores)
            ax.axvline(media, color='red', linestyle='--', linewidth=1.5)
            
            ax.set_title(f'Clase {clase}\nN={len(valores)}, μ={media:.4f}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('R', fontsize=10)
            ax.set_ylabel('Frecuencia', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        else:
            ax.text(0.5, 0.5, f'Clase {clase}\nSin datos', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
    
    # Ocultar ejes sobrantes
    for idx in range(len(R_por_clase), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Distribución de R por clase - TRAIN_MAC_60k', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico guardado: {output_path}")
    plt.close()


def imprimir_estadisticas(R_global, R_por_clase):
    """Imprime estadísticas detalladas."""
    print("\n" + "="*60)
    print("ESTADÍSTICAS DE R - TRAIN_MAC_60k")
    print("="*60)
    
    print(f"\n📊 DISTRIBUCIÓN GLOBAL:")
    print(f"  Total de valores: {len(R_global):,}")
    print(f"  Media: {np.mean(R_global):.6f}")
    print(f"  Mediana: {np.median(R_global):.6f}")
    print(f"  Desviación estándar: {np.std(R_global):.6f}")
    print(f"  Mínimo: {R_global.min():.6f}")
    print(f"  Máximo: {R_global.max():.6f}")
    print(f"  Rango: {R_global.max() - R_global.min():.6f}")
    
    if R_por_clase:
        print(f"\n📊 ESTADÍSTICAS POR CLASE:")
        print(f"{'Clase':<8} {'N':<8} {'Media':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-" * 60)
        
        for clase in sorted(R_por_clase.keys()):
            valores = R_por_clase[clase]
            if len(valores) > 0:
                print(f"{clase:<8} {len(valores):<8} {np.mean(valores):<12.6f} "
                      f"{np.std(valores):<12.6f} {valores.min():<12.6f} {valores.max():<12.6f}")


def main():
    print("📊 Analizando distribución de R desde metricas_completas_TRAIN_MAC_60k.pt")
    
    # Crear directorio de salida
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Cargar datos
    datos = cargar_datos(PT_PATH)
    
    # Extraer valores de R
    R_global, R_por_clase = extraer_R_por_clase(datos)
    
    if len(R_global) == 0:
        print("❌ No se encontraron valores de R en el archivo.")
        print("\n🔍 Estructura completa del archivo:")
        for key, value in datos.items():
            print(f"  {key}: {type(value)}")
            if isinstance(value, dict):
                for subkey in list(value.keys())[:5]:
                    print(f"    └─ {subkey}: {type(value[subkey])}")
        return
    
    # Imprimir estadísticas
    imprimir_estadisticas(R_global, R_por_clase)
    
    # Generar gráficos
    print("\n📈 Generando gráficos...")
    
    output_total = OUTPUT_DIR / 'distribucion_R_total_TRAIN_MAC_60k.png'
    graficar_distribucion_total(R_global, output_total)
    
    if R_por_clase:
        output_clases = OUTPUT_DIR / 'distribucion_R_por_clase_TRAIN_MAC_60k.png'
        graficar_distribucion_por_clase(R_por_clase, output_clases)
    
    print("\n✨ ¡Análisis completado!")
    print(f"📁 Archivos guardados en: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
