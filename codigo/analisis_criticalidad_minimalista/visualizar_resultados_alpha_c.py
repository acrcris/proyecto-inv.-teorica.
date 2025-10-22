#!/usr/bin/env python3
"""
visualizar_resultados_alpha_c.py

Script para generar visualizaciones completas de los resultados
de distribuciones de α_c.

Uso:
    python visualizar_resultados_alpha_c.py --input distribuciones_alpha_c_100imgs.pt
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def cargar_resultados(filepath):
    """Carga resultados desde archivo .pt"""
    print(f"Cargando resultados desde: {filepath}")
    results = torch.load(filepath, map_location='cpu')
    
    n_clases = len(results['clases'])
    n_imgs_total = sum(len(results['clases'][c]['imgs']) for c in results['clases'])
    n_alphas = len(results['alphas'])
    
    print(f"  ✅ {n_clases} clases")
    print(f"  ✅ {n_imgs_total} imágenes totales")
    print(f"  ✅ {n_alphas} puntos α")
    print()
    
    return results


def plot_distribucion_alpha_c(results, output_dir):
    """Genera histogramas y boxplots de α_c por clase"""
    print("Generando distribuciones de α_c...")
    
    alphas_c_por_clase = {}
    for clase in results['clases'].keys():
        alphas_c_por_clase[clase] = [
            img['alpha_c'] for img in results['clases'][clase]['imgs']
        ]
    
    # Figura grande con histogramas
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(range(10))
    
    for clase in range(10):
        ax = axes[clase]
        alphas_c = alphas_c_por_clase[clase]
        stats = results['clases'][clase]['estadisticas']
        
        ax.hist(alphas_c, bins=15, color=colors[clase], alpha=0.7, edgecolor='black')
        ax.axvline(stats['alpha_c_mean'], color='red', linestyle='--', linewidth=2,
                  label=f"Media: {stats['alpha_c_mean']:.5f}")
        ax.axvline(stats['alpha_c_median'], color='green', linestyle=':', linewidth=2,
                  label=f"Mediana: {stats['alpha_c_median']:.5f}")
        
        ax.set_xlabel("α_c", fontsize=10)
        ax.set_ylabel("Frecuencia", fontsize=10)
        ax.set_title(f"Clase {clase} (n={len(alphas_c)})", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'histogramas_alpha_c.pdf', dpi=150, bbox_inches='tight')
    print("  ✅ histogramas_alpha_c.pdf")
    
    # Boxplot comparativo
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data_boxplot = [alphas_c_por_clase[c] for c in range(10)]
    bp = ax.boxplot(data_boxplot, labels=[f'C{i}' for i in range(10)],
                    patch_artist=True, showmeans=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel("Clase", fontsize=12)
    ax.set_ylabel("α_c", fontsize=12)
    ax.set_title("Distribución de α_c por clase", fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplot_alpha_c.pdf', dpi=150, bbox_inches='tight')
    print("  ✅ boxplot_alpha_c.pdf")


def plot_curvas_promedio(results, output_dir):
    """Genera curvas R(α) promedio con desviación estándar"""
    print("Generando curvas R(α) promedio...")
    
    alphas = results['alphas']
    colors = plt.cm.tab10(range(10))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for clase in range(10):
        imgs_data = results['clases'][clase]['imgs']
        
        # Apilar todas las curvas R
        Rs = np.array([img['R'] for img in imgs_data])
        R_mean = Rs.mean(axis=0)
        R_std = Rs.std(axis=0)
        
        # Curva promedio
        ax.plot(alphas, R_mean, color=colors[clase], linewidth=2, 
               label=f'Clase {clase}', alpha=0.8)
        
        # Banda de desviación estándar
        ax.fill_between(alphas, R_mean - R_std, R_mean + R_std,
                       color=colors[clase], alpha=0.2)
    
    ax.set_xlabel("α", fontsize=12)
    ax.set_ylabel("R (Parámetro de orden)", fontsize=12)
    ax.set_title("Curvas R(α) promedio por clase con ± 1σ", fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(alphas[0], alphas[-1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'curvas_R_alpha_promedio.pdf', dpi=150, bbox_inches='tight')
    print("  ✅ curvas_R_alpha_promedio.pdf")


def plot_estadisticas_clases(results, output_dir):
    """Genera gráficos de estadísticas comparativas entre clases"""
    print("Generando estadísticas comparativas...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10(range(10))
    
    # α_c medio por clase
    ax = axes[0, 0]
    alphas_c_mean = [results['clases'][c]['estadisticas']['alpha_c_mean'] for c in range(10)]
    alphas_c_std = [results['clases'][c]['estadisticas']['alpha_c_std'] for c in range(10)]
    
    bars = ax.bar(range(10), alphas_c_mean, yerr=alphas_c_std, 
                  color=colors, alpha=0.7, edgecolor='black', capsize=5)
    ax.set_xlabel("Clase", fontsize=11)
    ax.set_ylabel("α_c medio ± σ", fontsize=11)
    ax.set_title("Punto crítico medio por clase", fontsize=12, fontweight='bold')
    ax.set_xticks(range(10))
    ax.grid(True, axis='y', alpha=0.3)
    
    # R final por clase
    ax = axes[0, 1]
    R_final_mean = [results['clases'][c]['estadisticas']['R_final_mean'] for c in range(10)]
    R_final_std = [results['clases'][c]['estadisticas']['R_final_std'] for c in range(10)]
    
    bars = ax.bar(range(10), R_final_mean, yerr=R_final_std,
                  color=colors, alpha=0.7, edgecolor='black', capsize=5)
    ax.set_xlabel("Clase", fontsize=11)
    ax.set_ylabel("R(α_max) medio ± σ", fontsize=11)
    ax.set_title("Sincronización final por clase", fontsize=12, fontweight='bold')
    ax.set_xticks(range(10))
    ax.grid(True, axis='y', alpha=0.3)
    
    # Variabilidad de α_c
    ax = axes[1, 0]
    variabilidad = alphas_c_std
    
    bars = ax.bar(range(10), variabilidad, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel("Clase", fontsize=11)
    ax.set_ylabel("σ(α_c)", fontsize=11)
    ax.set_title("Variabilidad de α_c por clase", fontsize=12, fontweight='bold')
    ax.set_xticks(range(10))
    ax.grid(True, axis='y', alpha=0.3)
    
    # Rango de α_c
    ax = axes[1, 1]
    alphas_c_min = [results['clases'][c]['estadisticas']['alpha_c_min'] for c in range(10)]
    alphas_c_max = [results['clases'][c]['estadisticas']['alpha_c_max'] for c in range(10)]
    
    for i in range(10):
        ax.plot([i, i], [alphas_c_min[i], alphas_c_max[i]], 
               color=colors[i], linewidth=3, marker='o', markersize=8)
        ax.scatter([i], [alphas_c_mean[i]], color='red', s=100, zorder=5, marker='x')
    
    ax.set_xlabel("Clase", fontsize=11)
    ax.set_ylabel("Rango de α_c", fontsize=11)
    ax.set_title("Rango [min, max] y media (×) de α_c", fontsize=12, fontweight='bold')
    ax.set_xticks(range(10))
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'estadisticas_comparativas.pdf', dpi=150, bbox_inches='tight')
    print("  ✅ estadisticas_comparativas.pdf")


def generar_informe_texto(results, output_dir):
    """Genera informe de texto con estadísticas detalladas"""
    print("Generando informe de texto...")
    
    with open(output_dir / 'informe_estadisticas.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("INFORME DE ANÁLISIS DE DISTRIBUCIONES DE α_c\n")
        f.write("="*70 + "\n\n")
        
        # Metadata
        f.write("INFORMACIÓN GENERAL\n")
        f.write("-"*70 + "\n")
        f.write(f"Fecha: {results['metadata']['timestamp']}\n")
        f.write(f"Dispositivo: {results['metadata']['device']}\n")
        f.write(f"Duración: {results['metadata'].get('duration_seconds', 0) / 60:.1f} minutos\n")
        f.write(f"PyTorch: {results['metadata']['pytorch_version']}\n")
        f.write("\n")
        
        # Parámetros
        f.write("PARÁMETROS DEL MODELO\n")
        f.write("-"*70 + "\n")
        params = results['params']
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Rango de α
        f.write("RANGO DE α\n")
        f.write("-"*70 + "\n")
        alphas = results['alphas']
        f.write(f"Valores: {len(alphas)} puntos\n")
        f.write(f"Rango: [{alphas[0]:.6f}, {alphas[-1]:.6f}]\n")
        f.write(f"Paso: {alphas[1] - alphas[0]:.8f}\n")
        f.write("\n")
        
        # Estadísticas por clase
        f.write("ESTADÍSTICAS POR CLASE\n")
        f.write("="*70 + "\n\n")
        
        for clase in range(10):
            stats = results['clases'][clase]['estadisticas']
            n_imgs = len(results['clases'][clase]['imgs'])
            
            f.write(f"Clase {clase} (n={n_imgs} imágenes)\n")
            f.write("-"*70 + "\n")
            f.write(f"  α_c medio:     {stats['alpha_c_mean']:.8f}\n")
            f.write(f"  α_c std:       {stats['alpha_c_std']:.8f}\n")
            f.write(f"  α_c mediana:   {stats['alpha_c_median']:.8f}\n")
            f.write(f"  α_c min:       {stats['alpha_c_min']:.8f}\n")
            f.write(f"  α_c max:       {stats['alpha_c_max']:.8f}\n")
            f.write(f"  R(α=0):        {stats['R_inicial_mean']:.6f}\n")
            f.write(f"  R(α→∞):        {stats['R_final_mean']:.6f} ± {stats['R_final_std']:.6f}\n")
            f.write("\n")
        
        # Resumen global
        f.write("RESUMEN GLOBAL\n")
        f.write("="*70 + "\n")
        
        all_alphas_c_mean = [results['clases'][c]['estadisticas']['alpha_c_mean'] for c in range(10)]
        all_R_final = [results['clases'][c]['estadisticas']['R_final_mean'] for c in range(10)]
        
        f.write(f"α_c mínimo entre clases:  Clase {np.argmin(all_alphas_c_mean)} ({np.min(all_alphas_c_mean):.6f})\n")
        f.write(f"α_c máximo entre clases:  Clase {np.argmax(all_alphas_c_mean)} ({np.max(all_alphas_c_mean):.6f})\n")
        f.write(f"R final mínimo:           Clase {np.argmin(all_R_final)} ({np.min(all_R_final):.4f})\n")
        f.write(f"R final máximo:           Clase {np.argmax(all_R_final)} ({np.max(all_R_final):.4f})\n")
    
    print("  ✅ informe_estadisticas.txt")


def main(args):
    print("="*70)
    print("VISUALIZACIÓN DE RESULTADOS DE α_c")
    print("="*70)
    print()
    
    # Cargar resultados
    results = cargar_resultados(args.input)
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Directorio de salida: {output_dir}\n")
    
    # Generar visualizaciones
    print("="*70)
    print("GENERANDO VISUALIZACIONES")
    print("="*70)
    
    plot_distribucion_alpha_c(results, output_dir)
    plot_curvas_promedio(results, output_dir)
    plot_estadisticas_clases(results, output_dir)
    generar_informe_texto(results, output_dir)
    
    print()
    print("="*70)
    print("✅ VISUALIZACIÓN COMPLETA")
    print("="*70)
    print(f"Archivos generados en: {output_dir}")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualización de resultados de distribuciones de α_c',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', required=True,
                       help='Archivo .pt con resultados')
    parser.add_argument('--output_dir', default='plots_distribucion_alpha_c',
                       help='Directorio de salida para gráficas')
    
    args = parser.parse_args()
    main(args)
