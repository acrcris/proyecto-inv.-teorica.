"""
Visualización de resultados de alpha crítico.
Genera gráficas de distribución y curvas R(alpha) por clase.
"""
import json
import numpy as np
import os
from pathlib import Path
"""Configurar caché de Matplotlib en carpeta local y backend Agg."""
_mpl_dir = Path(__file__).parent / "_mplconfig"
try:
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))
except Exception:
    pass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse


def plot_alpha_c_distribution(results_data, output_dir):
    """
    Grafica la distribución de alpha_c por clase.
    
    Args:
        results_data: Diccionario con resultados del análisis
        output_dir: Directorio donde guardar las gráficas
    """
    results = results_data['results']
    clases = sorted([int(k) for k in results.keys()])
    
    # Extraer estadísticas por clase
    alpha_c_means = []
    alpha_c_stds = []
    alpha_c_values_all = []
    
    for clase in clases:
        clase_str = str(clase)
        alpha_c_means.append(results[clase_str]['alpha_c_mean'])
        alpha_c_stds.append(results[clase_str]['alpha_c_std'])
        alpha_c_values_all.extend(results[clase_str]['alpha_c_values'])
    
    # Crear figura con múltiples subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Barplot con medias y desviaciones estándar por clase
    ax1 = plt.subplot(2, 3, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(clases)))
    bars = ax1.bar(clases, alpha_c_means, yerr=alpha_c_stds, 
                   capsize=5, alpha=0.7, color=colors, edgecolor='black')
    ax1.set_xlabel('Clase (dígito)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('α_c promedio', fontsize=12, fontweight='bold')
    ax1.set_title('α_c promedio por clase', fontsize=14, fontweight='bold')
    ax1.set_xticks(clases)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Box plot de alpha_c por clase
    ax2 = plt.subplot(2, 3, 2)
    alpha_c_by_class = [results[str(c)]['alpha_c_values'] for c in clases]
    bp = ax2.boxplot(alpha_c_by_class, labels=clases, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_xlabel('Clase (dígito)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('α_c', fontsize=12, fontweight='bold')
    ax2.set_title('Distribución de α_c por clase', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Histograma de todos los alpha_c
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(alpha_c_values_all, bins=30, alpha=0.7, color='steelblue', 
             edgecolor='black')
    ax3.axvline(np.mean(alpha_c_values_all), color='red', linestyle='--', 
                linewidth=2, label=f'Media global: {np.mean(alpha_c_values_all):.3f}')
    ax3.set_xlabel('α_c', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
    ax3.set_title('Distribución global de α_c', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Violin plot
    ax4 = plt.subplot(2, 3, 4)
    parts = ax4.violinplot(alpha_c_by_class, positions=clases, 
                           showmeans=True, showmedians=True)
    ax4.set_xlabel('Clase (dígito)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('α_c', fontsize=12, fontweight='bold')
    ax4.set_title('Distribución tipo violín de α_c', fontsize=14, fontweight='bold')
    ax4.set_xticks(clases)
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Scatter plot de valores individuales
    ax5 = plt.subplot(2, 3, 5)
    for i, clase in enumerate(clases):
        valores = results[str(clase)]['alpha_c_values']
        x_positions = np.random.normal(clase, 0.04, size=len(valores))
        ax5.scatter(x_positions, valores, alpha=0.6, color=colors[i], s=50)
    ax5.set_xlabel('Clase (dígito)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('α_c', fontsize=12, fontweight='bold')
    ax5.set_title('Valores individuales de α_c', fontsize=14, fontweight='bold')
    ax5.set_xticks(clases)
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Comparación de desviaciones estándar
    ax6 = plt.subplot(2, 3, 6)
    ax6.bar(clases, alpha_c_stds, alpha=0.7, color=colors, edgecolor='black')
    ax6.set_xlabel('Clase (dígito)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Desviación estándar de α_c', fontsize=12, fontweight='bold')
    ax6.set_title('Variabilidad de α_c por clase', fontsize=14, fontweight='bold')
    ax6.set_xticks(clases)
    ax6.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar figura
    output_file = output_dir / 'distribucion_alpha_c.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica de distribución guardada en: {output_file}")
    plt.close()
    
    return output_file


def plot_order_curves(results_data, output_dir, max_classes=None):
    """
    Grafica las curvas R(alpha) para cada clase.
    
    Args:
        results_data: Diccionario con resultados del análisis
        output_dir: Directorio donde guardar las gráficas
        max_classes: Número máximo de clases a graficar (None = todas)
    """
    settings = results_data['settings']
    results = results_data['results']
    alphas = np.array(settings['alphas'])
    
    clases = sorted([int(k) for k in results.keys()])
    if max_classes:
        clases = clases[:max_classes]
    
    # Figura con curvas individuales
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(clases)))
    
    for idx, clase in enumerate(clases):
        ax = axes[idx]
        clase_str = str(clase)
        
        # Datos de la clase
        order_mean = np.array(results[clase_str]['order_curve_mean'])
        order_std = np.array(results[clase_str]['order_curve_std'])
        alpha_c_mean = results[clase_str]['alpha_c_mean']
        
        # Graficar curva con banda de error
        ax.plot(alphas, order_mean, linewidth=2, color=colors[idx], 
                label=f'Clase {clase}')
        ax.fill_between(alphas, order_mean - order_std, order_mean + order_std,
                        alpha=0.3, color=colors[idx])
        
        # Marcar punto crítico
        ax.axvline(alpha_c_mean, color='red', linestyle='--', linewidth=2,
                  label=f'α_c = {alpha_c_mean:.3f}')
        
        ax.set_xlabel('α (acoplamiento externo)', fontsize=10)
        ax.set_ylabel('R (parámetro de orden)', fontsize=10)
        ax.set_title(f'Clase {clase}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([alphas[0], alphas[-1]])
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Guardar figura
    output_file = output_dir / 'curvas_R_alpha_por_clase.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Curvas R(α) guardadas en: {output_file}")
    plt.close()
    
    # Figura comparativa con todas las curvas juntas
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, clase in enumerate(clases):
        clase_str = str(clase)
        order_mean = np.array(results[clase_str]['order_curve_mean'])
        alpha_c_mean = results[clase_str]['alpha_c_mean']
        
        ax.plot(alphas, order_mean, linewidth=2, color=colors[idx], 
                label=f'Clase {clase} (α_c={alpha_c_mean:.3f})', alpha=0.7)
    
    ax.set_xlabel('α (acoplamiento externo)', fontsize=14, fontweight='bold')
    ax.set_ylabel('R (parámetro de orden)', fontsize=14, fontweight='bold')
    ax.set_title('Comparación de curvas R(α) entre clases', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([alphas[0], alphas[-1]])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    output_file = output_dir / 'comparacion_curvas_R_alpha.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Comparación de curvas guardada en: {output_file}")
    plt.close()
    
    return output_file


def plot_gradient_analysis(results_data, output_dir, selected_classes=None):
    """
    Grafica el análisis de gradientes dR/dα para identificar puntos críticos.
    
    Args:
        results_data: Diccionario con resultados del análisis
        output_dir: Directorio donde guardar las gráficas
        selected_classes: Lista de clases a graficar (None = todas)
    """
    settings = results_data['settings']
    results = results_data['results']
    alphas = np.array(settings['alphas'])
    
    clases = sorted([int(k) for k in results.keys()])
    if selected_classes:
        clases = [c for c in clases if c in selected_classes]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(clases)))
    
    for idx, clase in enumerate(clases):
        ax = axes[idx]
        clase_str = str(clase)
        
        order_mean = np.array(results[clase_str]['order_curve_mean'])
        alpha_c_mean = results[clase_str]['alpha_c_mean']
        
        # Calcular gradiente
        gradient = np.gradient(order_mean, alphas)
        
        # Graficar gradiente
        ax.plot(alphas, gradient, linewidth=2, color=colors[idx])
        ax.axvline(alpha_c_mean, color='red', linestyle='--', linewidth=2,
                  label=f'α_c = {alpha_c_mean:.3f}')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        ax.set_xlabel('α', fontsize=10)
        ax.set_ylabel('dR/dα', fontsize=10)
        ax.set_title(f'Clase {clase} - Gradiente', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([alphas[0], alphas[-1]])
    
    plt.tight_layout()
    
    output_file = output_dir / 'analisis_gradientes.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Análisis de gradientes guardado en: {output_file}")
    plt.close()
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Visualizar resultados de análisis de alpha crítico'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Archivo JSON con resultados')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directorio de salida (default: mismo que input)')
    
    args = parser.parse_args()
    
    # Cargar datos
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"No se encuentra el archivo: {input_path}")
    
    print(f"\n{'='*60}")
    print(f"Visualización de resultados de α_c")
    print(f"{'='*60}\n")
    print(f"Cargando datos desde: {input_path}")
    
    with open(input_path, 'r') as f:
        results_data = json.load(f)
    
    # Directorio de salida
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent / 'graficas_alpha_c'
    
    output_dir.mkdir(exist_ok=True)
    print(f"Directorio de salida: {output_dir}\n")
    
    # Generar gráficas
    print("Generando gráficas...\n")
    
    # 1. Distribuciones
    plot_alpha_c_distribution(results_data, output_dir)
    
    # 2. Curvas R(alpha)
    plot_order_curves(results_data, output_dir)
    
    # 3. Análisis de gradientes
    plot_gradient_analysis(results_data, output_dir)
    
    print(f"\n{'='*60}")
    print(f"✓ Visualización completada")
    print(f"{'='*60}")
    print(f"\nTodas las gráficas guardadas en:")
    print(f"  {output_dir.absolute()}\n")
    print(f"Archivos generados:")
    for file in sorted(output_dir.glob('*.png')):
        print(f"  - {file.name}")
    print()


if __name__ == '__main__':
    main()
