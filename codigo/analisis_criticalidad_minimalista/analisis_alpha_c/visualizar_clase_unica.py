"""
Visualización de resultados de alpha crítico para UNA SOLA CLASE.
Genera gráficas específicas para análisis detallado.
"""
import json
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from utils import setup_matplotlib, save_figure

# Configurar matplotlib al inicio
setup_matplotlib()


def plot_single_class_analysis(results_data, output_dir):
    """
    Grafica análisis detallado para una sola clase.
    """
    settings = results_data['settings']
    results = results_data['results']
    
    target_class = settings['target_class']
    alphas = np.array(settings['alphas'])
    alpha_c_values = np.array(results['alpha_c_values'])
    order_mean = np.array(results['order_curve_mean'])
    order_std = np.array(results['order_curve_std'])
    alpha_c_mean = results['alpha_c_mean']
    alpha_c_std = results['alpha_c_std']
    
    # Crear figura con 4 subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Curva R(α) promedio con banda de error
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(alphas, order_mean, linewidth=3, color='steelblue', label='R(α) promedio')
    ax1.fill_between(alphas, order_mean - order_std, order_mean + order_std,
                     alpha=0.3, color='steelblue', label='± 1 std')
    ax1.axvline(alpha_c_mean, color='red', linestyle='--', linewidth=2.5,
               label=f'α_c = {alpha_c_mean:.3f} ± {alpha_c_std:.3f}')
    ax1.set_xlabel('α (acoplamiento externo)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('R (parámetro de orden)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Curva R(α) - Clase {target_class}', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([alphas[0], alphas[-1]])
    ax1.set_ylim([0, 1])
    
    # 2. Análisis de gradiente dR/dα
    ax2 = plt.subplot(2, 2, 2)
    gradient = np.gradient(order_mean, alphas)
    ax2.plot(alphas, gradient, linewidth=2.5, color='darkgreen', label='dR/dα')
    ax2.axvline(alpha_c_mean, color='red', linestyle='--', linewidth=2.5,
               label=f'α_c (máx. gradiente)')
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
    ax2.set_xlabel('α (acoplamiento externo)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('dR/dα', fontsize=13, fontweight='bold')
    ax2.set_title('Gradiente de la transición', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([alphas[0], alphas[-1]])
    
    # 3. Distribución de α_c individuales
    ax3 = plt.subplot(2, 2, 3)
    n_images = len(alpha_c_values)
    colors = plt.cm.viridis(np.linspace(0, 1, n_images))
    
    # Barplot
    bars = ax3.bar(range(1, n_images + 1), alpha_c_values, color=colors, 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.axhline(alpha_c_mean, color='red', linestyle='--', linewidth=2.5,
               label=f'Promedio: {alpha_c_mean:.3f}')
    ax3.axhline(alpha_c_mean + alpha_c_std, color='orange', linestyle=':', 
               linewidth=2, alpha=0.7, label=f'± std: {alpha_c_std:.3f}')
    ax3.axhline(alpha_c_mean - alpha_c_std, color='orange', linestyle=':', 
               linewidth=2, alpha=0.7)
    
    ax3.set_xlabel('Imagen #', fontsize=13, fontweight='bold')
    ax3.set_ylabel('α_c', fontsize=13, fontweight='bold')
    ax3.set_title(f'α_c por imagen - Clase {target_class}', fontsize=15, fontweight='bold')
    ax3.set_xticks(range(1, n_images + 1))
    ax3.legend(fontsize=11)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Histograma y estadísticas
    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(alpha_c_values, bins=15, alpha=0.7, color='purple', 
            edgecolor='black', linewidth=1.5)
    ax4.axvline(alpha_c_mean, color='red', linestyle='--', linewidth=2.5,
               label=f'Media: {alpha_c_mean:.3f}')
    ax4.axvline(results['alpha_c_median'], color='blue', linestyle='-.', 
               linewidth=2.5, label=f'Mediana: {results["alpha_c_median"]:.3f}')
    
    ax4.set_xlabel('α_c', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Frecuencia', fontsize=13, fontweight='bold')
    ax4.set_title('Distribución de α_c', fontsize=15, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(axis='y', alpha=0.3)
    
    # Añadir texto con estadísticas
    stats_text = f'n = {n_images}\n'
    stats_text += f'μ = {alpha_c_mean:.3f}\n'
    stats_text += f'σ = {alpha_c_std:.3f}\n'
    stats_text += f'min = {results["alpha_c_min"]:.3f}\n'
    stats_text += f'max = {results["alpha_c_max"]:.3f}'
    ax4.text(0.98, 0.97, stats_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Guardar
    output_file = output_dir / f'analisis_clase_{target_class}_optimizado.png'
    save_figure(fig, output_file)
    print(f"✓ Análisis detallado guardado en: {output_file}")
    plt.close()
    
    return output_file


def plot_parameter_summary(results_data, output_dir):
    """
    Crea un resumen visual de los parámetros utilizados.
    """
    settings = results_data['settings']
    results = results_data['results']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    target_class = settings['target_class']
    
    # Título
    title_text = f"Análisis de α_c - Clase {target_class}\n"
    title_text += f"Configuración de parámetros\n"
    ax.text(0.5, 0.95, title_text, ha='center', va='top', fontsize=18, 
           fontweight='bold', transform=ax.transAxes)
    
    # Parámetros del modelo
    params_text = "PARÁMETROS DEL MODELO KURAMOTO:\n\n"
    params_text += f"  • T_steps (timesteps):         {settings['timesteps']}\n"
    params_text += f"  • γ (gamma):                   {settings['gamma']}\n"
    params_text += f"  • Δt (delta_t):                {settings['delta_t']}\n"
    params_text += f"  • ksize (ventana conv):        {settings['ksize']}\n"
    params_text += f"  • init_omg (freq. naturales):  {settings['init_omg']}\n"
    params_text += f"  • canales:                     {settings['channels']}\n"
    params_text += f"  • ventana promedio R(t):       {settings['window']}\n"
    params_text += f"  • tamaño imagen:               {settings['img_size']}×{settings['img_size']}\n\n"
    
    # Barrido de alpha
    params_text += "BARRIDO DE α:\n\n"
    params_text += f"  • rango: [{settings['alphas'][0]}, {settings['alphas'][-1]}]\n"
    params_text += f"  • paso: {settings['alphas'][1] - settings['alphas'][0]:.3f}\n"
    params_text += f"  • puntos evaluados: {len(settings['alphas'])}\n\n"
    
    # Resultados
    params_text += "RESULTADOS:\n\n"
    params_text += f"  • Número de imágenes:    {settings['num_images']}\n"
    params_text += f"  • α_c promedio:          {results['alpha_c_mean']:.3f} ± {results['alpha_c_std']:.3f}\n"
    params_text += f"  • α_c mínimo:            {results['alpha_c_min']:.3f}\n"
    params_text += f"  • α_c máximo:            {results['alpha_c_max']:.3f}\n"
    params_text += f"  • α_c mediana:           {results['alpha_c_median']:.3f}\n"
    
    ax.text(0.1, 0.85, params_text, ha='left', va='top', fontsize=12, 
           family='monospace', transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    output_file = output_dir / f'parametros_clase_{settings["target_class"]}.png'
    save_figure(fig, output_file)
    print(f"✓ Resumen de parámetros guardado en: {output_file}")
    plt.close()
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Visualizar resultados de alpha crítico para una clase'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Archivo JSON con resultados')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directorio de salida')
    
    args = parser.parse_args()
    
    # Cargar datos
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"No se encuentra: {input_path}")
    
    print(f"\n{'='*70}")
    print(f"Visualización de resultados")
    print(f"{'='*70}\n")
    print(f"Cargando: {input_path.name}")
    
    with open(input_path, 'r') as f:
        results_data = json.load(f)
    
    # Directorio de salida
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        target_class = results_data['settings']['target_class']
        output_dir = input_path.parent / f'graficas_clase_{target_class}_optimizado'
    
    output_dir.mkdir(exist_ok=True)
    print(f"Guardando en: {output_dir}\n")
    
    # Generar gráficas
    print("Generando visualizaciones...\n")
    plot_single_class_analysis(results_data, output_dir)
    plot_parameter_summary(results_data, output_dir)
    
    print(f"\n{'='*70}")
    print(f"✓ Completado")
    print(f"{'='*70}\n")
    print(f"Archivos generados en: {output_dir.absolute()}")
    for file in sorted(output_dir.glob('*.png')):
        print(f"  • {file.name}")
    print()


if __name__ == '__main__':
    main()
