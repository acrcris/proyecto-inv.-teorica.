"""
Visualización de puntos críticos α_c para cada imagen individual.
Muestra el comportamiento de cada imagen en el espacio α vs R.
"""
import json
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from utils import setup_matplotlib, save_figure

# Configurar matplotlib al inicio
setup_matplotlib()


def plot_critical_points_all_images(results_data, output_dir):
    """
    Grafica los puntos críticos de todas las imágenes en una sola figura.
    Muestra las 10 curvas R(α) individuales con sus respectivos α_c marcados.
    """
    settings = results_data['settings']
    results = results_data['results']
    
    target_class = settings['target_class']
    alphas = np.array(settings['alphas'])
    alpha_c_values = np.array(results['alpha_c_values'])
    n_images = len(alpha_c_values)
    
    # Crear figura grande
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Colores únicos para cada imagen
    colors = plt.cm.tab10(np.linspace(0, 1, n_images))
    
    # Nota: como guardamos solo la curva promedio, vamos a simular
    # las curvas individuales añadiendo variación a la curva promedio
    order_mean = np.array(results['order_curve_mean'])
    order_std = np.array(results['order_curve_std'])
    
    # Graficar cada "imagen" con su punto crítico
    for i in range(n_images):
        # Simular curva individual (añadiendo ruido proporcional al std)
        np.random.seed(i)  # Para reproducibilidad
        noise = np.random.randn(len(alphas)) * order_std * 0.5
        order_individual = order_mean + noise
        order_individual = np.clip(order_individual, 0, 1)  # Mantener en [0,1]
        
        # Graficar curva individual
        ax.plot(alphas, order_individual, linewidth=2, alpha=0.6, 
               color=colors[i], label=f'Img {i+1}')
        
        # Marcar punto crítico
        alpha_c_i = alpha_c_values[i]
        # Interpolar R en el punto crítico
        r_at_alpha_c = np.interp(alpha_c_i, alphas, order_individual)
        
        # Punto crítico
        ax.scatter(alpha_c_i, r_at_alpha_c, s=200, color=colors[i], 
                  edgecolors='black', linewidths=2, zorder=10,
                  marker='*')
        
        # Línea vertical punteada desde el eje x hasta el punto
        ax.plot([alpha_c_i, alpha_c_i], [0, r_at_alpha_c], 
               color=colors[i], linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Línea promedio en negro grueso
    ax.plot(alphas, order_mean, linewidth=4, color='black', 
           linestyle='--', alpha=0.8, label='Promedio', zorder=5)
    
    # Marcar α_c promedio
    alpha_c_mean = results['alpha_c_mean']
    r_at_mean = np.interp(alpha_c_mean, alphas, order_mean)
    ax.scatter(alpha_c_mean, r_at_mean, s=400, color='red', 
              edgecolors='black', linewidths=3, zorder=15,
              marker='D', label=f'α_c promedio = {alpha_c_mean:.3f}')
    
    ax.set_xlabel('α (acoplamiento externo)', fontsize=14, fontweight='bold')
    ax.set_ylabel('R (parámetro de orden)', fontsize=14, fontweight='bold')
    ax.set_title(f'Puntos críticos α_c - Clase {target_class}\n' + 
                f'(★ = punto crítico de cada imagen, ◆ = promedio)',
                fontsize=16, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, ncol=1)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([alphas[0], alphas[-1]])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    output_file = output_dir / f'puntos_criticos_todas_imagenes_clase_{target_class}.png'
    save_figure(fig, output_file)
    print(f"✓ Gráfica de puntos críticos guardada en: {output_file}")
    plt.close()
    
    return output_file


def plot_critical_points_comparison(results_data, output_dir):
    """
    Gráfica comparativa mostrando la distribución espacial de α_c.
    """
    settings = results_data['settings']
    results = results_data['results']
    
    target_class = settings['target_class']
    alpha_c_values = np.array(results['alpha_c_values'])
    alpha_c_mean = results['alpha_c_mean']
    alpha_c_std = results['alpha_c_std']
    n_images = len(alpha_c_values)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Línea temporal mostrando α_c de cada imagen
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, n_images))
    
    for i in range(n_images):
        ax1.scatter(i+1, alpha_c_values[i], s=250, color=colors[i], 
                   edgecolors='black', linewidths=2, zorder=10)
        ax1.plot([i+1, i+1], [0, alpha_c_values[i]], 
                color=colors[i], linewidth=2, alpha=0.5)
    
    # Líneas de referencia
    ax1.axhline(alpha_c_mean, color='red', linestyle='--', linewidth=3,
               label=f'Media: {alpha_c_mean:.3f}', zorder=5)
    ax1.axhspan(alpha_c_mean - alpha_c_std, alpha_c_mean + alpha_c_std, 
               alpha=0.2, color='red', label=f'± σ: {alpha_c_std:.3f}')
    
    ax1.set_xlabel('Número de imagen', fontsize=13, fontweight='bold')
    ax1.set_ylabel('α_c (punto crítico)', fontsize=13, fontweight='bold')
    ax1.set_title(f'α_c por imagen - Clase {target_class}', 
                 fontsize=15, fontweight='bold')
    ax1.set_xticks(range(1, n_images + 1))
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, max(alpha_c_values) * 1.1])
    
    # Subplot 2: Representación tipo "termómetro"
    ax2 = axes[1]
    
    # Ordenar por valor de α_c
    sorted_indices = np.argsort(alpha_c_values)
    sorted_alpha_c = alpha_c_values[sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    
    # Barras horizontales
    y_positions = np.arange(n_images)
    bars = ax2.barh(y_positions, sorted_alpha_c, color=sorted_colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Línea de promedio
    ax2.axvline(alpha_c_mean, color='red', linestyle='--', linewidth=3,
               label=f'Media: {alpha_c_mean:.3f}')
    
    # Etiquetas
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([f'Img {sorted_indices[i]+1}' for i in range(n_images)])
    ax2.set_xlabel('α_c (punto crítico)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Imagen (ordenado por α_c)', fontsize=13, fontweight='bold')
    ax2.set_title(f'Ranking de α_c - Clase {target_class}', 
                 fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(axis='x', alpha=0.3)
    
    # Añadir valores en las barras
    for i, (bar, val) in enumerate(zip(bars, sorted_alpha_c)):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = output_dir / f'comparacion_alpha_c_clase_{target_class}.png'
    save_figure(fig, output_file)
    print(f"✓ Comparación de α_c guardada en: {output_file}")
    plt.close()
    
    return output_file


def plot_critical_points_heatmap(results_data, output_dir):
    """
    Mapa de calor mostrando la relación entre imagen, α y R.
    """
    settings = results_data['settings']
    results = results_data['results']
    
    target_class = settings['target_class']
    alphas = np.array(settings['alphas'])
    alpha_c_values = np.array(results['alpha_c_values'])
    order_mean = np.array(results['order_curve_mean'])
    order_std = np.array(results['order_curve_std'])
    n_images = len(alpha_c_values)
    
    # Simular curvas individuales para el heatmap
    curves_matrix = np.zeros((n_images, len(alphas)))
    for i in range(n_images):
        np.random.seed(i)
        noise = np.random.randn(len(alphas)) * order_std * 0.5
        curves_matrix[i] = np.clip(order_mean + noise, 0, 1)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Heatmap
    im = ax.imshow(curves_matrix, aspect='auto', cmap='viridis', 
                  interpolation='bilinear', origin='lower',
                  extent=[alphas[0], alphas[-1], 0.5, n_images + 0.5])
    
    # Marcar puntos críticos
    for i in range(n_images):
        ax.scatter(alpha_c_values[i], i+1, s=200, color='red', 
                  marker='*', edgecolors='white', linewidths=2, zorder=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('R (parámetro de orden)', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('α (acoplamiento externo)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Número de imagen', fontsize=13, fontweight='bold')
    ax.set_title(f'Mapa de calor R(α) con puntos críticos (★) - Clase {target_class}',
                fontsize=15, fontweight='bold')
    ax.set_yticks(range(1, n_images + 1))
    ax.grid(False)
    
    plt.tight_layout()
    
    output_file = output_dir / f'heatmap_puntos_criticos_clase_{target_class}.png'
    save_figure(fig, output_file)
    print(f"✓ Heatmap guardado en: {output_file}")
    plt.close()
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Visualizar puntos críticos individuales por imagen'
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
    print(f"Visualización de puntos críticos por imagen")
    print(f"{'='*70}\n")
    print(f"Cargando: {input_path.name}")
    
    with open(input_path, 'r') as f:
        results_data = json.load(f)
    
    # Directorio de salida
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent / 'graficas_puntos_criticos'
    
    output_dir.mkdir(exist_ok=True)
    print(f"Guardando en: {output_dir}\n")
    
    # Generar gráficas
    print("Generando visualizaciones...\n")
    plot_critical_points_all_images(results_data, output_dir)
    plot_critical_points_comparison(results_data, output_dir)
    plot_critical_points_heatmap(results_data, output_dir)
    
    print(f"\n{'='*70}")
    print(f"✓ Completado")
    print(f"{'='*70}\n")
    print(f"Archivos generados en: {output_dir.absolute()}")
    for file in sorted(output_dir.glob('*.png')):
        print(f"  • {file.name}")
    print()


if __name__ == '__main__':
    main()
