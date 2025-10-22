#!/usr/bin/env python3
"""
analizar_curvas_individuales_R_alpha.py

Genera curvas R(α) INDIVIDUALES para cada imagen.
Cada imagen tiene su propia curva completa R vs alpha.

Salida:
- PDFs con curvas individuales por clase
- Análisis de variabilidad entre imágenes de la misma clase
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import argparse

# Configuración
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10


def load_and_organize_data(filepath: str) -> dict:
    """
    Carga datos y organiza por imagen individual.
    
    Returns:
        dict con estructura:
        {
            clase: {
                img_idx: {
                    'alphas': [alpha1, alpha2, ...],
                    'R_values': [R1, R2, ...]
                }
            }
        }
    """
    print("Cargando datos...")
    data = torch.load(filepath, weights_only=False)
    print(f"  Total registros: {len(data['data'])}")
    
    # Organizar por clase e imagen
    organized = {}
    
    for record in data['data']:
        clase = record['label']
        img_idx = record['img_idx']
        alpha = record['alpha']
        R_final = record['R_final']
        
        if clase not in organized:
            organized[clase] = {}
        
        if img_idx not in organized[clase]:
            organized[clase][img_idx] = {
                'alphas': [],
                'R_values': []
            }
        
        organized[clase][img_idx]['alphas'].append(alpha)
        organized[clase][img_idx]['R_values'].append(R_final)
    
    # Ordenar alphas para cada imagen
    for clase in organized:
        for img_idx in organized[clase]:
            # Convertir a arrays y ordenar
            alphas = np.array(organized[clase][img_idx]['alphas'])
            R_values = np.array(organized[clase][img_idx]['R_values'])
            
            # Ordenar por alpha
            sort_idx = np.argsort(alphas)
            organized[clase][img_idx]['alphas'] = alphas[sort_idx]
            organized[clase][img_idx]['R_values'] = R_values[sort_idx]
    
    print("\nEstructura de datos:")
    for clase in sorted(organized.keys()):
        n_imgs = len(organized[clase])
        n_alphas = len(organized[clase][0]['alphas']) if n_imgs > 0 else 0
        print(f"  Clase {clase}: {n_imgs} imágenes × {n_alphas} alphas")
    
    return organized


def plot_curvas_individuales_por_clase(data_org, output_dir='plots_curvas_individuales'):
    """
    Genera PDFs con curvas R(α) individuales para cada clase.
    Un PDF por clase mostrando todas las curvas de sus imágenes.
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    print(f"\n📊 Generando curvas individuales por clase...")
    
    for clase in sorted(data_org.keys()):
        print(f"\n   Clase {clase}:")
        
        imagenes = data_org[clase]
        n_imgs = len(imagenes)
        
        # Crear PDF para esta clase
        pdf_path = Path(output_dir) / f'curvas_individuales_clase_{clase}.pdf'
        
        with PdfPages(pdf_path) as pdf:
            # Página 1: Todas las curvas superpuestas
            fig, ax = plt.subplots(figsize=(12, 8))
            
            colors = plt.cm.tab20(np.linspace(0, 1, n_imgs))
            
            for idx, (img_idx, img_data) in enumerate(sorted(imagenes.items())):
                alphas = img_data['alphas']
                R_values = img_data['R_values']
                
                ax.plot(alphas, R_values, 'o-', 
                       color=colors[idx], alpha=0.7, linewidth=2,
                       markersize=4, label=f'Img {img_idx}')
            
            ax.set_xlabel('Alpha (α)', fontsize=14, fontweight='bold')
            ax.set_ylabel('R (parámetro de orden)', fontsize=14, fontweight='bold')
            ax.set_title(f'Curvas R(α) Individuales - Clase {clase} (N={n_imgs} imágenes)', 
                        fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                     fontsize=8, ncol=2 if n_imgs > 10 else 1)
            
            plt.tight_layout()
            pdf.savefig(dpi=150, bbox_inches='tight')
            plt.close()
            
            # Página 2: Grid de curvas individuales pequeñas
            n_cols = 5
            n_rows = (n_imgs + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3*n_rows))
            if n_imgs == 1:
                axes = np.array([axes])
            axes = axes.ravel()
            
            for idx, (img_idx, img_data) in enumerate(sorted(imagenes.items())):
                ax = axes[idx]
                alphas = img_data['alphas']
                R_values = img_data['R_values']
                
                ax.plot(alphas, R_values, 'o-', color='blue', 
                       linewidth=2, markersize=3)
                ax.set_title(f'Img {img_idx}', fontsize=10, fontweight='bold')
                ax.set_xlabel('α', fontsize=9)
                ax.set_ylabel('R', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-0.05, 1.05)
                ax.tick_params(labelsize=8)
            
            # Ocultar ejes no usados
            for idx in range(n_imgs, len(axes)):
                axes[idx].axis('off')
            
            fig.suptitle(f'Curvas R(α) Individuales - Clase {clase}', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(dpi=150, bbox_inches='tight')
            plt.close()
            
            # Página 3: Análisis de variabilidad
            fig = plt.figure(figsize=(14, 10))
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            
            # Recolectar todos los datos
            all_alphas = imagenes[0]['alphas']  # Asumimos mismos alphas para todos
            R_matrix = np.array([img_data['R_values'] for img_data in imagenes.values()])
            
            # 1. Media y desviación estándar
            ax1 = fig.add_subplot(gs[0, :])
            R_mean = np.mean(R_matrix, axis=0)
            R_std = np.std(R_matrix, axis=0)
            R_min = np.min(R_matrix, axis=0)
            R_max = np.max(R_matrix, axis=0)
            
            ax1.plot(all_alphas, R_mean, 'k-', linewidth=3, label='Media')
            ax1.fill_between(all_alphas, R_mean - R_std, R_mean + R_std, 
                           alpha=0.3, color='blue', label='±1σ')
            ax1.fill_between(all_alphas, R_min, R_max, 
                           alpha=0.2, color='red', label='Min-Max')
            
            ax1.set_xlabel('Alpha (α)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('R', fontsize=12, fontweight='bold')
            ax1.set_title('Estadísticos de Curvas Individuales', 
                         fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=10)
            ax1.set_ylim(-0.05, 1.05)
            
            # 2. Desviación estándar vs alpha
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.plot(all_alphas, R_std, 'o-', color='purple', linewidth=2)
            ax2.set_xlabel('Alpha (α)', fontsize=11)
            ax2.set_ylabel('Desv. Estándar (σ)', fontsize=11)
            ax2.set_title('Variabilidad entre Imágenes', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # 3. Coeficiente de variación
            ax3 = fig.add_subplot(gs[1, 1])
            CV = (R_std / (R_mean + 1e-10)) * 100
            ax3.plot(all_alphas, CV, 'o-', color='orange', linewidth=2)
            ax3.set_xlabel('Alpha (α)', fontsize=11)
            ax3.set_ylabel('Coef. Variación (%)', fontsize=11)
            ax3.set_title('CV = (σ/μ) × 100', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # 4. Heatmap de R values
            ax4 = fig.add_subplot(gs[2, :])
            im = ax4.imshow(R_matrix, aspect='auto', cmap='viridis', 
                          interpolation='nearest')
            ax4.set_xlabel('Índice de Alpha', fontsize=11)
            ax4.set_ylabel('Imagen', fontsize=11)
            ax4.set_title('Heatmap: R(α) para cada Imagen', 
                         fontsize=12, fontweight='bold')
            
            # Etiquetas de alpha en x
            n_ticks = min(len(all_alphas), 11)
            tick_positions = np.linspace(0, len(all_alphas)-1, n_ticks, dtype=int)
            ax4.set_xticks(tick_positions)
            ax4.set_xticklabels([f'{all_alphas[i]:.1f}' for i in tick_positions], 
                              rotation=45)
            
            plt.colorbar(im, ax=ax4, label='R')
            
            fig.suptitle(f'Análisis de Variabilidad - Clase {clase}', 
                        fontsize=16, fontweight='bold')
            pdf.savefig(dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"      ✓ PDF generado: {pdf_path}")


def plot_comparacion_clases_media(data_org, output_dir='plots_curvas_individuales'):
    """
    Compara las curvas medias entre todas las clases.
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    print(f"\n📊 Generando comparación entre clases...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Recolectar datos de todas las clases
    clases_stats = {}
    
    for clase in sorted(data_org.keys()):
        imagenes = data_org[clase]
        all_alphas = imagenes[0]['alphas']
        R_matrix = np.array([img_data['R_values'] for img_data in imagenes.values()])
        
        clases_stats[clase] = {
            'alphas': all_alphas,
            'R_mean': np.mean(R_matrix, axis=0),
            'R_std': np.std(R_matrix, axis=0),
            'R_min': np.min(R_matrix, axis=0),
            'R_max': np.max(R_matrix, axis=0)
        }
    
    # 1. Curvas medias
    ax1 = axes[0, 0]
    for clase in sorted(clases_stats.keys()):
        stats = clases_stats[clase]
        ax1.plot(stats['alphas'], stats['R_mean'], 'o-', 
                color=colors[clase], linewidth=2, markersize=4,
                label=f'Clase {clase}')
    
    ax1.set_xlabel('Alpha (α)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('R medio', fontsize=12, fontweight='bold')
    ax1.set_title('Curvas R(α) Medias por Clase', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, ncol=2)
    ax1.set_ylim(-0.05, 1.05)
    
    # 2. Desviación estándar
    ax2 = axes[0, 1]
    for clase in sorted(clases_stats.keys()):
        stats = clases_stats[clase]
        ax2.plot(stats['alphas'], stats['R_std'], 'o-', 
                color=colors[clase], linewidth=2, markersize=4,
                label=f'Clase {clase}')
    
    ax2.set_xlabel('Alpha (α)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('σ (R)', fontsize=12, fontweight='bold')
    ax2.set_title('Variabilidad Intra-Clase', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, ncol=2)
    
    # 3. Rangos (Max - Min)
    ax3 = axes[1, 0]
    for clase in sorted(clases_stats.keys()):
        stats = clases_stats[clase]
        R_range = stats['R_max'] - stats['R_min']
        ax3.plot(stats['alphas'], R_range, 'o-', 
                color=colors[clase], linewidth=2, markersize=4,
                label=f'Clase {clase}')
    
    ax3.set_xlabel('Alpha (α)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Rango (Max - Min)', fontsize=12, fontweight='bold')
    ax3.set_title('Rango de R entre Imágenes', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9, ncol=2)
    
    # 4. Tabla de estadísticos en α=1.0
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Encontrar índice de α=1.0
    alpha_idx = np.argmin(np.abs(clases_stats[0]['alphas'] - 1.0))
    
    tabla_data = []
    for clase in sorted(clases_stats.keys()):
        stats = clases_stats[clase]
        tabla_data.append([
            f'Clase {clase}',
            f'{stats["R_mean"][alpha_idx]:.4f}',
            f'{stats["R_std"][alpha_idx]:.4f}',
            f'{stats["R_min"][alpha_idx]:.4f}',
            f'{stats["R_max"][alpha_idx]:.4f}'
        ])
    
    tabla = ax4.table(cellText=tabla_data,
                     colLabels=['Clase', 'R medio', 'σ', 'Min', 'Max'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 2)
    
    # Estilo de tabla
    for i in range(len(tabla_data) + 1):
        for j in range(5):
            cell = tabla[(i, j)]
            if i == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax4.set_title(f'Estadísticos en α = 1.0', fontsize=13, fontweight='bold', pad=20)
    
    plt.suptitle('Comparación entre Clases - Curvas R(α) Individuales', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'comparacion_clases_curvas_individuales.pdf'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Comparación guardada: {output_path}")
    plt.close()


def print_estadisticos_por_clase(data_org):
    """Imprime estadísticos de las curvas individuales."""
    print("\n" + "="*70)
    print("ESTADÍSTICOS DE CURVAS R(α) INDIVIDUALES")
    print("="*70)
    
    for clase in sorted(data_org.keys()):
        imagenes = data_org[clase]
        n_imgs = len(imagenes)
        
        # Calcular estadísticos en diferentes puntos de alpha
        all_alphas = imagenes[0]['alphas']
        R_matrix = np.array([img_data['R_values'] for img_data in imagenes.values()])
        
        print(f"\nClase {clase}: {n_imgs} imágenes")
        print("-" * 70)
        
        # Puntos clave de alpha
        alpha_points = [0.0, 0.1, 0.5, 1.0, 2.0]
        
        for alpha_val in alpha_points:
            idx = np.argmin(np.abs(all_alphas - alpha_val))
            actual_alpha = all_alphas[idx]
            
            R_values = R_matrix[:, idx]
            
            print(f"  α = {actual_alpha:.2f}:")
            print(f"    R medio: {np.mean(R_values):.4f}")
            print(f"    σ:       {np.std(R_values):.4f}")
            print(f"    Min:     {np.min(R_values):.4f}")
            print(f"    Max:     {np.max(R_values):.4f}")
            print(f"    Rango:   {np.max(R_values) - np.min(R_values):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Analiza curvas R(α) individuales para cada imagen'
    )
    parser.add_argument('--input', default='datos_R_alpha_10imgs.pt',
                       help='Archivo de datos .pt')
    parser.add_argument('--output_dir', default='plots_curvas_individuales',
                       help='Directorio de salida')
    
    args = parser.parse_args()
    
    print(f"{'='*70}")
    print(f"ANÁLISIS DE CURVAS R(α) INDIVIDUALES")
    print(f"{'='*70}")
    print(f"Archivo de entrada: {args.input}")
    print(f"Directorio de salida: {args.output_dir}")
    
    # Cargar y organizar datos
    data_org = load_and_organize_data(args.input)
    
    # Generar visualizaciones
    print(f"\n{'='*70}")
    print("GENERANDO VISUALIZACIONES...")
    print(f"{'='*70}")
    
    plot_curvas_individuales_por_clase(data_org, args.output_dir)
    plot_comparacion_clases_media(data_org, args.output_dir)
    print_estadisticos_por_clase(data_org)
    
    print(f"\n{'='*70}")
    print(f"✅ ANÁLISIS COMPLETO")
    print(f"{'='*70}")
    print(f"Resultados guardados en: {args.output_dir}/")
    print(f"  - Curvas individuales: curvas_individuales_clase_*.pdf (10 archivos)")
    print(f"  - Comparación clases: comparacion_clases_curvas_individuales.pdf")


if __name__ == '__main__':
    main()
