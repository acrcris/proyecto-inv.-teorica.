#!/usr/bin/env python3
"""
visualizar_curvas_finas_R_alpha.py

Visualiza curvas R(α) de alta resolución y puntos críticos α_c.
Genera gráficas similares a la imagen de referencia.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import argparse

plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11


def plot_curva_transicion_estilo_referencia(alphas, R_values, alpha_c=None, 
                                            titulo="Transición de fase en función del acoplamiento externo",
                                            output_path=None):
    """
    Genera gráfica estilo la imagen de referencia.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Curva principal
    ax.plot(alphas, R_values, '-', linewidth=2.5, color='#2E5C9A')
    
    # Marcar punto crítico si existe
    if alpha_c is not None:
        idx_c = np.argmin(np.abs(alphas - alpha_c))
        ax.axvline(alpha_c, color='red', linestyle='--', linewidth=1.5, 
                  alpha=0.7, label=f'α_c = {alpha_c:.3f}')
        ax.plot(alphas[idx_c], R_values[idx_c], 'ro', markersize=10,
               label=f'R(α_c) = {R_values[idx_c]:.3f}')
    
    ax.set_xlabel('Factor de acoplamiento α (escala de C)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Parámetro de orden R', fontsize=14, fontweight='bold')
    ax.set_title(titulo, fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_ylim(-0.05, max(R_values) * 1.1)
    ax.set_xlim(alphas[0], alphas[-1])
    
    if alpha_c is not None:
        ax.legend(fontsize=12, loc='lower right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Guardado: {output_path}")
    
    plt.close()


def plot_analisis_completo(data, output_dir='plots_curvas_finas'):
    """Genera análisis completo de curvas finas."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    alphas = data['alphas']
    curvas = data['curvas']
    
    print(f"\n📊 Generando visualizaciones...")
    
    # 1. PDF individual por clase con todas las imágenes
    for clase in sorted(curvas.keys()):
        pdf_path = Path(output_dir) / f'curvas_finas_clase_{clase}.pdf'
        
        with PdfPages(pdf_path) as pdf:
            # Página 1: Todas las curvas de la clase superpuestas
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1.1: Curvas R(α)
            ax = axes[0, 0]
            colors = plt.cm.tab10(np.linspace(0, 1, len(curvas[clase])))
            
            for idx, (img_idx, img_data) in enumerate(sorted(curvas[clase].items())):
                R = img_data['R']
                alpha_c = img_data['alpha_c']
                ax.plot(alphas, R, '-', linewidth=2, color=colors[idx], 
                       alpha=0.7, label=f'Img {img_idx} (α_c={alpha_c:.3f})')
            
            ax.set_xlabel('Alpha (α)', fontsize=12, fontweight='bold')
            ax.set_ylabel('R', fontsize=12, fontweight='bold')
            ax.set_title(f'Curvas R(α) - Clase {clase}', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, ncol=2)
            ax.set_ylim(-0.05, 1.05)
            
            # 1.2: Derivadas dR/dα
            ax = axes[0, 1]
            for idx, (img_idx, img_data) in enumerate(sorted(curvas[clase].items())):
                dR = img_data['dR']
                alpha_c = img_data['alpha_c']
                ax.plot(alphas, dR, '-', linewidth=2, color=colors[idx], alpha=0.7)
                
                # Marcar máximo
                idx_c = img_data['idx_c']
                ax.plot(alpha_c, dR[idx_c], 'o', color=colors[idx], markersize=8)
            
            ax.set_xlabel('Alpha (α)', fontsize=12, fontweight='bold')
            ax.set_ylabel('dR/dα', fontsize=12, fontweight='bold')
            ax.set_title('Derivada dR/dα (Detectar α_c)', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
            
            # 1.3: Distribución de α_c
            ax = axes[1, 0]
            alphas_c = [img_data['alpha_c'] for img_data in curvas[clase].values()]
            ax.hist(alphas_c, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(np.mean(alphas_c), color='red', linestyle='--', 
                      linewidth=2, label=f'Media = {np.mean(alphas_c):.3f}')
            ax.set_xlabel('α_c', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
            ax.set_title('Distribución de Puntos Críticos', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 1.4: Curva promedio con intervalo de confianza
            ax = axes[1, 1]
            R_matrix = np.array([img_data['R'] for img_data in curvas[clase].values()])
            R_mean = np.mean(R_matrix, axis=0)
            R_std = np.std(R_matrix, axis=0)
            
            ax.plot(alphas, R_mean, '-', linewidth=3, color='blue', label='Media')
            ax.fill_between(alphas, R_mean - R_std, R_mean + R_std, 
                           alpha=0.3, color='blue', label='±1σ')
            
            alpha_c_mean = np.mean(alphas_c)
            ax.axvline(alpha_c_mean, color='red', linestyle='--', linewidth=2,
                      label=f'α_c medio = {alpha_c_mean:.3f}')
            
            ax.set_xlabel('Alpha (α)', fontsize=12, fontweight='bold')
            ax.set_ylabel('R', fontsize=12, fontweight='bold')
            ax.set_title('Curva Promedio con Variabilidad', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
            
            plt.suptitle(f'Análisis Completo - Clase {clase}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(dpi=150, bbox_inches='tight')
            plt.close()
            
            # Página 2: Curva estilo referencia para cada imagen
            n_imgs = len(curvas[clase])
            n_cols = 3
            n_rows = (n_imgs + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
            if n_imgs == 1:
                axes = np.array([axes])
            axes = axes.ravel()
            
            for idx, (img_idx, img_data) in enumerate(sorted(curvas[clase].items())):
                ax = axes[idx]
                R = img_data['R']
                alpha_c = img_data['alpha_c']
                
                ax.plot(alphas, R, '-', linewidth=2.5, color='#2E5C9A')
                ax.axvline(alpha_c, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                
                idx_c = img_data['idx_c']
                ax.plot(alpha_c, R[idx_c], 'ro', markersize=8)
                
                ax.set_xlabel('α', fontsize=10)
                ax.set_ylabel('R', fontsize=10)
                ax.set_title(f'Img {img_idx}: α_c = {alpha_c:.3f}', 
                           fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle=':')
                ax.set_ylim(-0.05, 1.05)
            
            # Ocultar ejes no usados
            for idx in range(n_imgs, len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle(f'Transiciones Individuales - Clase {clase}', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"   ✓ Clase {clase}: {pdf_path}")
    
    # 2. Comparación entre clases
    print(f"\n📊 Generando comparación entre clases...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # 2.1: Curvas medias por clase
    ax = axes[0, 0]
    alphas_c_por_clase = {}
    
    for clase in sorted(curvas.keys()):
        R_matrix = np.array([img_data['R'] for img_data in curvas[clase].values()])
        R_mean = np.mean(R_matrix, axis=0)
        
        alphas_c = [img_data['alpha_c'] for img_data in curvas[clase].values()]
        alphas_c_por_clase[clase] = alphas_c
        alpha_c_mean = np.mean(alphas_c)
        
        ax.plot(alphas, R_mean, '-', linewidth=2.5, color=colors[clase],
               label=f'Clase {clase} (α_c={alpha_c_mean:.3f})')
    
    ax.set_xlabel('Alpha (α)', fontsize=12, fontweight='bold')
    ax.set_ylabel('R medio', fontsize=12, fontweight='bold')
    ax.set_title('Curvas Medias por Clase', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    ax.set_ylim(-0.05, 1.05)
    
    # 2.2: Puntos críticos por clase
    ax = axes[0, 1]
    clase_labels = []
    alpha_c_data = []
    
    for clase in sorted(curvas.keys()):
        alphas_c = alphas_c_por_clase[clase]
        clase_labels.append(f'C{clase}')
        alpha_c_data.append(alphas_c)
    
    bp = ax.boxplot(alpha_c_data, labels=clase_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Clase', fontsize=12, fontweight='bold')
    ax.set_ylabel('α_c', fontsize=12, fontweight='bold')
    ax.set_title('Distribución de α_c por Clase', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2.3: Derivadas en región crítica
    ax = axes[1, 0]
    
    # Zoom en región crítica [0, 0.5]
    idx_region = alphas <= 0.5
    alphas_zoom = alphas[idx_region]
    
    for clase in sorted(curvas.keys()):
        dR_matrix = np.array([img_data['dR'] for img_data in curvas[clase].values()])
        dR_mean = np.mean(dR_matrix, axis=0)
        
        ax.plot(alphas_zoom, dR_mean[idx_region], '-', linewidth=2.5, 
               color=colors[clase], label=f'Clase {clase}')
    
    ax.set_xlabel('Alpha (α)', fontsize=12, fontweight='bold')
    ax.set_ylabel('dR/dα medio', fontsize=12, fontweight='bold')
    ax.set_title('Derivadas en Región Crítica (α ≤ 0.5)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    
    # 2.4: Tabla de estadísticos
    ax = axes[1, 1]
    ax.axis('off')
    
    tabla_data = []
    for clase in sorted(curvas.keys()):
        alphas_c = alphas_c_por_clase[clase]
        R_matrix = np.array([img_data['R'] for img_data in curvas[clase].values()])
        
        # R en α=1.0
        idx_1 = np.argmin(np.abs(alphas - 1.0))
        R_at_1 = np.mean(R_matrix[:, idx_1])
        
        tabla_data.append([
            f'Clase {clase}',
            f'{np.mean(alphas_c):.4f}',
            f'{np.std(alphas_c):.4f}',
            f'{np.min(alphas_c):.4f}',
            f'{np.max(alphas_c):.4f}',
            f'{R_at_1:.4f}'
        ])
    
    tabla = ax.table(cellText=tabla_data,
                    colLabels=['Clase', 'α_c medio', 'σ(α_c)', 'Min', 'Max', 'R(α=1)'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1, 2)
    
    for i in range(len(tabla_data) + 1):
        for j in range(6):
            cell = tabla[(i, j)]
            if i == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax.set_title('Estadísticos de Puntos Críticos', fontsize=13, fontweight='bold', pad=20)
    
    plt.suptitle('Comparación Entre Clases - Análisis de α_c', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'comparacion_clases_curvas_finas.pdf'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Comparación: {output_path}")
    plt.close()
    
    # 3. Generar imagen estilo referencia (curva promedio global)
    print(f"\n📊 Generando curva estilo referencia...")
    
    # Promedio de TODAS las clases
    all_R = []
    for clase in sorted(curvas.keys()):
        for img_data in curvas[clase].values():
            all_R.append(img_data['R'])
    
    R_global = np.mean(all_R, axis=0)
    
    # α_c global
    dR_global = np.gradient(R_global, alphas)
    idx_c_global = np.argmax(dR_global)
    alpha_c_global = alphas[idx_c_global]
    
    output_path = Path(output_dir) / 'transicion_global_estilo_referencia.pdf'
    plot_curva_transicion_estilo_referencia(
        alphas, R_global, alpha_c_global,
        titulo="Transición de fase en función del acoplamiento externo",
        output_path=output_path
    )


def main():
    parser = argparse.ArgumentParser(
        description='Visualiza curvas R(α) de alta resolución'
    )
    parser.add_argument('--input', default='curvas_finas_R_alpha.pt',
                       help='Archivo de datos .pt')
    parser.add_argument('--output_dir', default='plots_curvas_finas',
                       help='Directorio de salida')
    
    args = parser.parse_args()
    
    print(f"{'='*70}")
    print(f"VISUALIZACIÓN DE CURVAS R(α) DE ALTA RESOLUCIÓN")
    print(f"{'='*70}")
    print(f"Archivo de entrada: {args.input}")
    print(f"Directorio de salida: {args.output_dir}")
    
    # Cargar datos
    print("\nCargando datos...")
    data = torch.load(args.input, weights_only=False)
    print(f"  Clases: {len(data['curvas'])}")
    print(f"  Alphas: {len(data['alphas'])} puntos")
    print(f"  Rango: [{data['alphas'][0]:.2f}, {data['alphas'][-1]:.2f}]")
    
    # Generar visualizaciones
    plot_analisis_completo(data, args.output_dir)
    
    print(f"\n{'='*70}")
    print(f"✅ VISUALIZACIÓN COMPLETA")
    print(f"{'='*70}")
    print(f"Resultados en: {args.output_dir}/")


if __name__ == '__main__':
    main()
