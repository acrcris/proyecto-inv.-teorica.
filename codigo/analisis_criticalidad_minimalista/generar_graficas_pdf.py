"""
Generar gráficas de distribución para todas las métricas en formato PDF
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

print('📊 GENERANDO GRÁFICAS DE DISTRIBUCIÓN EN PDF')
print('='*80)

# Cargar datos
data = torch.load('resultados_kuramoto_full_dataset_CORRECTED/metricas_completas_CORRECTED.pt', weights_only=False)
metricas = [m for m in data['metricas'] if m.get('success', False)]

print(f'Procesando {len(metricas)} imágenes...')

# Calcular métricas por imagen
resultados = []
for m in metricas:
    R_series = m['R_series']
    R_median = np.median(R_series)
    R_final = R_series[-1]
    
    global_series = m['global_series']
    global_median = np.median(global_series)
    
    PSD_slope = m['PSD_slope']
    DFA_alpha = m['DFA_alpha']
    
    MI_matrix = m['MI_matrix']
    mi_values = MI_matrix[np.triu_indices_from(MI_matrix, k=1)]
    MI_median = np.median(mi_values)
    
    entropy_dict = m['entropy_by_channel']
    entropy_values = [entropy_dict[f'Canal_{i}'] for i in range(4)]
    Entropy_median = np.median(entropy_values)
    
    corr_matrix = m['correlation_matrix']
    corr_values = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    Corr_median = np.median(corr_values)
    
    resultados.append({
        'label': m['label'],
        'R_median': R_median,
        'R_final': R_final,
        'Global_median': global_median,
        'PSD_slope': PSD_slope,
        'DFA_alpha': DFA_alpha,
        'MI_median': MI_median,
        'Entropy_median': Entropy_median,
        'Corr_median': Corr_median,
    })

df = pd.DataFrame(resultados)

# Crear directorio para gráficas
os.makedirs('resultados_kuramoto_full_dataset_CORRECTED/graficas_distribuciones', exist_ok=True)

# Definir métricas con sus características
metricas_info = {
    'R_median': {
        'nombre': 'R (Parámetro de Orden) - Mediana',
        'xlabel': 'R (mediana temporal)',
        'objetivo': 0.5,
        'color': 'blue'
    },
    'R_final': {
        'nombre': 'R final (Valor Estacionario)',
        'xlabel': 'R (valor final)',
        'objetivo': None,
        'color': 'darkblue'
    },
    'Global_median': {
        'nombre': 'Magnitud Global - Mediana',
        'xlabel': 'Magnitud Global (mediana)',
        'objetivo': None,
        'color': 'green'
    },
    'PSD_slope': {
        'nombre': 'PSD Slope (Ley de Potencias)',
        'xlabel': 'Pendiente PSD',
        'objetivo': -1.0,
        'color': 'red'
    },
    'DFA_alpha': {
        'nombre': 'DFA Alpha (Fluctuaciones)',
        'xlabel': 'α (DFA)',
        'objetivo': 1.0,
        'color': 'purple'
    },
    'MI_median': {
        'nombre': 'Mutual Information - Mediana',
        'xlabel': 'MI (mediana entre canales)',
        'objetivo': None,
        'color': 'orange'
    },
    'Entropy_median': {
        'nombre': 'Entropía de Shannon - Mediana',
        'xlabel': 'Entropía (bits)',
        'objetivo': None,
        'color': 'brown'
    },
    'Corr_median': {
        'nombre': 'Correlación - Mediana',
        'xlabel': 'Correlación (mediana entre canales)',
        'objetivo': None,
        'color': 'magenta'
    }
}

print()
print('Generando gráficas en PDF...')
print()

# Generar una figura por métrica
for metrica, info in metricas_info.items():
    print(f'  📈 {info["nombre"]}...')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Distribución: {info["nombre"]}', fontsize=16, fontweight='bold')
    
    valores = df[metrica].values
    
    # 1. Histograma general con KDE
    ax = axes[0, 0]
    ax.hist(valores, bins=50, density=True, alpha=0.7, color=info['color'], edgecolor='black')
    
    # KDE
    kde = stats.gaussian_kde(valores)
    x_range = np.linspace(valores.min(), valores.max(), 200)
    ax.plot(x_range, kde(x_range), 'k-', linewidth=2, label='KDE')
    
    # Mediana
    mediana = np.median(valores)
    ax.axvline(mediana, color='red', linestyle='--', linewidth=2, label=f'Mediana={mediana:.4f}')
    
    # Objetivo si existe
    if info['objetivo'] is not None:
        ax.axvline(info['objetivo'], color='green', linestyle=':', linewidth=2, label=f'Objetivo={info["objetivo"]}')
    
    ax.set_xlabel(info['xlabel'], fontsize=11)
    ax.set_ylabel('Densidad', fontsize=11)
    ax.set_title('Histograma + KDE (Todas las clases)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Q-Q Plot (vs Normal)
    ax = axes[0, 1]
    stats.probplot(valores, dist='norm', plot=ax)
    ax.set_title('Q-Q Plot (vs Normal)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Test de normalidad
    muestra = np.random.choice(valores, size=min(5000, len(valores)), replace=False)
    stat, p_value = stats.shapiro(muestra)
    skew = stats.skew(valores)
    kurt = stats.kurtosis(valores)
    
    ax.text(0.05, 0.95, f'Shapiro-Wilk: p={p_value:.6f}\nSkew={skew:.3f}\nKurt={kurt:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Boxplot por clase
    ax = axes[1, 0]
    data_by_class = [df[df['label'] == c][metrica].values for c in range(10)]
    bp = ax.boxplot(data_by_class, tick_labels=[str(i) for i in range(10)],
                    patch_artist=True, showmeans=True)
    
    # Colorear boxplots
    for patch in bp['boxes']:
        patch.set_facecolor(info['color'])
        patch.set_alpha(0.6)
    
    # Línea objetivo
    if info['objetivo'] is not None:
        ax.axhline(info['objetivo'], color='green', linestyle=':', linewidth=2, label=f'Objetivo={info["objetivo"]}')
        ax.legend()
    
    ax.set_xlabel('Clase', fontsize=11)
    ax.set_ylabel(info['xlabel'], fontsize=11)
    ax.set_title('Distribución por Clase (Boxplot)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Violin plot por clase
    ax = axes[1, 1]
    positions = list(range(10))
    parts = ax.violinplot(data_by_class, positions=positions, showmeans=True, showmedians=True)
    
    # Colorear violins
    for pc in parts['bodies']:
        pc.set_facecolor(info['color'])
        pc.set_alpha(0.6)
    
    # Línea objetivo
    if info['objetivo'] is not None:
        ax.axhline(info['objetivo'], color='green', linestyle=':', linewidth=2, label=f'Objetivo={info["objetivo"]}')
        ax.legend()
    
    ax.set_xlabel('Clase', fontsize=11)
    ax.set_ylabel(info['xlabel'], fontsize=11)
    ax.set_title('Distribución por Clase (Violin Plot)', fontsize=12)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(i) for i in range(10)])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Guardar en PDF
    filename = f'resultados_kuramoto_full_dataset_CORRECTED/graficas_distribuciones/distribucion_{metrica}.pdf'
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f'     ✅ Guardado: {filename}')

print()
print('='*80)
print('✅ GRÁFICAS PDF COMPLETADAS')
print(f'   Directorio: resultados_kuramoto_full_dataset_CORRECTED/graficas_distribuciones/')
print(f'   Total: {len(metricas_info)} archivos PDF generados')
print('='*80)
