"""
Generar gráficas de distribución individuales para cada clase y cada métrica en PDF
VERSIÓN MAC - Análisis de 60k imágenes con 6 métricas principales
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

print('📊 GENERANDO GRÁFICAS POR CLASE - 6 MÉTRICAS PRINCIPALES EN PDF')
print('='*80)
print('Versión: Apple M3 (MPS) - 60,002 imágenes')
print('='*80)
print()

# Cargar datos de Mac
data = torch.load('resultados_kuramoto_TRAIN_MAC_60k/metricas_completas_TRAIN_MAC_60k.pt', weights_only=False)
metricas = [m for m in data['metricas'] if m.get('success', False)]

print(f'✅ Dataset cargado: {len(metricas)} imágenes')
print()

# Calcular las 6 métricas principales por imagen
print('📊 Calculando métricas por imagen...')
resultados = []
for i, m in enumerate(metricas):
    if i % 10000 == 0:
        print(f'   Procesadas: {i}/{len(metricas)}')
    
    # 1. R_stationary (valor estacionario)
    R_stationary = m['R_stationary']
    
    # 2. Magnitud media por canal (promedio de los 4 canales)
    mag_channel = m['mag_channel_mean_final']  # Array de 4 valores
    mag_channel_mean = np.mean(mag_channel)
    
    # 3. PSD_slope
    PSD_slope = m['PSD_slope']
    
    # 4. DFA_alpha
    DFA_alpha = m['DFA_alpha']
    
    # 5. Mutual Info
    MI = m['mutual_info']
    
    # 6. Entropía Shannon por canal (promedio de los 4 canales)
    shannon_entropy = m['shannon_entropy_by_channel']  # Array de 4 valores
    shannon_entropy_mean = np.mean(shannon_entropy)
    
    resultados.append({
        'label': m['label'],
        'R_stationary': R_stationary,
        'mag_channel_mean': mag_channel_mean,
        'PSD_slope': PSD_slope,
        'DFA_alpha': DFA_alpha,
        'mutual_info': MI,
        'shannon_entropy_mean': shannon_entropy_mean,
    })

df = pd.DataFrame(resultados)
print(f'✅ Métricas calculadas para {len(df)} imágenes')
print()

# Crear directorio para gráficas por clase
os.makedirs('resultados_kuramoto_TRAIN_MAC_60k/graficas_por_clase', exist_ok=True)

# Definir las 6 métricas principales con sus características
metricas_info = {
    'R_stationary': {
        'nombre': 'R (Parámetro de Orden) - Valor Estacionario',
        'xlabel': 'R (valor estacionario)',
        'objetivo': 0.5,
        'color': 'blue',
        'descripcion': 'Sincronización de osciladores en estado estacionario'
    },
    'mag_channel_mean': {
        'nombre': 'Magnitud Media por Canal (promedio 4 canales)',
        'xlabel': 'Magnitud media',
        'objetivo': None,
        'color': 'green',
        'descripcion': 'Actividad media de los 4 canales de osciladores'
    },
    'PSD_slope': {
        'nombre': 'PSD Slope (Pendiente del Espectro de Potencia)',
        'xlabel': 'Pendiente PSD',
        'objetivo': -2.0,
        'color': 'red',
        'descripcion': 'Exponente de la ley de potencia 1/f^α (-2 = ruido rosa/criticalidad)'
    },
    'DFA_alpha': {
        'nombre': 'DFA Alpha (Exponente de Fluctuaciones)',
        'xlabel': 'α (DFA)',
        'objetivo': 1.0,
        'color': 'purple',
        'descripcion': 'Correlaciones de largo alcance (1.0 = ruido 1/f/criticalidad)'
    },
    'mutual_info': {
        'nombre': 'Mutual Information (Información Mutua)',
        'xlabel': 'MI (nats)',
        'objetivo': None,
        'color': 'orange',
        'descripcion': 'Dependencia estadística entre mitades de la serie temporal'
    },
    'shannon_entropy_mean': {
        'nombre': 'Entropía de Shannon (promedio 4 canales)',
        'xlabel': 'Entropía (bits)',
        'objetivo': None,
        'color': 'brown',
        'descripcion': 'Complejidad/aleatoriedad promedio de los 4 canales'
    }
}

print()
print('Generando gráficas individuales por clase en PDF...')
print()

total_graficas = 0

# Generar una figura por métrica y por clase
for metrica, info in metricas_info.items():
    print(f'📊 Métrica: {info["nombre"]}')
    
    for clase in range(10):
        # Filtrar datos de esta clase
        df_clase = df[df['label'] == clase]
        n = len(df_clase)
        
        if n < 2:
            print(f'   ⚠️  Clase {clase}: Muy pocas muestras (N={n}), omitiendo...')
            continue
        
        valores = df_clase[metrica].values
        
        # Crear figura con 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{info["nombre"]} - CLASE {clase} (N={n} imágenes)', 
                     fontsize=16, fontweight='bold')
        
        # 1. Histograma + KDE
        ax = axes[0, 0]
        ax.hist(valores, bins=30, density=True, alpha=0.7, color=info['color'], edgecolor='black')
        
        # KDE solo si hay suficientes datos
        if n > 5:
            kde = stats.gaussian_kde(valores)
            x_range = np.linspace(valores.min(), valores.max(), 200)
            ax.plot(x_range, kde(x_range), 'k-', linewidth=2, label='KDE')
        
        # Mediana y cuartiles
        mediana = np.median(valores)
        q1 = np.percentile(valores, 25)
        q3 = np.percentile(valores, 75)
        
        ax.axvline(mediana, color='red', linestyle='--', linewidth=2, label=f'Md={mediana:.4f}')
        ax.axvline(q1, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Q1={q1:.4f}')
        ax.axvline(q3, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Q3={q3:.4f}')
        
        # Objetivo si existe
        if info['objetivo'] is not None:
            ax.axvline(info['objetivo'], color='lime', linestyle='-.', linewidth=2.5, 
                       label=f'Objetivo={info["objetivo"]}')
        
        ax.set_xlabel(info['xlabel'], fontsize=11)
        ax.set_ylabel('Densidad', fontsize=11)
        ax.set_title(f'Histograma + KDE', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 2. Q-Q Plot
        ax = axes[0, 1]
        if n >= 3:
            stats.probplot(valores, dist='norm', plot=ax)
            ax.set_title('Q-Q Plot (vs Normal)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Test de normalidad
            if n >= 3 and n <= 5000:
                stat, p_value = stats.shapiro(valores)
            elif n > 5000:
                muestra = np.random.choice(valores, size=5000, replace=False)
                stat, p_value = stats.shapiro(muestra)
            else:
                p_value = np.nan
            
            skew = stats.skew(valores)
            kurt = stats.kurtosis(valores)
            
            normal_str = '✓ Normal' if p_value > 0.05 else '✗ NO Normal'
            textstr = f'{normal_str}\np={p_value:.6f}\nSkew={skew:.3f}\nKurt={kurt:.3f}'
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        else:
            ax.text(0.5, 0.5, 'Insuficientes datos\npara Q-Q plot', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title('Q-Q Plot (vs Normal)', fontsize=12, fontweight='bold')
        
        # 3. Boxplot con comparación vs todas las clases
        ax = axes[1, 0]
        
        # Boxplot de esta clase
        bp_clase = ax.boxplot([valores], positions=[0], widths=0.6,
                              patch_artist=True, showmeans=True,
                              labels=[f'Clase {clase}'])
        bp_clase['boxes'][0].set_facecolor(info['color'])
        bp_clase['boxes'][0].set_alpha(0.8)
        
        # Boxplot de todas las demás clases (gris transparente)
        df_otras = df[df['label'] != clase]
        valores_otras = df_otras[metrica].values
        
        bp_otras = ax.boxplot([valores_otras], positions=[1], widths=0.6,
                              patch_artist=True, showmeans=True,
                              labels=['Otras clases'])
        bp_otras['boxes'][0].set_facecolor('gray')
        bp_otras['boxes'][0].set_alpha(0.4)
        
        # Línea objetivo
        if info['objetivo'] is not None:
            ax.axhline(info['objetivo'], color='lime', linestyle='-.', linewidth=2.5, 
                       label=f'Objetivo={info["objetivo"]}')
            ax.legend(fontsize=9)
        
        ax.set_ylabel(info['xlabel'], fontsize=11)
        ax.set_title('Boxplot: Clase vs Resto', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Estadísticas descriptivas
        ax = axes[1, 1]
        ax.axis('off')
        
        # Calcular estadísticas
        media = np.mean(valores)
        std = np.std(valores)
        iqr = q3 - q1
        min_val = np.min(valores)
        max_val = np.max(valores)
        
        # Distancia al objetivo
        if info['objetivo'] is not None:
            dist_objetivo = abs(mediana - info['objetivo'])
            dist_text = f'Distancia a objetivo: {dist_objetivo:.4f}'
        else:
            dist_text = ''
        
        stats_text = f'''ESTADÍSTICAS DESCRIPTIVAS - CLASE {clase}
        
N (muestras):          {n}

Mediana:               {mediana:.6f}
Media:                 {media:.6f}
Desv. Estándar:        {std:.6f}

Q1 (percentil 25):     {q1:.6f}
Q3 (percentil 75):     {q3:.6f}
IQR (Q3 - Q1):         {iqr:.6f}

Mínimo:                {min_val:.6f}
Máximo:                {max_val:.6f}
Rango:                 {max_val - min_val:.6f}

Skewness:              {skew:.6f}
Kurtosis:              {kurt:.6f}

{dist_text}
'''
        
        ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        
        # Guardar en PDF
        filename = f'resultados_kuramoto_TRAIN_MAC_60k/graficas_por_clase/clase_{clase:02d}_{metrica}.pdf'
        plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=150)
        plt.close()
        
        total_graficas += 1
        
        if clase == 0 or clase == 9:
            print(f'   ✅ Clase {clase}: {filename}')
        elif clase == 1:
            print(f'   ... (generando clases 1-8) ...')

print()
print('='*80)
print('✅ GRÁFICAS POR CLASE COMPLETADAS')
print('='*80)
print(f'📂 Directorio: resultados_kuramoto_TRAIN_MAC_60k/graficas_por_clase/')
print(f'📊 Total: {total_graficas} archivos PDF generados')
print(f'📝 Estructura: clase_XX_<metrica>.pdf')
print()
print('📋 Métricas analizadas:')
for i, (metrica, info) in enumerate(metricas_info.items(), 1):
    print(f'   {i}. {info["nombre"]}')
print()
print('🎯 Objetivos de criticalidad:')
print('   - R_stationary: 0.5 (sincronización parcial)')
print('   - PSD_slope: -2.0 (ruido rosa, ley 1/f²)')
print('   - DFA_alpha: 1.0 (ruido 1/f, criticalidad)')
print('='*80)
