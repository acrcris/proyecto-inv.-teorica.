"""
Script: optimizar_parametros_secuencial.py

FASE EXPLORATORIA - Búsqueda Secuencial de Parámetros Óptimos
=============================================================

Optimiza UN parámetro a la vez para encontrar configuración preliminar.
Usa GPU si está disponible para acelerar el proceso.

OBJETIVO:
---------
Encontrar parámetros que acerquen las métricas a valores críticos:
  • R_final ≈ 0.5 (sincronización crítica)
  • DFA α ≈ 1.0 (correlaciones de largo alcance críticas)
  • PSD slope ≈ -1.0 (ruido rosa / 1/f)

ESTRATEGIA:
-----------
Optimización secuencial (greedy):
  1. Fijar todos menos T_steps → probar 5 valores → elegir mejor
  2. Fijar T_steps*, variar gamma → probar 5 valores → elegir mejor
  3. Repetir para del_t, init_omg, ksize

Total: 5 + 5 + 5 + 5 + 5 = 25 combinaciones

Con GPU: ~1 hora
Sin GPU: ~3-5 horas

FUNCIÓN OBJETIVO:
-----------------
score = |R_final - 0.5| + 0.8×|DFA_α - 1.0| + 0.8×|PSD_slope + 1.0|

Minimizar score → Mejor configuración
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime
import pandas as pd

# Agregar el path del módulo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kuramoto.modelo import KBlock
from datasets.loader import MNISTLoader
from analisis.criticalidad import (
    KuramotoMetrics,
    DFA,
    PSD,
    Entropia,
    MutualInformation,
    Correlacion
)

# Configuración
RESULTS_DIR = "resultados_optimizacion_secuencial"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Parámetros a explorar
PARAM_SEARCH_SPACE = {
    'T_steps': [50, 75, 100, 125, 150],
    'gamma': [0.3, 0.5, 0.7, 0.9, 1.1],
    'del_t': [0.5, 0.7, 0.9, 1.1, 1.3],
    'init_omg': [0.05, 0.1, 0.2, 0.3, 0.5],
    'ksize': [3, 5, 7, 9, 11]
}

# Parámetros fijos iniciales (baseline)
BASELINE_PARAMS = {
    'T_steps': 100,
    'gamma': 0.7,
    'del_t': 0.9,
    'init_omg': 0.1,
    'ksize': 7,
    'ch': 4,
    'n': 4,
    'T_model': 4,
    'img_size': 64
}

# Número de imágenes por evaluación (subset)
N_IMAGES_PER_CONFIG = 100  # 10 por clase × 10 clases
IMAGES_PER_CLASS = 10

# Pesos para función objetivo
WEIGHTS = {
    'R_final': 1.0,
    'DFA_alpha': 0.8,
    'PSD_slope': 0.8
}


def detectar_dispositivo():
    """Detecta si hay GPU disponible."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("⚠️  GPU no disponible, usando CPU")
    return device


def calcular_metricas_imagen(xs, es, params):
    """
    Calcula las 6 métricas de criticalidad para una imagen.
    
    Returns:
        dict con R_final, DFA_alpha, PSD_slope, Magnitud, MI, Entropía
    """
    metricas = {}
    
    try:
        # 1. R_final (valor estacionario)
        R_series = KuramotoMetrics.order_parameter(xs, ch_pair=(0, 1))
        if torch.is_tensor(R_series):
            R_series = R_series.cpu().numpy()
        metricas['R_final'] = float(R_series[-1])  # Último valor
        metricas['R_median'] = float(np.median(R_series))
        
        # 2. Magnitud media por canal
        series = KuramotoMetrics.magnitudes_mean_series(xs)
        if torch.is_tensor(series):
            series = series.cpu().numpy()
        global_series = series.mean(axis=1)
        metricas['Magnitud_mean'] = float(global_series.mean())
        metricas['Magnitud_std'] = float(global_series.std())
        
        # 3. DFA alpha
        try:
            dfa_analyzer = DFA()
            alpha = dfa_analyzer.compute(R_series)
            metricas['DFA_alpha'] = float(alpha) if alpha is not None else np.nan
        except Exception as e:
            metricas['DFA_alpha'] = np.nan
        
        # 4. PSD slope
        try:
            psd_analyzer = PSD()
            slope = psd_analyzer.compute(R_series)
            metricas['PSD_slope'] = float(slope) if slope is not None else np.nan
        except Exception as e:
            metricas['PSD_slope'] = np.nan
        
        # 5. Mutual Information (promedio entre pares de canales)
        try:
            mi_analyzer = MutualInformation()
            mi_matrix = mi_analyzer.compute(xs)
            if mi_matrix is not None:
                # Promedio de valores off-diagonal
                mask = ~np.eye(mi_matrix.shape[0], dtype=bool)
                metricas['MI_mean'] = float(mi_matrix[mask].mean())
            else:
                metricas['MI_mean'] = np.nan
        except Exception as e:
            metricas['MI_mean'] = np.nan
        
        # 6. Entropía Shannon (promedio entre canales)
        try:
            entropy_analyzer = Entropia()
            entropy_dict = entropy_analyzer.compute(xs)
            if entropy_dict:
                entropias = list(entropy_dict.values())
                metricas['Entropy_mean'] = float(np.mean(entropias))
                metricas['Entropy_std'] = float(np.std(entropias))
            else:
                metricas['Entropy_mean'] = np.nan
                metricas['Entropy_std'] = np.nan
        except Exception as e:
            metricas['Entropy_mean'] = np.nan
            metricas['Entropy_std'] = np.nan
        
        metricas['success'] = True
        
    except Exception as e:
        print(f"Error calculando métricas: {e}")
        metricas['success'] = False
        for key in ['R_final', 'DFA_alpha', 'PSD_slope', 'Magnitud_mean', 'MI_mean', 'Entropy_mean']:
            metricas[key] = np.nan
    
    return metricas


def calcular_score(metricas):
    """
    Calcula score de criticalidad.
    
    score = w1×|R_final - 0.5| + w2×|DFA_α - 1.0| + w3×|PSD_slope + 1.0|
    
    Score bajo = más cercano al estado crítico
    """
    score = 0.0
    
    if not np.isnan(metricas['R_final']):
        score += WEIGHTS['R_final'] * abs(metricas['R_final'] - 0.5)
    else:
        score += 10.0  # Penalización por falla
    
    if not np.isnan(metricas['DFA_alpha']):
        score += WEIGHTS['DFA_alpha'] * abs(metricas['DFA_alpha'] - 1.0)
    else:
        score += 10.0
    
    if not np.isnan(metricas['PSD_slope']):
        score += WEIGHTS['PSD_slope'] * abs(metricas['PSD_slope'] + 1.0)
    else:
        score += 10.0
    
    return score


def evaluar_configuracion(params, test_loader, device, n_images=100):
    """
    Evalúa una configuración de parámetros en un subset del dataset.
    
    Args:
        params: dict con parámetros del modelo
        test_loader: DataLoader de MNIST
        device: torch.device
        n_images: número de imágenes a evaluar
    
    Returns:
        dict con métricas agregadas y score
    """
    print(f"\n   Evaluando: T_steps={params['T_steps']}, gamma={params['gamma']:.2f}, "
          f"del_t={params['del_t']:.2f}, init_omg={params['init_omg']:.2f}, ksize={params['ksize']}")
    
    # Inicializar modelo con nuevos parámetros
    kblock = KBlock(
        n=params['n'],
        ch=params['ch'],
        T=params['T_model'],
        ksize=params['ksize'],
        init_omg=params['init_omg']
    ).to(device)
    
    metricas_lista = []
    
    # Procesar subset de imágenes
    with tqdm(total=n_images, desc="      Procesando", leave=False) as pbar:
        count = 0
        for batch_idx, (img_batch, label_batch) in enumerate(test_loader):
            if count >= n_images:
                break
            
            img = img_batch[0].to(device)
            label = label_batch[0].item()
            
            # Forward pass con parámetros específicos
            xs, es = kblock.forward_with_params(
                img,
                T_steps=params['T_steps'],
                gamma=params['gamma'],
                del_t=params['del_t']
            )
            
            # Calcular métricas
            metricas = calcular_metricas_imagen(xs, es, params)
            metricas['label'] = label
            metricas_lista.append(metricas)
            
            count += 1
            pbar.update(1)
    
    # Agregar métricas
    df = pd.DataFrame(metricas_lista)
    
    # Filtrar valores válidos
    df_valid = df[df['success'] == True]
    
    if len(df_valid) == 0:
        print("      ❌ Todas las evaluaciones fallaron")
        return {
            'score': 999.0,
            'n_valid': 0,
            'n_total': len(df)
        }
    
    # Calcular medianas (más robustas que medias)
    resultado = {
        'R_final_median': df_valid['R_final'].median(),
        'DFA_alpha_median': df_valid['DFA_alpha'].median(),
        'PSD_slope_median': df_valid['PSD_slope'].median(),
        'Magnitud_mean': df_valid['Magnitud_mean'].median(),
        'MI_mean': df_valid['MI_mean'].median(),
        'Entropy_mean': df_valid['Entropy_mean'].median(),
        'n_valid': len(df_valid),
        'n_total': len(df)
    }
    
    # Calcular score
    metricas_para_score = {
        'R_final': resultado['R_final_median'],
        'DFA_alpha': resultado['DFA_alpha_median'],
        'PSD_slope': resultado['PSD_slope_median']
    }
    resultado['score'] = calcular_score(metricas_para_score)
    
    print(f"      Score: {resultado['score']:.3f} | "
          f"R={resultado['R_final_median']:.3f} | "
          f"DFA={resultado['DFA_alpha_median']:.2f} | "
          f"PSD={resultado['PSD_slope_median']:.2f}")
    
    return resultado


def optimizar_secuencial(test_loader, device):
    """
    Optimización secuencial: optimiza un parámetro a la vez.
    """
    print("\n" + "="*80)
    print("🔍 OPTIMIZACIÓN SECUENCIAL DE PARÁMETROS")
    print("="*80)
    
    params_actual = BASELINE_PARAMS.copy()
    resultados_completos = []
    
    orden_optimizacion = ['T_steps', 'gamma', 'del_t', 'init_omg', 'ksize']
    
    for param_name in orden_optimizacion:
        print(f"\n{'='*80}")
        print(f"📊 Optimizando: {param_name}")
        print(f"{'='*80}")
        print(f"Configuración actual: {params_actual}")
        print()
        
        valores_a_probar = PARAM_SEARCH_SPACE[param_name]
        resultados_param = []
        
        for valor in valores_a_probar:
            # Crear configuración temporal
            params_temp = params_actual.copy()
            params_temp[param_name] = valor
            
            # Evaluar
            resultado = evaluar_configuracion(params_temp, test_loader, device, N_IMAGES_PER_CONFIG)
            resultado['param_name'] = param_name
            resultado['param_value'] = valor
            resultado.update(params_temp)  # Guardar todos los parámetros
            
            resultados_param.append(resultado)
            resultados_completos.append(resultado)
        
        # Encontrar mejor valor para este parámetro
        df_param = pd.DataFrame(resultados_param)
        mejor_idx = df_param['score'].idxmin()
        mejor_valor = df_param.loc[mejor_idx, 'param_value']
        mejor_score = df_param.loc[mejor_idx, 'score']
        
        # Actualizar configuración
        params_actual[param_name] = mejor_valor
        
        print(f"\n✅ Mejor {param_name}: {mejor_valor} (score: {mejor_score:.3f})")
        print(f"   Configuración actualizada: {params_actual}")
    
    # Guardar resultados
    df_completo = pd.DataFrame(resultados_completos)
    csv_path = os.path.join(RESULTS_DIR, 'resultados_optimizacion_secuencial.csv')
    df_completo.to_csv(csv_path, index=False)
    print(f"\n💾 Resultados guardados en: {csv_path}")
    
    return params_actual, df_completo


def main():
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*15 + "OPTIMIZACIÓN SECUENCIAL DE PARÁMETROS" + " "*27 + "║")
    print("║" + " "*20 + "Fase Exploratoria - Grid Search" + " "*27 + "║")
    print("╚" + "="*78 + "╝")
    
    inicio = time.time()
    
    # Detectar dispositivo
    device = detectar_dispositivo()
    print()
    
    # Configuración
    print("📋 CONFIGURACIÓN:")
    print(f"   • Estrategia: Optimización Secuencial (greedy)")
    print(f"   • Parámetros a optimizar: {list(PARAM_SEARCH_SPACE.keys())}")
    print(f"   • Valores por parámetro: 5")
    print(f"   • Total evaluaciones: {sum(len(v) for v in PARAM_SEARCH_SPACE.values())}")
    print(f"   • Imágenes por evaluación: {N_IMAGES_PER_CONFIG}")
    print()
    
    print("🎯 FUNCIÓN OBJETIVO:")
    print(f"   score = {WEIGHTS['R_final']}×|R_final - 0.5| + "
          f"{WEIGHTS['DFA_alpha']}×|DFA_α - 1.0| + "
          f"{WEIGHTS['PSD_slope']}×|PSD_slope + 1.0|")
    print("   Minimizar score → Acercarse al estado crítico")
    print()
    
    print("📊 PARÁMETROS BASELINE:")
    for key, value in BASELINE_PARAMS.items():
        if key not in ['ch', 'n', 'T_model', 'img_size']:
            print(f"   • {key}: {value}")
    print()
    
    # Cargar dataset
    print("📂 Cargando dataset MNIST...")
    mnist_loader = MNISTLoader(root='./data', batch_size=1, img_size=BASELINE_PARAMS['img_size'])
    _, test_loader = mnist_loader.get_mnist(batch_size=1, train_split=False)
    print(f"✅ Dataset cargado: {len(test_loader.dataset):,} imágenes")
    print()
    
    # Estimar tiempo
    total_evaluaciones = sum(len(v) for v in PARAM_SEARCH_SPACE.values())
    segundos_por_imagen = 0.5 if device.type == 'cuda' else 1.5
    tiempo_estimado = (total_evaluaciones * N_IMAGES_PER_CONFIG * segundos_por_imagen) / 3600
    print(f"⏱️  TIEMPO ESTIMADO: {tiempo_estimado:.1f} horas")
    print(f"   ({segundos_por_imagen}s por imagen × {N_IMAGES_PER_CONFIG} imágenes × {total_evaluaciones} configuraciones)")
    print()
    
    input("Presiona Enter para comenzar la optimización...")
    
    # Ejecutar optimización
    mejor_config, df_resultados = optimizar_secuencial(test_loader, device)
    
    # Resumen final
    print("\n" + "="*80)
    print("🏆 RESULTADO FINAL")
    print("="*80)
    print("\n📊 MEJOR CONFIGURACIÓN ENCONTRADA:")
    for key in ['T_steps', 'gamma', 'del_t', 'init_omg', 'ksize']:
        print(f"   • {key}: {mejor_config[key]}")
    
    # Evaluar configuración final en más imágenes
    print("\n🔬 VALIDACIÓN CON MÁS IMÁGENES (500)...")
    resultado_final = evaluar_configuracion(mejor_config, test_loader, device, n_images=500)
    
    print("\n📈 MÉTRICAS FINALES:")
    print(f"   • R_final: {resultado_final['R_final_median']:.4f} (objetivo: 0.5000)")
    print(f"   • DFA α: {resultado_final['DFA_alpha_median']:.4f} (objetivo: 1.0000)")
    print(f"   • PSD slope: {resultado_final['PSD_slope_median']:.4f} (objetivo: -1.0000)")
    print(f"   • Score total: {resultado_final['score']:.4f}")
    print()
    
    print("📊 MÉTRICAS SECUNDARIAS:")
    print(f"   • Magnitud media: {resultado_final['Magnitud_mean']:.4f}")
    print(f"   • MI media: {resultado_final['MI_mean']:.4f}")
    print(f"   • Entropía media: {resultado_final['Entropy_mean']:.4f}")
    print()
    
    # Comparar con baseline
    print("📊 COMPARACIÓN CON BASELINE:")
    print(f"   Baseline score: ~3.264 (actual)")
    print(f"   Mejor score: {resultado_final['score']:.4f}")
    mejora = ((3.264 - resultado_final['score']) / 3.264) * 100
    print(f"   Mejora: {mejora:.1f}%")
    print()
    
    tiempo_total = (time.time() - inicio) / 3600
    print(f"⏱️  Tiempo total: {tiempo_total:.2f} horas")
    print()
    
    # Guardar configuración óptima
    config_path = os.path.join(RESULTS_DIR, 'mejor_configuracion.txt')
    with open(config_path, 'w') as f:
        f.write("MEJOR CONFIGURACIÓN ENCONTRADA\n")
        f.write("="*50 + "\n\n")
        f.write("Parámetros:\n")
        for key in ['T_steps', 'gamma', 'del_t', 'init_omg', 'ksize']:
            f.write(f"  {key}: {mejor_config[key]}\n")
        f.write(f"\nMétricas (validación con 500 imágenes):\n")
        f.write(f"  R_final: {resultado_final['R_final_median']:.4f}\n")
        f.write(f"  DFA α: {resultado_final['DFA_alpha_median']:.4f}\n")
        f.write(f"  PSD slope: {resultado_final['PSD_slope_median']:.4f}\n")
        f.write(f"  Score: {resultado_final['score']:.4f}\n")
    
    print(f"💾 Configuración óptima guardada en: {config_path}")
    print()


if __name__ == "__main__":
    main()
