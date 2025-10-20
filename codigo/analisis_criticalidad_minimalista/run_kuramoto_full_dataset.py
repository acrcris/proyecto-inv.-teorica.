"""
Script: run_kuramoto_full_dataset_CORRECTED.py

VERSIÓN CORREGIDA - Sin promedios prematuros

CORRECCIÓN METODOLÓGICA:
========================
En lugar de guardar:
  R_mean = mean(R(t))  ← 1 valor por imagen

Guardamos:
  R_series = [R(t=0), R(t=1), ..., R(t=100)]  ← 101 valores por imagen

Esto permite:
  1. Analizar la distribución REAL de R en cada momento t
  2. Decidir si usar media/mediana DESPUÉS de verificar normalidad
  3. Análisis temporal completo de la dinámica

Para 10,000 imágenes × 101 pasos temporales = 1,010,000 valores de R
Esto sí permite análisis robusto de distribuciones.

ADVERTENCIA: Archivos más grandes (~500 MB vs ~50 MB)
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime

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
RESULTS_DIR = "resultados_kuramoto_full_dataset_CORRECTED"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Parámetros del modelo Kuramoto
CH = 4
N = 4
T_STEPS = 100
T_MODEL = 4
KSIZE = 7
INIT_OMG = 0.1
IMG_SIZE = 64

CHECKPOINT_INTERVAL = 100

def calcular_metricas_imagen_completas(xs, es, idx, label):
    """
    Calcula métricas SIN promediar prematuramente.
    Guarda series temporales completas.
    
    CORRECCIÓN: En lugar de R_mean, guardamos R_series completa
    """
    # Convertir a tensors
    if isinstance(xs, list):
        xs = torch.stack([x.detach() if x.requires_grad else x for x in xs])
    if isinstance(es, list):
        es = torch.tensor([e.detach().item() if torch.is_tensor(e) else e for e in es])
    
    metricas = {
        'idx': idx,
        'label': int(label),
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # 1. Parámetro de orden de Kuramoto - SERIE COMPLETA
        R_series = KuramotoMetrics.order_parameter(xs, ch_pair=(0, 1))
        if torch.is_tensor(R_series):
            R_series = R_series.cpu().numpy()
        metricas['R_series'] = R_series  # Array de 101 valores
        metricas['R_length'] = len(R_series)
        
        # 2. Series temporales de magnitud - COMPLETA
        series = KuramotoMetrics.magnitudes_mean_series(xs)
        if torch.is_tensor(series):
            series = series.cpu().numpy()
        global_series = series.mean(axis=1)
        if torch.is_tensor(global_series):
            global_series = global_series.cpu().numpy()
        metricas['global_series'] = global_series  # Array de 101 valores
        
        # 3. Energías - SERIE COMPLETA
        if torch.is_tensor(es):
            es_numpy = es.cpu().numpy()
        else:
            es_numpy = np.array(es)
        metricas['energy_series'] = es_numpy  # Array de 101 valores
        
        # 4. DFA - guardamos alpha pero también los datos para recalcular
        try:
            scales, F, alpha = DFA.dfa(global_series)
            metricas['DFA_alpha'] = float(alpha)
            metricas['DFA_scales'] = scales
            metricas['DFA_F'] = F
        except Exception as e:
            metricas['DFA_alpha'] = np.nan
            metricas['DFA_error'] = str(e)
        
        # 5. PSD - guardamos slope pero también los datos para recalcular
        try:
            f, Pxx, slope = PSD.psd_slope(global_series)
            metricas['PSD_slope'] = float(slope)
            metricas['PSD_freqs'] = f
            metricas['PSD_power'] = Pxx
        except Exception as e:
            metricas['PSD_slope'] = np.nan
            metricas['PSD_error'] = str(e)
        
        # 6. Entropía - SERIE POR CANAL (sin promediar)
        if isinstance(xs, list):
            xs_tensor = torch.stack(xs)
        else:
            xs_tensor = xs
        entropy_results = Entropia.entropy_analysis(xs_tensor)
        
        # Guardar entropía por canal, no promediada
        metricas['entropy_by_channel'] = {
            k: v['Entropía de Shannon'] for k, v in entropy_results.items()
        }
        
        # 7. Correlación - MATRIZ COMPLETA (sin promediar)
        corr_matrix = Correlacion.pearson_matrix(series)
        if torch.is_tensor(corr_matrix):
            corr_matrix = corr_matrix.cpu().numpy()
        metricas['correlation_matrix'] = corr_matrix
        
        # 8. Información Mutua - MATRIZ COMPLETA (sin promediar)
        ch_count = series.shape[1]
        MI = np.zeros((ch_count, ch_count))
        for i in range(ch_count):
            for j in range(ch_count):
                MI[i, j] = MutualInformation.mutual_info(series[:, i], series[:, j])
        metricas['MI_matrix'] = MI
        
        # 9. Varianza temporal - SERIE COMPLETA
        variance_series = np.var(series, axis=1)  # Varianza por tiempo
        if torch.is_tensor(variance_series):
            variance_series = variance_series.cpu().numpy()
        metricas['variance_series'] = variance_series
        
        metricas['success'] = True
        
    except Exception as e:
        metricas['success'] = False
        metricas['error'] = str(e)
    
    return metricas

def guardar_checkpoint(metricas_acumuladas, idx):
    """Guarda checkpoint - archivos grandes debido a series completas."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{idx:05d}.pt')
    torch.save({
        'metricas': metricas_acumuladas,
        'last_idx': idx,
        'timestamp': datetime.now().isoformat()
    }, checkpoint_path)
    
    # Calcular tamaño
    size_mb = os.path.getsize(checkpoint_path) / 1024 / 1024
    print(f"  💾 Checkpoint guardado: {checkpoint_path} ({size_mb:.1f} MB)")

def cargar_ultimo_checkpoint():
    """Carga el último checkpoint."""
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('checkpoint_')]
    if not checkpoints:
        return None, 0
    
    checkpoints.sort()
    ultimo = checkpoints[-1]
    checkpoint_path = os.path.join(CHECKPOINT_DIR, ultimo)
    
    print(f"🔄 Cargando checkpoint: {checkpoint_path}")
    data = torch.load(checkpoint_path)
    return data['metricas'], data['last_idx']

def main():
    """Función principal."""
    print("="*70)
    print("VERSIÓN CORREGIDA - SIN PROMEDIOS PREMATUROS")
    print("="*70)
    print()
    print("📊 CORRECCIÓN METODOLÓGICA:")
    print("  ✗ Antes: Guardábamos R_mean = mean(R(t))")
    print("  ✓ Ahora: Guardamos R_series = [R(t=0), R(t=1), ..., R(t=100)]")
    print()
    print("  Esto permite:")
    print("  1. Analizar distribución REAL sin colapsar datos")
    print("  2. Decidir media vs mediana DESPUÉS de verificar normalidad")
    print("  3. Análisis temporal completo de la dinámica")
    print()
    print(f"Fecha de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Cargar dataset
    print("📂 Cargando dataset MNIST...")
    mnist_loader = MNISTLoader(root='./data', batch_size=1, img_size=IMG_SIZE)
    _, test_loader = mnist_loader.get_mnist(batch_size=1, train_split=False)
    total_images = len(test_loader.dataset)
    print(f"✅ Dataset cargado: {total_images:,} imágenes en test set")
    print()
    
    # Inicializar modelo
    print("🧠 Inicializando modelo Kuramoto...")
    kblock = KBlock(n=N, ch=CH, T=T_MODEL, ksize=KSIZE, init_omg=INIT_OMG)
    print(f"   - Pasos temporales: {T_STEPS}")
    print(f"   - Por imagen: {T_STEPS+1} valores de R(t)")
    print(f"   - Total dataset: {total_images * (T_STEPS+1):,} valores de R")
    print()
    
    # Intentar cargar checkpoint
    metricas_acumuladas, start_idx = cargar_ultimo_checkpoint()
    if metricas_acumuladas is None:
        metricas_acumuladas = []
        start_idx = 0
        print("🆕 Iniciando desde cero")
    else:
        print(f"🔄 Reanudando desde imagen {start_idx + 1}/{total_images}")
        print(f"   Ya procesadas: {len(metricas_acumuladas)} imágenes")
    print()
    
    # Estimación
    print("⏱️  ESTIMACIÓN:")
    print("   - Tiempo por imagen: ~1.5 segundos")
    print(f"   - Tiempo total: {(total_images - start_idx) * 1.5 / 3600:.1f} horas")
    print(f"   - Tamaño final estimado: ~500 MB (vs ~50 MB versión promediada)")
    print()
    
    print("="*70)
    print("PROCESANDO...")
    print("="*70)
    
    start_time = time.time()
    
    with tqdm(total=total_images, initial=start_idx, desc="Procesando", 
              unit="img", ncols=100) as pbar:
        
        for idx, (image, label) in enumerate(test_loader):
            
            if idx < start_idx:
                continue
            
            try:
                c = image.repeat(1, CH, 1, 1)
                x = torch.randn(1, CH, IMG_SIZE, IMG_SIZE)
                
                x_final, xs, es = kblock(x, c, T=T_STEPS, gamma=0.7, del_t=0.9, 
                                        return_xs=True, return_es=True)
                
                # Calcular métricas COMPLETAS (sin promediar)
                metricas = calcular_metricas_imagen_completas(xs, es, idx, label.item())
                metricas_acumuladas.append(metricas)
                
                pbar.update(1)
                pbar.set_postfix({
                    'Clase': label.item(),
                    'R_init': f"{metricas['R_series'][0]:.3f}",
                    'R_final': f"{metricas['R_series'][-1]:.3f}"
                })
                
                if (idx + 1) % CHECKPOINT_INTERVAL == 0:
                    guardar_checkpoint(metricas_acumuladas, idx)
                    
                    elapsed = time.time() - start_time
                    processed = idx - start_idx + 1
                    rate = processed / elapsed
                    remaining = (total_images - idx - 1) / rate / 3600
                    
                    print(f"\n  📊 Progreso: {idx+1}/{total_images} ({100*(idx+1)/total_images:.1f}%)")
                    print(f"  ⏱️  Velocidad: {rate:.2f} img/s")
                    print(f"  ⏰ Tiempo restante: {remaining:.2f} horas\n")
                
            except Exception as e:
                print(f"\n⚠️  Error en imagen {idx}: {e}")
                metricas_acumuladas.append({
                    'idx': idx,
                    'label': int(label.item()),
                    'success': False,
                    'error': str(e)
                })
                pbar.update(1)
                continue
    
    # Guardar resultados finales
    print("\n" + "="*70)
    print("GUARDANDO RESULTADOS FINALES...")
    print("="*70)
    
    final_path = os.path.join(RESULTS_DIR, 'metricas_completas_CORRECTED.pt')
    torch.save({
        'metricas': metricas_acumuladas,
        'total_images': total_images,
        'T_steps': T_STEPS,
        'parametros': {
            'ch': CH,
            'n': N,
            'T_steps': T_STEPS,
            'T_model': T_MODEL,
            'ksize': KSIZE,
            'init_omg': INIT_OMG,
            'img_size': IMG_SIZE
        },
        'fecha': datetime.now().isoformat()
    }, final_path)
    
    size_mb = os.path.getsize(final_path) / 1024 / 1024
    print(f"✅ Resultados guardados: {final_path} ({size_mb:.1f} MB)")
    
    # Estadísticas
    exitosas = sum(1 for m in metricas_acumuladas if m.get('success', False))
    print()
    print("="*70)
    print("RESUMEN")
    print("="*70)
    print(f"Total procesadas: {len(metricas_acumuladas):,}")
    print(f"Exitosas: {exitosas:,} ({100*exitosas/len(metricas_acumuladas):.1f}%)")
    print(f"Total valores de R: {exitosas * (T_STEPS+1):,}")
    print(f"Tiempo total: {(time.time()-start_time)/3600:.2f} horas")
    print()
    print("📊 DATOS GUARDADOS SIN PROMEDIAR:")
    print("  • R_series: 101 valores por imagen")
    print("  • global_series: 101 valores por imagen")
    print("  • energy_series: 101 valores por imagen")
    print("  • correlation_matrix: Matriz completa")
    print("  • MI_matrix: Matriz completa")
    print()
    print("✓ Ahora se puede analizar la distribución CORRECTAMENTE")
    print("="*70)

if __name__ == "__main__":
    main()
