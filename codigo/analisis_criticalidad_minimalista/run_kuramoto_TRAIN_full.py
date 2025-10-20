"""
Script: run_kuramoto_TRAIN_full.py

OPCIÓN C: TRAINING SET COMPLETO (60,000 imágenes)

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

Para 60,000 imágenes × 101 pasos temporales = 6,060,000 valores de R
Esto garantiza máxima robustez estadística (N≈6,000 por clase)

CARACTERÍSTICAS:
- Dataset: MNIST Training Set (60,000 imágenes)
- ~6,000 muestras por clase (vs ~1,000 del test set)
- Tiempo estimado: 12-30 horas (según velocidad)
- Espacio estimado: ~3 GB
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

# Configuración - TRAINING SET
RESULTS_DIR = "resultados_kuramoto_TRAIN_FULL_60k"
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

CHECKPOINT_INTERVAL = 100  # Guardar cada 100 imágenes

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
        # 1. Parámetro de orden R(t) - SERIE TEMPORAL COMPLETA
        R_series = KuramotoMetrics.parametro_orden(xs)
        if torch.is_tensor(R_series):
            R_series = R_series.cpu().numpy()
        metricas['R_series'] = R_series  # Array de 101 valores
        
        # Valores resumen (para visualización rápida)
        metricas['R_init'] = float(R_series[0])
        metricas['R_final'] = float(R_series[-1])
        
        # 2. Serie global de sincronización - SERIE TEMPORAL COMPLETA
        series = KuramotoMetrics.serie_temporal(xs)
        if torch.is_tensor(series):
            series = series.cpu().numpy()
        global_series = series.mean(axis=1)  # Promedio espacial solamente
        if torch.is_tensor(global_series):
            global_series = global_series.cpu().numpy()
        metricas['global_sync_series'] = global_series  # Array temporal
        
        # 3. Entropía espectral - SERIE TEMPORAL COMPLETA
        if torch.is_tensor(es):
            es = es.cpu().numpy()
        metricas['entropy_series'] = es  # Array de 101 valores
        
        # 4. DFA sobre serie global
        try:
            dfa_alpha, _ = DFA.calcular(global_series, scales=np.logspace(0.5, 2, 20).astype(int))
            metricas['DFA_alpha'] = float(dfa_alpha)
        except:
            metricas['DFA_alpha'] = np.nan
        
        # 5. PSD sobre serie global
        try:
            freqs, psd_vals, slope, intercept = PSD.calcular(global_series, fs=1.0)
            metricas['PSD_slope'] = float(slope)
            metricas['PSD_intercept'] = float(intercept)
            # Guardamos PSD completa para análisis posterior
            metricas['PSD_freqs'] = freqs
            metricas['PSD_values'] = psd_vals
        except:
            metricas['PSD_slope'] = np.nan
            metricas['PSD_intercept'] = np.nan
        
        # 6. Matriz de correlación espacial - promediada temporalmente
        try:
            series_2d = series.reshape(series.shape[0], -1)
            corr_matrix = Correlacion.calcular(series_2d.T)
            if torch.is_tensor(corr_matrix):
                corr_matrix = corr_matrix.cpu().numpy()
            # Promediar correlaciones (excluyendo diagonal)
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            metricas['correlation_mean'] = float(corr_matrix[mask].mean())
            metricas['correlation_std'] = float(corr_matrix[mask].std())
        except:
            metricas['correlation_mean'] = np.nan
            metricas['correlation_std'] = np.nan
        
        # 7. Varianza temporal - SERIE TEMPORAL
        try:
            variance_series = np.var(series, axis=(1, 2, 3))
            if torch.is_tensor(variance_series):
                variance_series = variance_series.cpu().numpy()
            metricas['variance_series'] = variance_series  # Array temporal
            metricas['variance_mean'] = float(variance_series.mean())
        except:
            metricas['variance_mean'] = np.nan
        
        # 8. Información mutua
        try:
            # Usar primeros y últimos frames
            x_inicial = series[0].flatten()
            x_final = series[-1].flatten()
            mi = MutualInformation.calcular(x_inicial, x_final, bins=20)
            metricas['mutual_info'] = float(mi)
        except:
            metricas['mutual_info'] = np.nan
        
        metricas['success'] = True
        
    except Exception as e:
        metricas['success'] = False
        metricas['error'] = str(e)
    
    return metricas


def guardar_checkpoint(metricas_acumuladas, checkpoint_idx):
    """Guarda checkpoint de métricas"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{checkpoint_idx:05d}.pt")
    torch.save({
        'metricas': metricas_acumuladas,
        'checkpoint_idx': checkpoint_idx,
        'timestamp': datetime.now().isoformat()
    }, checkpoint_path)
    return checkpoint_path


def cargar_ultimo_checkpoint():
    """Carga el último checkpoint disponible"""
    checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('checkpoint_')])
    
    if not checkpoints:
        return None, 0
    
    ultimo_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
    data = torch.load(ultimo_checkpoint)
    
    return data['metricas'], data['checkpoint_idx']


def main():
    print("="*70)
    print("ANÁLISIS DE CRITICALIDAD - KURAMOTO SOBRE MNIST TRAIN SET COMPLETO")
    print("="*70)
    print()
    print("📌 CONFIGURACIÓN:")
    print(f"   - Dataset: MNIST Training Set (60,000 imágenes)")
    print(f"   - Pasos temporales: {T_STEPS}")
    print(f"   - Checkpoints cada: {CHECKPOINT_INTERVAL} imágenes")
    print(f"   - Directorio resultados: {RESULTS_DIR}/")
    print(f"Fecha de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Cargar dataset - TRAIN SET
    print("📂 Cargando MNIST Training Set...")
    mnist_loader = MNISTLoader(root='./data', batch_size=1, img_size=IMG_SIZE)
    train_loader, _ = mnist_loader.get_mnist(batch_size=1, train_split=True)
    total_images = len(train_loader.dataset)
    print(f"✅ Dataset cargado: {total_images:,} imágenes en training set")
    print(f"   - Muestras esperadas por clase: ~{total_images//10:,}")
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
    print("⏱️  ESTIMACIÓN (basada en velocidad ~4.5 img/s):")
    imgs_restantes = total_images - start_idx
    tiempo_segundos = imgs_restantes / 4.5
    tiempo_horas = tiempo_segundos / 3600
    print(f"   - Imágenes restantes: {imgs_restantes:,}")
    print(f"   - Tiempo estimado: {tiempo_horas:.1f} horas")
    print(f"   - Tamaño final estimado: ~3 GB")
    print(f"   - Checkpoints totales: {total_images // CHECKPOINT_INTERVAL}")
    print()
    
    print("="*70)
    print("PROCESANDO TRAINING SET...")
    print("="*70)
    
    start_time = time.time()
    
    with tqdm(total=total_images, initial=start_idx, desc="Procesando", 
              unit="img", ncols=100) as pbar:
        
        for idx, (image, label) in enumerate(train_loader):
            
            if idx < start_idx:
                continue
            
            # Procesar imagen
            image = image.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Ejecutar modelo Kuramoto
            xs = []
            es = []
            for t in range(T_STEPS + 1):
                x, e = kblock(image)
                xs.append(x.detach().cpu())
                es.append(e.detach().cpu().item())
            
            # Calcular métricas COMPLETAS
            metricas = calcular_metricas_imagen_completas(xs, es, idx, label.item())
            metricas_acumuladas.append(metricas)
            
            # Actualizar barra de progreso
            pbar.set_postfix({
                'Clase': metricas['label'],
                'R_init': f"{metricas.get('R_init', 0):.3f}",
                'R_final': f"{metricas.get('R_final', 0):.3f}"
            })
            pbar.update(1)
            
            # Guardar checkpoint
            if (idx + 1) % CHECKPOINT_INTERVAL == 0:
                checkpoint_path = guardar_checkpoint(metricas_acumuladas, idx)
                elapsed = time.time() - start_time
                imgs_procesadas = idx + 1 - start_idx
                velocidad = imgs_procesadas / elapsed if elapsed > 0 else 0
                tqdm.write(f"💾 Checkpoint guardado: {checkpoint_path}")
                tqdm.write(f"   Velocidad: {velocidad:.2f} img/s")
    
    # Guardar resultado final
    final_path = os.path.join(RESULTS_DIR, "metricas_completas_TRAIN_60k.pt")
    torch.save({
        'metricas': metricas_acumuladas,
        'total_images': total_images,
        'config': {
            'CH': CH,
            'N': N,
            'T_STEPS': T_STEPS,
            'T_MODEL': T_MODEL,
            'KSIZE': KSIZE,
            'INIT_OMG': INIT_OMG,
            'IMG_SIZE': IMG_SIZE
        },
        'timestamp_final': datetime.now().isoformat(),
        'dataset': 'MNIST_TRAIN_60k'
    }, final_path)
    
    elapsed_total = time.time() - start_time
    
    print()
    print("="*70)
    print("✅ PROCESAMIENTO COMPLETO")
    print("="*70)
    print(f"Total imágenes procesadas: {len(metricas_acumuladas):,}")
    print(f"Tiempo total: {elapsed_total/3600:.2f} horas")
    print(f"Velocidad promedio: {len(metricas_acumuladas)/elapsed_total:.2f} img/s")
    print(f"Archivo final: {final_path}")
    print()
    print("SIGUIENTE PASO:")
    print("   1. Ejecutar: python analizar_distribuciones.py --dataset train")
    print("   2. Verificar normalidad de las 6,060,000 observaciones")
    print("   3. Ejecutar: python analizar_estadisticas_full_dataset.py --dataset train")
    print()


if __name__ == "__main__":
    main()
