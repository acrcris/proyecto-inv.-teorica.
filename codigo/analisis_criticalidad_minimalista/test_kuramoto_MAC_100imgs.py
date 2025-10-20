"""
Script: test_kuramoto_MAC_100imgs.py

TEST RÁPIDO: 10 imágenes por clase (100 totales)

Valida:
- Que MPS funcione correctamente
- Que se calculen todas las métricas solicitadas
- Que el formato de salida sea correcto
- Velocidad de procesamiento

Resultados en: test_resultados_MAC_100imgs/
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime
from collections import defaultdict

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

# Configuración - TEST
RESULTS_DIR = "test_resultados_MAC_100imgs"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Parámetros del modelo Kuramoto
CH = 4
N = 4
T_STEPS = 100
T_MODEL = 4
KSIZE = 7
INIT_OMG = 0.1
IMG_SIZE = 64

# TEST: 10 imágenes por clase
IMGS_PER_CLASS = 10

# Detectar dispositivo (MPS para Mac, CUDA para NVIDIA, CPU como fallback)
def get_device():
    """Detecta el mejor dispositivo disponible"""
    if torch.cuda.is_available():
        return torch.device('cuda'), 'CUDA'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps'), 'MPS (Apple Metal)'
    else:
        return torch.device('cpu'), 'CPU'

DEVICE, DEVICE_NAME = get_device()


def _shannon_entropy(arr: np.ndarray, bins: int = 64) -> float:
    """Entropía de Shannon (bits) de un arreglo 1D usando histogramas."""
    x = np.asarray(arr).ravel()
    if x.size == 0:
        return np.nan
    counts, _ = np.histogram(x, bins=bins)
    total = counts.sum()
    if total == 0:
        return np.nan
    p = counts[counts > 0].astype(np.float64) / total
    return float(-(p * np.log2(p)).sum())


def calcular_metricas_imagen_completas(xs, es, idx, label):
    """Calcula todas las métricas solicitadas"""
    # Convertir a tensors y mover a CPU para análisis
    if isinstance(xs, list):
        xs = torch.stack([x.cpu().detach() if x.requires_grad else x.cpu() for x in xs])
    if isinstance(es, list):
        es = torch.tensor([e.cpu().detach().item() if torch.is_tensor(e) else e for e in es])
    
    metricas = {
        'idx': idx,
        'label': int(label),
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # 1. Parámetro de orden R(t) - SERIE TEMPORAL COMPLETA
        R_series = KuramotoMetrics.order_parameter(xs)
        if torch.is_tensor(R_series):
            R_series = R_series.cpu().numpy()
        metricas['R_series'] = R_series  # Array de 101 valores
        metricas['R_init'] = float(R_series[0])
        metricas['R_final'] = float(R_series[-1])
        metricas['R_stationary'] = float(R_series[-1])  # Valor estacionario
        
        # 2. Serie global de sincronización
        series = KuramotoMetrics.magnitudes_mean_series(xs)
        if torch.is_tensor(series):
            series = series.cpu().numpy()
        global_series = series.mean(axis=1)  # Promedio sobre canales
        if torch.is_tensor(global_series):
            global_series = global_series.cpu().numpy()
        metricas['global_sync_series'] = global_series  # Array temporal
        
        # 2b. Magnitud media por canal en estado estacionario
        try:
            # series tiene forma (T, ch) -> último frame
            mag_ch = series[-1]  # shape (ch,)
            metricas['mag_channel_mean_final'] = mag_ch.astype(np.float32)
        except Exception:
            metricas['mag_channel_mean_final'] = None
        
        # 2c. Entropía de Shannon por canal (estado estacionario)
        try:
            # Necesitamos el último frame completo para cada canal
            x_final = xs[-1]
            if torch.is_tensor(x_final):
                x_final = x_final.cpu().numpy()
            # x_final shape: [1, ch, H, W] o [ch, H, W]
            if x_final.ndim == 4:
                x_final = x_final[0]  # [ch, H, W]
            entropias = [
                _shannon_entropy(x_final[c], bins=64) for c in range(x_final.shape[0])
            ]
            metricas['shannon_entropy_by_channel'] = np.asarray(entropias, dtype=np.float32)
        except Exception:
            metricas['shannon_entropy_by_channel'] = None
        
        # 3. Entropía espectral - SERIE TEMPORAL COMPLETA
        if torch.is_tensor(es):
            es = es.cpu().numpy()
        metricas['entropy_series'] = es
        
        # 4. DFA sobre serie global
        try:
            scales, F, dfa_alpha = DFA.dfa(global_series, scales=np.logspace(0.5, 2, 20).astype(int))
            metricas['DFA_alpha'] = float(dfa_alpha)
        except:
            metricas['DFA_alpha'] = np.nan
        
        # 5. PSD sobre serie global
        try:
            freqs, psd_vals, slope = PSD.psd_slope(global_series, fs=1.0)
            metricas['PSD_slope'] = float(slope)
            metricas['PSD_freqs'] = freqs
            metricas['PSD_values'] = psd_vals
        except:
            metricas['PSD_slope'] = np.nan
        
        # 6. Matriz de correlación espacial
        try:
            corr_matrix = Correlacion.pearson_matrix(series)
            if torch.is_tensor(corr_matrix):
                corr_matrix = corr_matrix.cpu().numpy()
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            metricas['correlation_mean'] = float(corr_matrix[mask].mean())
            metricas['correlation_std'] = float(corr_matrix[mask].std())
        except:
            metricas['correlation_mean'] = np.nan
            metricas['correlation_std'] = np.nan
        
        # 7. Varianza temporal
        try:
            # series tiene forma (T, ch) -> varianza sobre tiempo por canal
            variance_series = np.var(series, axis=1)  # varianza en canales por tiempo
            metricas['variance_series'] = variance_series
            metricas['variance_mean'] = float(variance_series.mean())
        except:
            metricas['variance_mean'] = np.nan
        
        # 8. Información mutua (entre primera y segunda mitad de la serie)
        try:
            # Dividir la serie global en dos mitades
            mid = len(global_series) // 2
            first_half = global_series[:mid]
            second_half = global_series[mid:mid + len(first_half)]  # Mismo tamaño
            
            # Calcular MI con bins reducidos para más robustez
            mi = MutualInformation.mutual_info(first_half, second_half, bins=16)
            metricas['mutual_info'] = float(mi) if np.isfinite(mi) else np.nan
        except Exception as e:
            metricas['mutual_info'] = np.nan
        
        metricas['success'] = True
        
    except Exception as e:
        metricas['success'] = False
        metricas['error'] = str(e)
    
    return metricas


def main():
    print("="*70)
    print("TEST RÁPIDO - 100 IMÁGENES (10 POR CLASE)")
    print("VERSIÓN APPLE M3 (MPS/Metal)")
    print("="*70)
    print()
    print("🖥️  DISPOSITIVO:")
    print(f"   - {DEVICE_NAME}")
    print(f"   - PyTorch device: {DEVICE}")
    print()
    print("📌 CONFIGURACIÓN TEST:")
    print(f"   - Imágenes por clase: {IMGS_PER_CLASS}")
    print(f"   - Total imágenes: {IMGS_PER_CLASS * 10}")
    print(f"   - Pasos temporales: {T_STEPS}")
    print(f"   - Directorio resultados: {RESULTS_DIR}/")
    print()
    
    # Cargar dataset - TRAIN SET
    print("📂 Cargando MNIST Training Set...")
    mnist_loader = MNISTLoader(root='./data', batch_size=1, img_size=IMG_SIZE)
    train_loader, _ = mnist_loader.get_mnist(batch_size=1, train_split=True)
    print(f"✅ Dataset cargado: {len(train_loader.dataset):,} imágenes disponibles")
    print()
    
    # Inicializar modelo
    print("🧠 Inicializando modelo Kuramoto...")
    kblock = KBlock(n=N, ch=CH, T=T_MODEL, ksize=KSIZE, init_omg=INIT_OMG)
    kblock = kblock.to(DEVICE)
    print(f"   - Modelo en: {DEVICE}")
    print()
    
    # Seleccionar 10 imágenes por clase
    print("🎯 Seleccionando imágenes balanceadas por clase...")
    class_counters = defaultdict(int)
    selected_indices = []
    
    for idx, (image, label) in enumerate(train_loader):
        label_val = label.item()
        if class_counters[label_val] < IMGS_PER_CLASS:
            selected_indices.append((idx, label_val))
            class_counters[label_val] += 1
            
        # Verificar si ya tenemos suficientes
        if all(count >= IMGS_PER_CLASS for count in class_counters.values()):
            break
    
    print(f"✅ Seleccionadas {len(selected_indices)} imágenes")
    for clase in sorted(class_counters.keys()):
        print(f"   Clase {clase}: {class_counters[clase]} imágenes")
    print()
    
    # Procesar imágenes seleccionadas
    print("="*70)
    print("PROCESANDO TEST...")
    print("="*70)
    
    metricas_acumuladas = []
    start_time = time.time()
    
    with tqdm(total=len(selected_indices), desc="Test", unit="img", ncols=100) as pbar:
        for img_num, (idx, expected_label) in enumerate(selected_indices):
            # Obtener imagen
            image, label = train_loader.dataset[idx]
            image = image.unsqueeze(0).to(DEVICE)  # Añadir batch dimension
            
            # Preparar campo de acoplamiento (c) y estado inicial (x)
            c = image.repeat(1, CH, 1, 1)  # Replicar en canales
            x = torch.randn(1, CH, IMG_SIZE, IMG_SIZE).to(DEVICE)  # Estado inicial aleatorio
            
            # Ejecutar modelo Kuramoto
            x_final, xs, es = kblock(
                x, c,
                T=T_STEPS,
                gamma=0.7,
                del_t=0.9,
                return_xs=True,
                return_es=True
            )
            
            # Calcular métricas
            metricas = calcular_metricas_imagen_completas(xs, es, idx, label)
            metricas_acumuladas.append(metricas)
            
            # Actualizar barra
            pbar.set_postfix({
                'Clase': metricas['label'],
                'R_stat': f"{metricas.get('R_stationary', 0):.3f}",
                'DFA': f"{metricas.get('DFA_alpha', 0):.3f}",
                'MI': f"{metricas.get('mutual_info', 0):.3f}"
            })
            pbar.update(1)
    
    elapsed_total = time.time() - start_time
    velocidad = len(selected_indices) / elapsed_total
    
    # Guardar resultado
    final_path = os.path.join(RESULTS_DIR, "metricas_test_100imgs_MAC.pt")
    torch.save({
        'metricas': metricas_acumuladas,
        'total_images': len(selected_indices),
        'imgs_per_class': IMGS_PER_CLASS,
        'config': {
            'CH': CH,
            'N': N,
            'T_STEPS': T_STEPS,
            'T_MODEL': T_MODEL,
            'KSIZE': KSIZE,
            'INIT_OMG': INIT_OMG,
            'IMG_SIZE': IMG_SIZE
        },
        'timestamp': datetime.now().isoformat(),
        'dataset': 'MNIST_TRAIN_test_100imgs',
        'device': DEVICE_NAME,
        'platform': 'Apple M3'
    }, final_path)
    
    print()
    print("="*70)
    print("✅ TEST COMPLETO")
    print("="*70)
    print(f"Total imágenes procesadas: {len(metricas_acumuladas)}")
    print(f"Tiempo total: {elapsed_total:.2f} segundos ({elapsed_total/60:.1f} min)")
    print(f"Velocidad: {velocidad:.2f} img/s")
    print(f"Dispositivo: {DEVICE_NAME}")
    print(f"Archivo guardado: {final_path}")
    print()
    
    # Verificar métricas
    print("📊 VERIFICACIÓN DE MÉTRICAS:")
    ejemplos_ok = sum(1 for m in metricas_acumuladas if m.get('success', False))
    print(f"   Éxito: {ejemplos_ok}/{len(metricas_acumuladas)}")
    
    if ejemplos_ok > 0:
        m_ejemplo = next(m for m in metricas_acumuladas if m.get('success', False))
        print(f"\n   Ejemplo (clase {m_ejemplo['label']}):")
        print(f"   - R_stationary: {m_ejemplo.get('R_stationary', 'N/A'):.4f}")
        print(f"   - PSD_slope: {m_ejemplo.get('PSD_slope', 'N/A'):.4f}")
        print(f"   - DFA_alpha: {m_ejemplo.get('DFA_alpha', 'N/A'):.4f}")
        print(f"   - mutual_info: {m_ejemplo.get('mutual_info', 'N/A'):.4f}")
        
        mag = m_ejemplo.get('mag_channel_mean_final')
        if mag is not None:
            print(f"   - mag_channel_mean_final: {mag} (shape: {mag.shape})")
        
        shannon = m_ejemplo.get('shannon_entropy_by_channel')
        if shannon is not None:
            print(f"   - shannon_entropy_by_channel: {shannon} (shape: {shannon.shape})")
    
    print()
    print("⏱️  ESTIMACIÓN PARA 60,000 IMÁGENES:")
    tiempo_60k_segundos = 60000 / velocidad
    tiempo_60k_horas = tiempo_60k_segundos / 3600
    print(f"   A esta velocidad ({velocidad:.2f} img/s):")
    print(f"   - Tiempo estimado: {tiempo_60k_horas:.1f} horas")
    print()
    print("📊 SIGUIENTE PASO - Analizar resultados del test:")
    print("   python3 analizar_test_100imgs_mac.py")
    print()


if __name__ == "__main__":
    main()
