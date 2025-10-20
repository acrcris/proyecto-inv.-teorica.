"""
Script: run_kuramoto_full_dataset.py

FASE 2.1 - Análisis con múltiples imágenes por clase

Ejecuta la dinámica de Kuramoto sobre TODO el test set de MNIST (10,000 imágenes).
Para cada imagen:
- Ejecuta KBlock con T=100 pasos
- Guarda estados de evolución
- Calcula métricas de criticalidad

ADVERTENCIA: Este proceso puede tomar varias horas dependiendo del hardware.
Se recomienda ejecutar en modo background o usar nohup.

Resultados guardados en: resultados_kuramoto_full_dataset/
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
from datasets.loader import DatasetFactory
from analisis.criticalidad import (
    KuramotoMetrics,
    DFA,
    PSD,
    Entropia,
    MutualInformation,
    Correlacion
)

# Configuración
RESULTS_DIR = "resultados_kuramoto_full_dataset"
METRICS_DIR = os.path.join(RESULTS_DIR, "metricas")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Parámetros del modelo Kuramoto
CH = 4          # Canales
N = 4           # Dimensión (1 oscilador de 4D por píxel)
T = 100         # Pasos de integración
GAMMA = 0.7     # Acoplamiento
DEL_T = 0.9     # Paso temporal
IMG_SIZE = 64   # Tamaño de imagen

# Control de checkpoints
CHECKPOINT_INTERVAL = 100  # Guardar checkpoint cada 100 imágenes

def calcular_metricas_imagen(xs, es, idx, label):
    """
    Calcula todas las métricas de criticalidad para una imagen.
    
    Args:
        xs: Estados de posición (T+1, CH, N, H, W)
        es: Estados de momentum (T+1, CH, N, H, W)
        idx: Índice de la imagen
        label: Etiqueta de la imagen
    
    Returns:
        dict: Diccionario con todas las métricas
    """
    metricas = {
        'idx': idx,
        'label': int(label),
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # 1. Parámetro de orden de Kuramoto
        R = KuramotoMetrics.order_parameter(xs, ch_pair=(0, 1))
        metricas['R_mean'] = float(R.mean())
        metricas['R_std'] = float(R.std())
        metricas['R_min'] = float(R.min())
        metricas['R_max'] = float(R.max())
        metricas['R_final'] = float(R[-1])
        
        # 2. Series temporales de magnitud
        series = KuramotoMetrics.magnitudes_mean_series(xs)
        global_series = series.mean(axis=1)
        
        # 3. DFA
        try:
            scales, F, alpha = DFA.dfa(global_series)
            metricas['DFA_alpha'] = float(alpha)
        except Exception as e:
            metricas['DFA_alpha'] = np.nan
            metricas['DFA_error'] = str(e)
        
        # 4. PSD
        try:
            f, Pxx, slope = PSD.psd_slope(global_series)
            metricas['PSD_slope'] = float(slope)
        except Exception as e:
            metricas['PSD_slope'] = np.nan
            metricas['PSD_error'] = str(e)
        
        # 5. Entropía de Shannon
        if isinstance(xs, list):
            xs_tensor = torch.stack(xs)
        else:
            xs_tensor = xs
        entropy_results = Entropia.entropy_analysis(xs_tensor)
        entropias = [v['Entropía de Shannon'] for v in entropy_results.values()]
        metricas['Entropia_mean'] = float(np.mean(entropias))
        metricas['Entropia_std'] = float(np.std(entropias))
        
        # 6. Correlación
        corr_matrix = Correlacion.pearson_matrix(series)
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        metricas['Corr_mean'] = float(corr_matrix[mask].mean())
        metricas['Corr_std'] = float(corr_matrix[mask].std())
        
        # 7. Información Mutua
        ch_count = series.shape[1]
        MI = np.zeros((ch_count, ch_count))
        for i in range(ch_count):
            for j in range(ch_count):
                MI[i, j] = MutualInformation.mutual_info(series[:, i], series[:, j])
        metricas['MI_mean'] = float(MI[mask].mean())
        metricas['MI_std'] = float(MI[mask].std())
        
        # 8. Varianza de la serie
        metricas['Variance'] = float(global_series.var())
        
        metricas['success'] = True
        
    except Exception as e:
        metricas['success'] = False
        metricas['error'] = str(e)
    
    return metricas

def guardar_checkpoint(metricas_acumuladas, idx):
    """Guarda un checkpoint con las métricas acumuladas hasta el momento."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{idx:05d}.pt')
    torch.save({
        'metricas': metricas_acumuladas,
        'last_idx': idx,
        'timestamp': datetime.now().isoformat()
    }, checkpoint_path)
    print(f"  💾 Checkpoint guardado: {checkpoint_path}")

def cargar_ultimo_checkpoint():
    """Carga el último checkpoint disponible, si existe."""
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('checkpoint_')]
    if not checkpoints:
        return None, 0
    
    # Ordenar por índice y tomar el último
    checkpoints.sort()
    ultimo = checkpoints[-1]
    checkpoint_path = os.path.join(CHECKPOINT_DIR, ultimo)
    
    print(f"🔄 Cargando checkpoint: {checkpoint_path}")
    data = torch.load(checkpoint_path)
    return data['metricas'], data['last_idx']

def main():
    """Función principal."""
    print("="*70)
    print("FASE 2.1 - ANÁLISIS CON TODO EL TEST SET DE MNIST")
    print("="*70)
    print(f"Fecha de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Cargar dataset
    print("📂 Cargando dataset MNIST...")
    dataset_factory = DatasetFactory(root='./data', img_size=IMG_SIZE)
    _, test_loader = dataset_factory.get_mnist(batch_size=1, train_split=False)
    total_images = len(test_loader.dataset)
    print(f"✅ Dataset cargado: {total_images:,} imágenes en test set")
    print()
    
    # Inicializar modelo Kuramoto
    print("🧠 Inicializando modelo Kuramoto...")
    kblock = KBlock(ch=CH, n=N, gamma=GAMMA, del_t=DEL_T)
    print(f"   - Canales: {CH}")
    print(f"   - Dimensión: {N}")
    print(f"   - Pasos temporales: {T}")
    print(f"   - Gamma: {GAMMA}")
    print(f"   - Delta t: {DEL_T}")
    print()
    
    # Intentar cargar checkpoint previo
    metricas_acumuladas, start_idx = cargar_ultimo_checkpoint()
    if metricas_acumuladas is None:
        metricas_acumuladas = []
        start_idx = 0
        print("🆕 Iniciando desde cero")
    else:
        print(f"🔄 Reanudando desde imagen {start_idx + 1}/{total_images}")
        print(f"   Ya procesadas: {len(metricas_acumuladas)} imágenes")
    print()
    
    # Estimación de tiempo
    print("⏱️  ESTIMACIÓN DE TIEMPO:")
    print("   - Tiempo por imagen: ~1-2 segundos")
    print(f"   - Tiempo total estimado: {(total_images - start_idx) * 1.5 / 3600:.1f} horas")
    print(f"   - Checkpoints cada: {CHECKPOINT_INTERVAL} imágenes")
    print()
    
    print("="*70)
    print("PROCESANDO IMÁGENES...")
    print("="*70)
    
    start_time = time.time()
    
    # Procesar todas las imágenes
    with tqdm(total=total_images, initial=start_idx, desc="Procesando", 
              unit="img", ncols=100) as pbar:
        
        for idx, (image, label) in enumerate(test_loader):
            
            # Si ya procesamos esta imagen en un checkpoint previo, saltarla
            if idx < start_idx:
                continue
            
            try:
                # Ejecutar dinámica de Kuramoto
                xs_list = []
                es_list = []
                
                x, e = kblock.init(image)
                xs_list.append(x.clone())
                es_list.append(e.clone())
                
                for t in range(T):
                    x, e = kblock.forward_dynamics(x, e)
                    xs_list.append(x.clone())
                    es_list.append(e.clone())
                
                xs = torch.stack(xs_list)
                es = torch.stack(es_list)
                
                # Calcular métricas
                metricas = calcular_metricas_imagen(xs, es, idx, label.item())
                metricas_acumuladas.append(metricas)
                
                # Actualizar barra de progreso
                pbar.update(1)
                pbar.set_postfix({
                    'Clase': label.item(),
                    'R': f"{metricas.get('R_mean', 0):.3f}",
                    'DFA': f"{metricas.get('DFA_alpha', 0):.3f}"
                })
                
                # Guardar checkpoint periódicamente
                if (idx + 1) % CHECKPOINT_INTERVAL == 0:
                    guardar_checkpoint(metricas_acumuladas, idx)
                    
                    # Mostrar estadísticas parciales
                    elapsed = time.time() - start_time
                    processed = idx - start_idx + 1
                    rate = processed / elapsed
                    remaining = (total_images - idx - 1) / rate / 3600
                    
                    print(f"\n  📊 Progreso: {idx+1}/{total_images} ({100*(idx+1)/total_images:.1f}%)")
                    print(f"  ⏱️  Velocidad: {rate:.2f} img/s")
                    print(f"  ⏰ Tiempo restante estimado: {remaining:.2f} horas\n")
                
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
    
    final_path = os.path.join(RESULTS_DIR, 'metricas_completas.pt')
    torch.save({
        'metricas': metricas_acumuladas,
        'total_images': total_images,
        'parametros': {
            'ch': CH,
            'n': N,
            'T': T,
            'gamma': GAMMA,
            'del_t': DEL_T,
            'img_size': IMG_SIZE
        },
        'fecha': datetime.now().isoformat()
    }, final_path)
    print(f"✅ Métricas guardadas: {final_path}")
    
    # Estadísticas finales
    elapsed_total = time.time() - start_time
    successful = sum(1 for m in metricas_acumuladas if m.get('success', False))
    failed = len(metricas_acumuladas) - successful
    
    print("\n" + "="*70)
    print("ESTADÍSTICAS FINALES")
    print("="*70)
    print(f"Total de imágenes: {len(metricas_acumuladas):,}")
    print(f"Exitosas: {successful:,} ({100*successful/len(metricas_acumuladas):.1f}%)")
    print(f"Fallidas: {failed:,} ({100*failed/len(metricas_acumuladas):.1f}%)")
    print(f"Tiempo total: {elapsed_total/3600:.2f} horas")
    print(f"Velocidad promedio: {len(metricas_acumuladas)/elapsed_total:.2f} img/s")
    print()
    
    # Distribución por clase
    print("Distribución por clase:")
    labels_procesadas = [m['label'] for m in metricas_acumuladas if m.get('success', False)]
    for clase in range(10):
        count = labels_procesadas.count(clase)
        print(f"  Clase {clase}: {count:,} imágenes")
    
    print("\n" + "="*70)
    print("PROCESAMIENTO COMPLETADO ✅")
    print("="*70)
    print(f"Fecha de finalización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("📁 Resultados guardados en:")
    print(f"   {RESULTS_DIR}/")
    print()
    print("🔬 Siguiente paso: Ejecutar análisis estadístico con:")
    print("   python analizar_estadisticas_full_dataset.py")

if __name__ == "__main__":
    main()
