#!/usr/bin/env python3
"""
analizar_con_sqlite_OPTIMIZADO.py

Versión OPTIMIZADA para GPU del análisis de criticalidad.
Mejoras principales:
- Vectorización del barrido de alphas (procesamiento en batch)
- Eliminación de conversiones NumPy (todo en PyTorch)
- Cálculo de fases optimizado para MPS
- Reducción de transferencias GPU↔CPU

Ganancia esperada: 15-60x más rápido que versión original
"""
import os
import sys
import torch
import numpy as np
import argparse
import sqlite3
from datetime import datetime
from tqdm import tqdm
import torchvision.transforms.functional as TF
from pathlib import Path
import hashlib

# Añadir el directorio padre al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kuramoto.modelo import KBlock
from datasets.loader import MNISTLoader


def crear_base_datos(db_path):
    """Crea la base de datos SQLite con las tablas necesarias."""
    conn = sqlite3.connect(db_path, timeout=30.0)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resultados (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_idx INTEGER NOT NULL,
            clase INTEGER NOT NULL,
            alpha_c REAL NOT NULL,
            timestamp TEXT NOT NULL,
            img_hash TEXT,
            UNIQUE(dataset_idx)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    ''')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_clase ON resultados(clase)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_alpha_c ON resultados(alpha_c)')
    
    conn.commit()
    return conn


def guardar_metadata(conn, **kwargs):
    """Guarda metadatos de configuración."""
    cursor = conn.cursor()
    for key, value in kwargs.items():
        cursor.execute('''
            INSERT OR REPLACE INTO metadata (key, value)
            VALUES (?, ?)
        ''', (key, str(value)))
    conn.commit()


def obtener_imagenes_procesadas(conn):
    """Retorna set de índices ya procesados."""
    cursor = conn.cursor()
    cursor.execute('SELECT dataset_idx FROM resultados')
    return set(row[0] for row in cursor.fetchall())


def guardar_resultado(conn, dataset_idx, clase, alpha_c, img_hash=None):
    """Guarda resultado de una imagen con retry en caso de bloqueo."""
    import time
    max_retries = 5
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO resultados 
                (dataset_idx, clase, alpha_c, timestamp, img_hash)
                VALUES (?, ?, ?, ?, ?)
            ''', (dataset_idx, clase, alpha_c, datetime.now().isoformat(), img_hash))
            return True
        except sqlite3.OperationalError as e:
            if 'locked' in str(e) and attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            raise
    return False


def calcular_hash_imagen(img):
    """Calcula hash MD5 de la imagen."""
    img_bytes = img.cpu().numpy().tobytes()
    return hashlib.md5(img_bytes).hexdigest()


def get_device(device_arg='auto'):
    """Detecta y retorna el dispositivo disponible."""
    if device_arg == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_arg)


def calcular_order_parameter_gpu(xs_batch, device):
    """
    Calcula el parámetro de orden R optimizado para GPU.
    
    Args:
        xs_batch: Tensor [batch, T, ch, h, w] o [T, ch, h, w] o lista de tensors
        device: dispositivo (mps, cuda, cpu)
    
    Returns:
        R_values: Tensor [batch] con valores de R para cada alpha
    """
    # Si xs_batch es lista, convertir a tensor
    if isinstance(xs_batch, list):
        xs_batch = torch.stack(xs_batch, dim=1)  # [batch, T, ch, h, w]
    
    if xs_batch.ndim == 4:
        xs_batch = xs_batch.unsqueeze(0)  # [1, T, ch, h, w]
    
    batch_size, T, ch, h, w = xs_batch.shape
    
    # Tomar últimos 5 pasos temporales
    xs_tail = xs_batch[:, -5:, :, :, :]  # [batch, 5, ch, h, w]
    
    # Calcular fases usando atan2 (más eficiente que torch.angle en MPS)
    # Asumimos que los primeros 2 canales son (cos, sin) de la fase
    cos_component = xs_tail[:, :, 0, :, :]  # [batch, 5, h, w]
    sin_component = xs_tail[:, :, 1, :, :]  # [batch, 5, h, w]
    
    # Calcular fases
    phases = torch.atan2(sin_component, cos_component)  # [batch, 5, h, w]
    
    # Calcular parámetro de orden R = |<exp(i*theta)>|
    # Evitar números complejos en MPS usando componentes separadas
    cos_phases = torch.cos(phases)  # [batch, 5, h, w]
    sin_phases = torch.sin(phases)  # [batch, 5, h, w]
    
    # Promediar sobre espacio (h, w)
    mean_cos = cos_phases.mean(dim=(-2, -1))  # [batch, 5]
    mean_sin = sin_phases.mean(dim=(-2, -1))  # [batch, 5]
    
    # Calcular magnitud: R = sqrt(mean_cos² + mean_sin²)
    R_per_timestep = torch.sqrt(mean_cos**2 + mean_sin**2)  # [batch, 5]
    
    # Promediar sobre los 5 timesteps
    R_values = R_per_timestep.mean(dim=1)  # [batch]
    
    return R_values


def calcular_alpha_c_optimizado(kblock, img, alphas, ch, h, w, T, device, batch_size=50):
    """
    Versión OPTIMIZADA: Procesa múltiples alphas en paralelo (batch).
    
    Mejoras:
    1. Vectorización del loop de alphas
    2. Operaciones nativas de PyTorch (sin NumPy)
    3. Cálculo de fases optimizado para MPS
    4. Procesamiento en batches para eficiencia de memoria
    
    Args:
        batch_size: Número de alphas a procesar simultáneamente
                   (ajustar según memoria GPU disponible)
    """
    # Preparar imagen
    if img.ndim == 3 and img.shape[0] == 1:
        img_single = img[0]
    else:
        img_single = img
    
    img_resized = TF.resize(img_single.unsqueeze(0), [h, w])[0]
    img_channels = img_resized.repeat(ch, 1, 1).to(device)  # [ch, h, w]
    
    # Convertir alphas a tensor en GPU
    alphas_tensor = torch.tensor(alphas, dtype=torch.float32, device=device)
    num_alphas = len(alphas)
    
    order_params = []
    
    # Procesar alphas en batches
    with torch.no_grad():
        for batch_start in range(0, num_alphas, batch_size):
            batch_end = min(batch_start + batch_size, num_alphas)
            batch_alphas = alphas_tensor[batch_start:batch_end]  # [batch]
            batch_len = len(batch_alphas)
            
            # Escalar imagen por todos los alphas en el batch
            # [batch, ch, h, w] = [ch, h, w] * [batch, 1, 1, 1]
            c_batch = img_channels.unsqueeze(0) * batch_alphas.view(-1, 1, 1, 1)
            
            # Condiciones iniciales aleatorias para cada alpha
            x0_batch = torch.randn(batch_len, ch, h, w, device=device)
            
            # Ejecutar dinámica de Kuramoto para todo el batch
            _, xs_batch = kblock(
                x0_batch,
                c_batch,
                T=T,
                gamma=0.7,
                del_t=0.9,
                return_xs=True,
            )
            # xs_batch: [batch, T, ch, h, w]
            
            # Calcular parámetro de orden para cada alpha
            R_batch = calcular_order_parameter_gpu(xs_batch, device)
            
            order_params.append(R_batch)
    
    # Concatenar todos los batches
    order_curve = torch.cat(order_params, dim=0)  # [num_alphas]
    
    # Limpiar cache de GPU
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Convertir a numpy solo al final para el gradiente
    order_curve_np = order_curve.cpu().numpy()
    
    # Calcular α_c (máximo gradiente)
    gradient = np.gradient(order_curve_np)
    idx_max = np.argmax(gradient)
    alpha_c = alphas[idx_max]
    
    return float(alpha_c)


def main(args):
    start_time = datetime.now()
    
    print("="*70)
    print("ANÁLISIS DE CRITICALIDAD CON SQLITE - VERSIÓN OPTIMIZADA")
    print("="*70)
    print(f"Inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base de datos: {args.db_path}")
    print(f"Batch size (alphas): {args.alpha_batch_size}")
    print()
    
    # Configuración
    device = get_device(args.device)
    ch, n, h, w, T = 4, 4, 64, 64, 100
    
    print(f"✅ Usando dispositivo: {device}")
    if device.type == 'mps':
        print("   Metal Performance Shaders (Apple Silicon GPU)")
    elif device.type == 'cuda':
        print(f"   CUDA GPU: {torch.cuda.get_device_name()}")
    print()
    
    # Crear/conectar base de datos
    print("Conectando a base de datos...")
    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = crear_base_datos(db_path)
    
    # Guardar metadatos
    guardar_metadata(
        conn,
        inicio=start_time.isoformat(),
        device=str(device),
        version='optimizada_gpu',
        alpha_batch_size=args.alpha_batch_size,
        ch=ch, n=n, h=h, w=w, T=T,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_step=args.alpha_step
    )
    
    # Cargar imágenes ya procesadas
    procesadas = obtener_imagenes_procesadas(conn)
    print(f"✅ Imágenes ya procesadas: {len(procesadas)}")
    
    # Cargar dataset
    print("\nCargando MNIST train set...")
    mn = MNISTLoader(root=args.data_root, batch_size=1, img_size=64)
    train_loader, _ = mn.get_mnist(batch_size=1, train_split=True)
    
    total_imgs = len(train_loader.dataset)
    print(f"Total imágenes en train set: {total_imgs}")
    
    # Modelo
    print(f"\nInicializando modelo en {device}...")
    kblock = KBlock(n=n, ch=ch, T=T, ksize=7, init_omg=0.1).to(device)
    kblock.eval()
    
    # Array de alphas
    alphas = np.arange(args.alpha_min, args.alpha_max + args.alpha_step/2, args.alpha_step)
    print(f"Rango α: [{alphas[0]:.6f}, {alphas[-1]:.6f}], {len(alphas)} puntos")
    print(f"Procesamiento vectorizado: {args.alpha_batch_size} alphas simultáneos")
    print(f"Aceleración estimada: {len(alphas)/args.alpha_batch_size:.1f}x por imagen\n")
    
    # Procesamiento
    print("="*70)
    print("PROCESAMIENTO")
    print("="*70)
    
    procesadas_ahora = 0
    saltadas = 0
    
    with tqdm(total=total_imgs, desc="Procesando") as pbar:
        for dataset_idx, (img_batch, label_batch) in enumerate(train_loader):
            pbar.update(1)
            
            # Saltar si ya fue procesada
            if dataset_idx in procesadas:
                saltadas += 1
                pbar.set_postfix({
                    'nuevas': procesadas_ahora,
                    'saltadas': saltadas
                })
                continue
            
            # Procesar imagen con versión optimizada
            img = img_batch[0]
            clase = int(label_batch[0].item())
            img_hash = calcular_hash_imagen(img)
            
            alpha_c = calcular_alpha_c_optimizado(
                kblock, img, alphas, ch, h, w, T, device,
                batch_size=args.alpha_batch_size
            )
            
            # Guardar en DB
            guardar_resultado(conn, dataset_idx, clase, alpha_c, img_hash)
            procesadas_ahora += 1
            
            pbar.set_postfix({
                'clase': clase,
                'α_c': f'{alpha_c:.5f}',
                'nuevas': procesadas_ahora,
                'saltadas': saltadas
            })
            
            # Commit cada 50 imágenes
            if procesadas_ahora % 50 == 0:
                conn.commit()
                print(f"\n💾 Checkpoint: {procesadas_ahora} imágenes guardadas")
    
    conn.commit()
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*), AVG(alpha_c), MIN(alpha_c), MAX(alpha_c) FROM resultados')
    count, avg, min_val, max_val = cursor.fetchone()
    
    print(f"Total imágenes en DB: {count}")
    print(f"α_c promedio: {avg:.6f}")
    print(f"α_c rango: [{min_val:.6f}, {max_val:.6f}]")
    
    duration = (datetime.now() - start_time).total_seconds() / 60
    imgs_per_min = procesadas_ahora / duration if duration > 0 else 0
    
    print(f"\n⏱️  Duración total: {duration:.1f} minutos")
    print(f"📊 Velocidad: {imgs_per_min:.1f} imágenes/minuto")
    print(f"📊 Equivalente: {imgs_per_min*60:.0f} imágenes/hora")
    print(f"💾 Base de datos: {db_path}")
    
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Análisis de criticalidad OPTIMIZADO para GPU'
    )
    parser.add_argument('--device', type=str, default='auto',
                        help='Dispositivo: auto, cpu, cuda, mps')
    parser.add_argument('--db_path', type=str, 
                        default='resultados_criticalidad_optimizado.db',
                        help='Ruta de la base de datos SQLite')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Directorio raíz de datos MNIST')
    parser.add_argument('--alpha_min', type=float, default=0.0)
    parser.add_argument('--alpha_max', type=float, default=0.1)
    parser.add_argument('--alpha_step', type=float, default=0.0005)
    parser.add_argument('--alpha_batch_size', type=int, default=50,
                        help='Número de alphas a procesar simultáneamente (ajustar según RAM GPU)')
    
    args = parser.parse_args()
    main(args)
