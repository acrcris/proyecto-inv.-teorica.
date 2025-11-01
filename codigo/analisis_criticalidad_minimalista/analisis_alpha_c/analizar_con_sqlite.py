#!/usr/bin/env python3
"""
analizar_con_sqlite.py

Análisis de criticalidad con base de datos SQLite para máxima robustez.
Guarda cada imagen procesada con su índice único del dataset MNIST.

Ventajas:
- Base de datos transaccional (ACID)
- Identificador único por imagen (índice global en MNIST)
- Reanudación sin repetir imágenes
- Queries SQL para análisis
- Backup incremental fácil

Uso:
    python analizar_con_sqlite.py --device mps --db resultados.db
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kuramoto.modelo import KBlock
from datasets.loader import MNISTLoader


def crear_base_datos(db_path):
    """Crea la base de datos SQLite con las tablas necesarias."""
    conn = sqlite3.connect(db_path, timeout=30.0)  # Timeout de 30 segundos
    cursor = conn.cursor()
    
    # Tabla principal: resultados por imagen
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
    
    # Tabla de metadatos
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    ''')
    
    # Índices para búsquedas rápidas
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
    retry_delay = 1.0  # segundos
    
    for attempt in range(max_retries):
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO resultados 
                (dataset_idx, clase, alpha_c, timestamp, img_hash)
                VALUES (?, ?, ?, ?, ?)
            ''', (dataset_idx, clase, alpha_c, datetime.now().isoformat(), img_hash))
            conn.commit()
            return  # Éxito, salir
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Backoff exponencial
            else:
                raise  # Re-lanzar después de max_retries


def calcular_hash_imagen(img_tensor):
    """Calcula hash SHA256 de la imagen para verificación."""
    img_bytes = img_tensor.cpu().numpy().tobytes()
    return hashlib.sha256(img_bytes).hexdigest()[:16]


def get_device(device_arg='auto'):
    """Detecta el mejor dispositivo disponible."""
    if device_arg == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("✅ Usando GPU: Metal Performance Shaders (MPS)")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✅ Usando GPU: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            device = torch.device('cpu')
            print("⚠️  Usando CPU (GPU no disponible)")
    else:
        device = torch.device(device_arg)
        print(f"✅ Usando dispositivo: {device_arg}")
    
    return device


def calcular_alpha_c(kblock, img, alphas, ch=4, h=64, w=64, T=100, device='cpu'):
    """Calcula α_c para una imagen."""
    # Preparar imagen
    if img.ndim == 3 and img.shape[0] == 1:
        img_single = img[0]
    else:
        img_single = img
    
    img_resized = TF.resize(img_single.unsqueeze(0), [h, w])[0]
    img_channels = img_resized.repeat(ch, 1, 1).to(device)
    
    order_params = []
    
    for alpha in alphas:
        with torch.no_grad():
            c_scaled = img_channels.unsqueeze(0) * alpha
            x0 = torch.randn(1, ch, h, w, device=device)
            
            x_final, xs, es = kblock(
                x0, c_scaled, T=T, gamma=0.7, del_t=0.9,
                return_xs=True, return_es=True
            )
            
            # Calcular R promedio de últimos 5 pasos
            R_t = []
            for x_t in xs[-5:]:
                x_complex = torch.view_as_complex(
                    x_t[0, 0:2].permute(1, 2, 0).contiguous()
                )
                if x_complex.device.type == 'mps':
                    x_complex_cpu = x_complex.cpu()
                    phases = torch.angle(x_complex_cpu)
                else:
                    phases = torch.angle(x_complex)
                R = np.abs(np.mean(np.exp(1j * phases.cpu().detach().numpy())))
                R_t.append(R)
            
            R = np.mean(R_t)
            order_params.append(R)
    
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    
    order_curve = np.array(order_params)
    
    # Calcular α_c (máximo gradiente)
    gradient = np.gradient(order_curve)
    idx_max = np.argmax(gradient)
    alpha_c = alphas[idx_max]
    
    return float(alpha_c)


def main(args):
    start_time = datetime.now()
    
    print("="*70)
    print("ANÁLISIS DE CRITICALIDAD CON SQLITE")
    print("="*70)
    print(f"Inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base de datos: {args.db_path}")
    print()
    
    # Configuración
    device = get_device(args.device)
    clases = list(range(10))  # Todas las clases 0-9
    ch, n, h, w, T = 4, 4, 64, 64, 100
    
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
    
    # Contar imágenes disponibles
    total_imgs = len(train_loader.dataset)
    print(f"Total imágenes en train set: {total_imgs}")
    
    # Modelo
    print(f"\nInicializando modelo en {device}...")
    kblock = KBlock(n=n, ch=ch, T=T, ksize=7, init_omg=0.1).to(device)
    kblock.eval()
    
    # Array de alphas
    alphas = np.arange(args.alpha_min, args.alpha_max + args.alpha_step/2, args.alpha_step)
    print(f"Rango α: [{alphas[0]:.6f}, {alphas[-1]:.6f}], {len(alphas)} puntos")
    
    # Procesamiento
    print("\n" + "="*70)
    print("PROCESAMIENTO")
    print("="*70)
    
    procesadas_ahora = 0
    saltadas = 0
    
    with tqdm(total=total_imgs, desc="Procesando") as pbar:
        for dataset_idx, (img_batch, label_batch) in enumerate(train_loader):
            # Actualizar progreso
            pbar.update(1)
            
            # Saltar si ya fue procesada
            if dataset_idx in procesadas:
                saltadas += 1
                pbar.set_postfix({
                    'nuevas': procesadas_ahora,
                    'saltadas': saltadas
                })
                continue
            
            # Procesar imagen
            img = img_batch[0]
            clase = int(label_batch[0].item())
            img_hash = calcular_hash_imagen(img)
            
            alpha_c = calcular_alpha_c(kblock, img, alphas, ch, h, w, T, device)
            
            # Guardar en DB
            guardar_resultado(conn, dataset_idx, clase, alpha_c, img_hash)
            procesadas_ahora += 1
            
            pbar.set_postfix({
                'clase': clase,
                'α_c': f'{alpha_c:.5f}',
                'nuevas': procesadas_ahora,
                'saltadas': saltadas
            })
            
            # Commit cada 100 imágenes para no perder progreso
            if procesadas_ahora % 100 == 0:
                conn.commit()
                print(f"\n💾 Checkpoint automático: {procesadas_ahora} imágenes guardadas")
    
    # Commit final
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
    
    print("\nDistribución por clase:")
    cursor.execute('''
        SELECT clase, COUNT(*), AVG(alpha_c), STDEV(alpha_c)
        FROM (
            SELECT clase, alpha_c,
                   AVG(alpha_c) OVER (PARTITION BY clase) as mean,
                   (alpha_c - AVG(alpha_c) OVER (PARTITION BY clase)) * 
                   (alpha_c - AVG(alpha_c) OVER (PARTITION BY clase)) as sq_diff
            FROM resultados
        )
        GROUP BY clase
        ORDER BY clase
    ''')
    
    for row in cursor.fetchall():
        clase, count, mean, std = row
        std = std if std is not None else 0
        print(f"  Clase {clase}: {count} imgs, μ={mean:.6f}, σ={std:.6f}")
    
    duration = (datetime.now() - start_time).total_seconds() / 60
    print(f"\nDuración total: {duration:.1f} minutos")
    print(f"Base de datos guardada en: {db_path}")
    
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Análisis de criticalidad con SQLite')
    parser.add_argument('--device', type=str, default='auto',
                        help='Dispositivo: auto, cpu, cuda, mps')
    parser.add_argument('--db_path', type=str, default='resultados_criticalidad.db',
                        help='Ruta de la base de datos SQLite')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Directorio raíz de datos MNIST')
    parser.add_argument('--alpha_min', type=float, default=0.0,
                        help='Valor mínimo de alpha')
    parser.add_argument('--alpha_max', type=float, default=0.1,
                        help='Valor máximo de alpha')
    parser.add_argument('--alpha_step', type=float, default=0.0005,
                        help='Paso de alpha')
    
    args = parser.parse_args()
    main(args)
