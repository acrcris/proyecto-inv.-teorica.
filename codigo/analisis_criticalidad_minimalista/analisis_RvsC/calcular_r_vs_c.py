"""
Script: calcular_r_vs_c.py

Calcula el parámetro de orden R en función del parámetro de acoplamiento C
para todo el dataset MNIST Training Set.

Basado en run_kuramoto_TRAIN_MAC.py pero enfocado en curvas R vs C completas.
"""

import os
import sys
import sqlite3
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import time

# Agregar path del módulo padre
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kuramoto.modelo import KBlock
from datasets.loader import MNISTLoader

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Parámetros del modelo (IDÉNTICOS al notebook)
CH = 3  # ← Corregido de 4 a 3
N = 3   # ← Corregido de 4 a 3
T_STEPS = 30  # Pasos temporales para estabilización
T_MODEL = 4
KSIZE = 3  # ← Corregido de 7 a 3
INIT_OMG = 0.1
IMG_SIZE = 64

# Parámetros de acoplamiento C
C_MIN = 0.0
C_MAX = 0.4  # ← Rango [0, 0.4]
N_C_POINTS = 40  # ← 40 puntos con espaciado 0.01
c_range = np.arange(C_MIN, C_MAX, 0.01)  # ← Espaciado 0.01 para 40 puntos

# Semilla para reproducibilidad (IDÉNTICA al notebook)
SEED = 1

# Configuración de procesamiento
CHECKPOINT_INTERVAL = 50  # Guardar cada 50 imágenes
BATCH_SIZE = 1

# Directorios
RESULTS_DIR = Path(".")  # Usar directorio actual (analisis_RvsC)
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

DB_PATH = RESULTS_DIR / "mnist_R_vs_C.db"  # ← Usar nueva DB combinada

# =============================================================================
# DETECCIÓN DE DISPOSITIVO
# =============================================================================

def get_device():
    """Detecta el mejor dispositivo disponible"""
    if torch.cuda.is_available():
        return torch.device('cuda'), 'CUDA'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps'), 'MPS (Apple Metal)'
    else:
        return torch.device('cpu'), 'CPU'

DEVICE, DEVICE_NAME = get_device()

# =============================================================================
# REPRODUCIBILIDAD
# =============================================================================

def set_seed(seed):
    """Fija todas las semillas para reproducibilidad completa"""
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, 'mps'):
        # MPS no soporta deterministic algorithms completamente
        pass

set_seed(SEED)

# =============================================================================
# FUNCIÓN DE PARÁMETRO DE ORDEN
# =============================================================================

def kuramoto_order_parameter(xs):
    """
    Calcula el parámetro de orden de Kuramoto R(t)
    
    Args:
        xs: Lista de estados [T, B, C, H, W]
    
    Returns:
        np.array: Valores de R(t) para cada paso temporal
    """
    R_values = []
    
    for x in xs:
        # x tiene shape [B, C, H, W] donde C son canales (pares real/imag)
        if x.ndim == 4:
            x = x[0]  # Tomar primer batch
        
        # Calcular fase de cada oscilador (canales 0,1 son primer oscilador)
        theta = torch.atan2(x[1::2], x[0::2])  # Fase de cada oscilador
        
        # Parámetro de orden: magnitud del promedio de exp(i*theta)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        r_x = cos_theta.mean()
        r_y = sin_theta.mean()
        
        R = torch.sqrt(r_x**2 + r_y**2)
        R_values.append(R.item())
    
    return np.array(R_values)

# =============================================================================
# BASE DE DATOS
# =============================================================================

def inicializar_db(db_path):
    """Crea la base de datos con una tabla por clase"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Crear tabla para cada clase (0-9)
    for clase in range(10):
        tabla = f"clase_{clase}"
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {tabla} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_idx INTEGER NOT NULL,
                c_values BLOB NOT NULL,
                r_values BLOB NOT NULL,
                timestamp TEXT NOT NULL,
                UNIQUE(image_idx)
            )
        ''')
    
    # Crear tabla de metadatos
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    # Guardar configuración
    metadata = {
        'c_min': str(C_MIN),
        'c_max': str(C_MAX),
        'n_c_points': str(N_C_POINTS),
        't_steps': str(T_STEPS),
        'ch': str(CH),
        'n': str(N),
        'device': DEVICE_NAME,
        'created': datetime.now().isoformat()
    }
    
    for key, value in metadata.items():
        cursor.execute('INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)', (key, value))
    
    conn.commit()
    conn.close()

def guardar_resultado_db(db_path, clase, imagen_idx, c_values, r_values):
    """Guarda los resultados R vs C en la base de datos"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    tabla = f"clase_{clase}"
    
    # Serializar arrays con pickle
    c_blob = pickle.dumps(c_values)
    r_blob = pickle.dumps(r_values)
    
    timestamp = datetime.now().isoformat()
    
    try:
        cursor.execute(f'''
            INSERT INTO {tabla} (image_idx, c_values, r_values, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (imagen_idx, c_blob, r_blob, timestamp))
        conn.commit()
    except sqlite3.IntegrityError:
        # Imagen ya procesada, actualizar
        cursor.execute(f'''
            UPDATE {tabla}
            SET c_values = ?, r_values = ?, timestamp = ?
            WHERE image_idx = ?
        ''', (c_blob, r_blob, timestamp, imagen_idx))
        conn.commit()
    finally:
        conn.close()

def verificar_imagen_procesada(db_path, clase, imagen_idx):
    """Verifica si una imagen ya fue procesada"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    tabla = f"clase_{clase}"
    cursor.execute(f'SELECT COUNT(*) FROM {tabla} WHERE image_idx = ?', (imagen_idx,))
    count = cursor.fetchone()[0]
    
    conn.close()
    return count > 0

def contar_procesadas(db_path):
    """Cuenta cuántas imágenes se han procesado por clase"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    conteo = {}
    for clase in range(10):
        tabla = f"clase_{clase}"
        cursor.execute(f'SELECT COUNT(*) FROM {tabla}')
        conteo[clase] = cursor.fetchone()[0]
    
    conn.close()
    return conteo

# =============================================================================
# FUNCIÓN PRINCIPAL DE CÁLCULO
# =============================================================================

def calcular_R_vs_C(imagen, kblock, x_init, c_range, device):
    """
    Calcula R en función de C para una imagen dada.
    
    CRÍTICO: x_init es el MISMO para TODAS las imágenes (igual que notebook)
    
    Args:
        imagen: Tensor [1, 1, H, W] con la imagen
        kblock: Modelo Kuramoto
        x_init: Estado inicial [1, CH, H, W] - MISMO para todas las imágenes
        c_range: Array de valores C a evaluar
        device: Dispositivo (cuda/mps/cpu)
    
    Returns:
        r_values: Array con R correspondiente a cada C
    """
    imagen = imagen.to(device)
    r_values = []
    
    # Preprocesar imagen a CH canales
    img_resized = transforms.functional.resize(imagen, [IMG_SIZE, IMG_SIZE])
    img_channels = img_resized.repeat(1, CH, 1, 1).to(device)
    
    with torch.no_grad():
        for c_val in c_range:
            # Copiar estado inicial (MISMO para todas las imágenes y todos los C)
            x = x_init.clone().to(device)
            
            # Escalar campo de acoplamiento por c_val
            c = img_channels * c_val
            
            # Ejecutar dinámica
            _, xs = kblock(
                x, c,
                T=T_STEPS,
                gamma=0.7,
                del_t=0.9,
                return_xs=True
            )
            
            # Calcular parámetro de orden R(t)
            R_series = kuramoto_order_parameter(xs)
            
            # Usar valor en estado estacionario (último valor)
            R_final = R_series[-1]
            r_values.append(R_final)
            
            # Limpiar memoria en MPS
            if device.type == 'mps':
                torch.mps.empty_cache()
    
    return np.array(r_values)

# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Calcular R vs C para MNIST')
    parser.add_argument('--clases', type=int, nargs='+', default=None, help='Clases a procesar (ej: --clases 9)')
    args = parser.parse_args()
    
    # Seleccionar clases a procesar
    if args.clases:
        clases_a_procesar = args.clases
    else:
        clases_a_procesar = range(10)
    
    print("="*70)
    print("CÁLCULO DE R vs C - MNIST TRAINING SET")
    print("="*70)
    print()
    print("🖥️  DISPOSITIVO:")
    print(f"   - {DEVICE_NAME}")
    print(f"   - PyTorch device: {DEVICE}")
    print()
    print("📌 CONFIGURACIÓN (IDÉNTICA AL NOTEBOOK):")
    print(f"   - CH (canales): {CH}")
    print(f"   - N (dimensión): {N}")
    print(f"   - KSIZE (kernel): {KSIZE}")
    print(f"   - T (pasos): {T_STEPS}")
    print(f"   - Semilla: {SEED}")
    print(f"   - Rango C: [{C_MIN}, {C_MAX}]")
    print(f"   - Puntos C: {len(c_range)}")
    print(f"   - Espaciado C: 0.02")
    print(f"   - Base de datos: {DB_PATH}")
    print(f"   - Checkpoints cada: {CHECKPOINT_INTERVAL} imágenes")
    print(f"   - Clases a procesar: {list(clases_a_procesar)}")
    print()
    
    # Inicializar DB
    print("📂 Inicializando base de datos...")
    inicializar_db(DB_PATH)
    print(f"✅ Base de datos lista: {DB_PATH}")
    print()
    
    # Cargar dataset
    print("📂 Cargando MNIST Training Set...")
    mnist_loader = MNISTLoader(root='../data', batch_size=1, img_size=IMG_SIZE)
    train_loader, _ = mnist_loader.get_mnist(batch_size=1, train_split=True)
    total_images = len(train_loader.dataset)
    print(f"✅ Dataset cargado: {total_images:,} imágenes")
    print()
    
    # Verificar progreso actual
    conteo_actual = contar_procesadas(DB_PATH)
    total_procesadas = sum(conteo_actual.values())
    
    print(f"📊 Progreso actual:")
    print(f"   - Total procesadas: {total_procesadas:,}/{total_images:,}")
    for clase, count in conteo_actual.items():
        print(f"   - Clase {clase}: {count:,} imágenes")
    print()
    
    # Inicializar modelo
    print("🧠 Inicializando modelo Kuramoto...")
    set_seed(SEED)  # Reiniciar semilla antes de crear modelo
    kblock = KBlock(
        n=N, 
        ch=CH, 
        connectivity='conv',
        T=T_MODEL, 
        ksize=KSIZE, 
        init_omg=INIT_OMG,
        c_norm=None,  # ← IMPORTANTE: None como en notebook
        use_omega=True,
        use_omega_c=False  # ← IMPORTANTE: False como en notebook
    )
    kblock = kblock.to(DEVICE)
    kblock.eval()
    print(f"✅ Modelo inicializado en {DEVICE}")
    print(f"   c_norm=None, use_omega_c=False (igual que notebook)")
    print()
    
    # Estado inicial (MISMO para TODAS las imágenes - CRÍTICO)
    print("🎲 Creando estado inicial x_init...")
    x_init = torch.randn(1, CH, IMG_SIZE, IMG_SIZE)
    print(f"✅ Estado inicial creado con forma {x_init.shape}")
    print(f"   ⚠️  CRÍTICO: x_init se reutilizará para TODAS las imágenes")
    print()
    
    # Estimación de tiempo
    imgs_restantes = total_images - total_procesadas
    if DEVICE_NAME.startswith('MPS'):
        velocidad_estimada = 2.5  # img/s para Apple M3
    elif DEVICE_NAME == 'CUDA':
        velocidad_estimada = 4.0  # img/s para GPU NVIDIA
    else:
        velocidad_estimada = 0.5  # img/s para CPU
    
    tiempo_estimado_h = (imgs_restantes / velocidad_estimada) / 3600
    
    print("⏱️  ESTIMACIÓN:")
    print(f"   - Imágenes restantes: {imgs_restantes:,}")
    print(f"   - Velocidad estimada: ~{velocidad_estimada} img/s")
    print(f"   - Tiempo estimado: {tiempo_estimado_h:.1f} horas")
    print()
    
    print("="*70)
    print("PROCESANDO...")
    print("="*70)
    print()
    
    start_time = time.time()
    procesadas_sesion = 0
    saltadas = 0
    procesadas_por_clase = {c: 0 for c in range(10)}
    
    # Obtener conteo actual de la DB
    conteo = contar_procesadas(DB_PATH)
    
    # Contar imágenes totales por clase en el dataset
    imagenes_por_clase = {c: 0 for c in range(10)}
    for _, label in train_loader:
        imagenes_por_clase[label.item()] += 1
    
    # Calcular total de imágenes a procesar (solo de las clases seleccionadas)
    total_a_procesar = sum(imagenes_por_clase[c] for c in clases_a_procesar)
    
    # Barra de progreso con total correcto
    pbar = tqdm(total=total_a_procesar, desc="Procesando imágenes", ncols=120, unit="img")
    
    for idx, (imagen, label) in enumerate(train_loader):
        clase = label.item()
        
        # Filtrar por clases seleccionadas
        if clase not in clases_a_procesar:
            continue
        
        # Verificar si ya fue procesada
        if verificar_imagen_procesada(DB_PATH, clase, idx):
            saltadas += 1
            pbar.update(1)  # Actualizar barra aunque sea saltada
            # Actualizar info en barra de progreso
            info_clases = ' | '.join([f"C{c}:{conteo[c]}/{imagenes_por_clase[c]}" 
                                      for c in clases_a_procesar])
            pbar.set_postfix_str(f"{info_clases} | Skip:{saltadas}")
            continue
        
        # Calcular R vs C
        r_values = calcular_R_vs_C(imagen, kblock, x_init, c_range, DEVICE)
        
        # Guardar en DB
        guardar_resultado_db(DB_PATH, clase, idx, c_range, r_values)
        
        procesadas_sesion += 1
        procesadas_por_clase[clase] += 1
        
        # Actualizar conteo en memoria
        conteo[clase] += 1
        
        # Actualizar barra
        pbar.update(1)
        
        # Actualizar barra con info de todas las clases
        info_clases = ' | '.join([f"C{c}:{conteo[c]}/{imagenes_por_clase[c]}" 
                                  for c in clases_a_procesar])
        pbar.set_postfix_str(f"{info_clases} | R:[{r_values.min():.3f},{r_values.max():.3f}]")
        
        # Checkpoint cada CHECKPOINT_INTERVAL imágenes
        if procesadas_sesion % CHECKPOINT_INTERVAL == 0:
            elapsed = time.time() - start_time
            velocidad = procesadas_sesion / elapsed if elapsed > 0 else 0
            
            # Calcular cuántas imágenes faltan de las clases seleccionadas
            restantes = sum(imagenes_por_clase[c] - conteo[c] for c in clases_a_procesar)
            tiempo_restante = restantes / velocidad if velocidad > 0 else 0
            
            tqdm.write(f"\n💾 Checkpoint: {procesadas_sesion} imágenes procesadas")
            tqdm.write(f"   Velocidad: {velocidad:.2f} img/s")
            tqdm.write(f"   Tiempo restante: {tiempo_restante/3600:.1f}h")
            tqdm.write(f"   Progreso por clase:")
            for c in clases_a_procesar:
                pct = 100 * conteo[c] / imagenes_por_clase[c] if imagenes_por_clase[c] > 0 else 0
                tqdm.write(f"      - Clase {c}: {conteo[c]:,}/{imagenes_por_clase[c]:,} ({pct:.1f}%)")
            tqdm.write("")
    
    pbar.close()
    
    # Resumen final
    elapsed_total = time.time() - start_time
    conteo_final = contar_procesadas(DB_PATH)
    total_final = sum(conteo_final.values())
    
    print("\n" + "="*70)
    print("PROCESAMIENTO COMPLETADO")
    print("="*70)
    print(f"\n📊 Resumen:")
    print(f"   - Procesadas en esta sesión: {procesadas_sesion:,}")
    print(f"   - Saltadas (ya procesadas): {saltadas:,}")
    print(f"   - Total en DB: {total_final:,}/{total_images:,}")
    print(f"   - Tiempo total: {elapsed_total/3600:.2f} horas")
    if procesadas_sesion > 0:
        print(f"   - Velocidad promedio: {procesadas_sesion/elapsed_total:.2f} img/s")
    print(f"\n📁 Base de datos: {DB_PATH}")
    print(f"   Tamaño: {DB_PATH.stat().st_size / (1024**2):.1f} MB")
    print(f"\n✅ Distribución por clase:")
    for clase, count in conteo_final.items():
        porcentaje = (count / (total_images//10)) * 100 if total_images > 0 else 0
        print(f"   Clase {clase}: {count:,} ({porcentaje:.1f}%)")
    print()

if __name__ == "__main__":
    main()
