#!/usr/bin/env python3
"""Script para calcular R en el punto de criticalidad para imágenes MNIST.

Basado en `calcular_c_critico_local.py`, pero en lugar de barrer C para
cada imagen, usamos un valor fijo C_crit (por defecto 0.1769) y
calculamos el parámetro de orden de Kuramoto R(t) con `KuramotoMetrics`.

Guardamos en una base de datos SQLite un valor R_critico por imagen y
clase, con la misma estructura de auto-reinicio y estadísticas.
"""

import os
import sys
import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

import numpy as np
from tqdm.auto import tqdm
import random
import json

# Añadir path del módulo principal
SCRIPT_DIR = Path(__file__).parent
ANALISIS_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ANALISIS_ROOT))

from kuramoto import KBlock
from analisis.criticalidad import KuramotoMetrics


# ========================= CONFIGURACIÓN GLOBAL =========================

SEED = 1

# Parámetros por defecto (CH=3, N=3)
PARAMS_DEFAULT = {
    'seed': SEED,
    'ch': 3,
    'n': 3,
    'h': 64,
    'w': 64,
    'T': 30,
    'gamma': 0.7,
    'del_t': 0.9,
    'ksize': 3,
    'init_omg': 0.1,
}

# Parámetros CH=4, N=4 (como run_kuramoto_TRAIN_MAC.py)
PARAMS_CH4_N4 = {
    'seed': SEED,
    'ch': 4,
    'n': 4,
    'h': 64,
    'w': 64,
    'T': 100,
    'gamma': 0.7,
    'del_t': 0.9,
    'ksize': 7,
    'init_omg': 0.1,
}

PARAMS = PARAMS_DEFAULT.copy()

# Valor de C en el punto de criticalidad (puede ajustarse por CLI)
C_CRIT_DEFAULT = 0.1769


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.use_deterministic_algorithms(False)


def cleanup_mps_resources() -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()
        import gc
        gc.collect()


# ===================== CÁLCULO DE R EN CRITICALIDAD =====================


def kuramoto_R_series(xs):
    """Envuelve KuramotoMetrics.order_parameter para xs (lista de tensores)."""
    return KuramotoMetrics.order_parameter(xs, ch_pair=(0, 1))


def calculate_R_critico(img, kblock, x_init, device, c_crit,
                         ch=3, h=64, w=64, T=30, gamma=0.7, del_t=0.9):
    """Calcula R_critico para una imagen fija y un C_crit dado.

    - Preprocesa la imagen a (h, w) y la replica a `ch` canales
    - Ejecuta la dinámica Kuramoto con KBlock durante T pasos
    - Obtiene la serie R(t) y devuelve R_final = R(t_final)
    """
    img_resized = TF.resize(img, [h, w])
    img_channels = img_resized.repeat(ch, 1, 1).to(device)

    x = x_init.clone().to(device)
    c = img_channels.unsqueeze(0) * c_crit

    with torch.no_grad():
        x_final, xs = kblock(
            x,
            c,
            T=T,
            gamma=gamma,
            del_t=del_t,
            return_xs=True,
            return_es=False,
        )

    R_series = kuramoto_R_series(xs)
    R_final = float(R_series[-1])

    return {
        'R_critico': R_final,
        'R_series': R_series.tolist(),
    }


# ============================ BASE DE DATOS =============================


def create_database(db_path: Path) -> None:
    """Crea la BD SQLite con tablas para cada clase que guardan R_critico."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    for clase in range(10):
        cursor.execute(
            f'''
            CREATE TABLE IF NOT EXISTS clase_{clase} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_idx INTEGER UNIQUE NOT NULL,
                R_critico REAL NOT NULL,
                timestamp TEXT NOT NULL,
                seed INTEGER,
                n INTEGER,
                ch INTEGER,
                T INTEGER,
                gamma REAL,
                del_t REAL,
                c_crit REAL
            )
            '''
        )

    conn.commit()
    conn.close()
    print(f"✅ Base de datos R creada/verificada: {db_path}")


def save_result(db_path: Path, clase: int, image_idx: int, result: dict,
                params: dict, c_crit: float) -> None:
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    timestamp = datetime.now().isoformat()

    cursor.execute(
        f'''
        INSERT OR REPLACE INTO clase_{clase}
        (image_idx, R_critico, timestamp, seed, n, ch, T, gamma, del_t, c_crit)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            image_idx,
            result['R_critico'],
            timestamp,
            params['seed'],
            params['n'],
            params['ch'],
            params['T'],
            params['gamma'],
            params['del_t'],
            c_crit,
        ),
    )

    conn.commit()
    conn.close()


def get_processed_images(db_path: Path, clase: int):
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        cursor.execute(f'SELECT image_idx FROM clase_{clase}')
        processed = set(row[0] for row in cursor.fetchall())
    except sqlite3.OperationalError:
        processed = set()

    conn.close()
    return processed


def get_statistics(db_path: Path, clase: int):
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        cursor.execute(f'SELECT R_critico FROM clase_{clase}')
        vals = np.array([row[0] for row in cursor.fetchall()])
        if len(vals) > 0:
            stats = {
                'n_images': int(len(vals)),
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'median': float(np.median(vals)),
            }
        else:
            stats = None
    except sqlite3.OperationalError:
        stats = None

    conn.close()
    return stats


# ======================= PROCESAMIENTO POR CLASE ========================


def procesar_clase_R(
    clase,
    dataset,
    imagenes_por_clase,
    kblock,
    x_init,
    device,
    db_path: Path,
    c_crit: float,
    limite=None,
    verbose=True,
):
    if verbose:
        print(f"\n{'='*60}")
        print(f"PROCESANDO R_critico CLASE {clase}")
        print(f"{'='*60}")

    indices = imagenes_por_clase[clase]
    if limite:
        indices = indices[:limite]

    processed_images = get_processed_images(db_path, clase)
    indices_pendientes = [idx for idx in indices if idx not in processed_images]

    if verbose:
        print(f"✅ Imágenes ya procesadas: {len(processed_images)}")
        print(f"⏳ Imágenes pendientes: {len(indices_pendientes)}")

    if len(indices_pendientes) == 0:
        if verbose:
            print(f"✨ Clase {clase} ya está completamente procesada (R_critico).")
        return get_statistics(db_path, clase)

    for i, idx in enumerate(tqdm(indices_pendientes, desc=f"Clase {clase}", disable=not verbose)):
        img, label = dataset[idx]
        result = calculate_R_critico(
            img,
            kblock,
            x_init,
            device,
            c_crit=c_crit,
            ch=PARAMS['ch'],
            h=PARAMS['h'],
            w=PARAMS['w'],
            T=PARAMS['T'],
            gamma=PARAMS['gamma'],
            del_t=PARAMS['del_t'],
        )

        save_result(db_path, clase, idx, result, PARAMS, c_crit)

        if device.type == 'mps' and (i + 1) % 100 == 0:
            cleanup_mps_resources()

    if device.type == 'mps':
        cleanup_mps_resources()

    stats = get_statistics(db_path, clase)
    if verbose and stats:
        print(f"\n📊 Estadísticas R_critico Clase {clase}:")
        print(f"  Imágenes: {stats['n_images']}")
        print(f"  R_critico medio: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Rango: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Mediana: {stats['median']:.4f}")

    return stats


# =============================== MAIN ===================================


def main():
    parser = argparse.ArgumentParser(
        description='Calcular R_critico para imágenes MNIST usando KuramotoMetrics',
    )
    parser.add_argument('--clases', type=int, nargs='+', help='Clases a procesar (0-9)')
    parser.add_argument('--all', action='store_true', help='Procesar todas las clases')
    parser.add_argument('--limite', type=int, default=None, help='Límite de imágenes por clase')
    parser.add_argument(
        '--db',
        type=str,
        default='resultados_c_critical/mnist_R_critico.db',
        help='Ruta de la base de datos SQLite para R_critico',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Dispositivo a usar',
    )
    parser.add_argument(
        '--c-crit',
        type=float,
        default=C_CRIT_DEFAULT,
        help='Valor de C en el punto de criticalidad',
    )
    parser.add_argument(
        '--use-ch4-n4',
        action='store_true',
        help='Usar parámetros CH=4, N=4, T=100, KSIZE=7 (gamma=0.7, del_t=0.9)',
    )

    args = parser.parse_args()

    if not args.all and not args.clases:
        parser.error('Debes especificar --clases o --all')

    if args.all:
        clases_a_procesar = list(range(10))
    else:
        clases_a_procesar = args.clases

    # Ajustar parámetros según opción
    global PARAMS
    if args.use_ch4_n4:
        PARAMS = PARAMS_CH4_N4.copy()
        print("🔧 Usando parámetros CH=4, N=4, T=100, KSIZE=7, gamma=0.7, del_t=0.9")
    else:
        PARAMS = PARAMS_DEFAULT.copy()
        print("🔧 Usando parámetros por defecto: CH=3, N=3, T=30, KSIZE=3")

    set_seed(SEED)
    print(f"🔧 Semilla configurada: {SEED}")

    # Selección de dispositivo
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🖥️  Usando dispositivo: CUDA")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"🖥️  Usando dispositivo: MPS (Apple Silicon)")
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        else:
            device = torch.device('cpu')
            print(f"🖥️  Usando dispositivo: CPU")
    else:
        device = torch.device(args.device)
        print(f"🖥️  Usando dispositivo: {device}")

    print(f"📦 PyTorch version: {torch.__version__}")

    # Directorio de salida y BD
    output_dir = SCRIPT_DIR / 'resultados_c_critical'
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / Path(args.db).name

    create_database(db_path)

    # Dataset MNIST - siempre desde torchvision
    print("\n📥 Cargando dataset MNIST desde torchvision...")
    transform = transforms.Compose([transforms.ToTensor()])
    data_dir = ANALISIS_ROOT.parent.parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = torchvision.datasets.MNIST(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    print(f"✅ Dataset cargado: {len(train_dataset)} imágenes")

    imagenes_por_clase = {i: [] for i in range(10)}
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        imagenes_por_clase[label].append(i)

    print("\n📊 Distribución por clase:")
    for clase in range(10):
        print(f"  Clase {clase}: {len(imagenes_por_clase[clase])} imágenes")

    # Modelo KBlock
    print("\n🧠 Inicializando KBlock para R_critico:")
    for k, v in PARAMS.items():
        print(f"  {k}: {v}")

    set_seed(SEED)
    kblock = KBlock(
        n=PARAMS['n'],
        ch=PARAMS['ch'],
        connectivity='conv',
        T=PARAMS['T'],
        ksize=PARAMS['ksize'],
        init_omg=PARAMS['init_omg'],
        c_norm=None,
        use_omega_c=False,
    ).to(device)

    x_init = torch.randn(1, PARAMS['ch'], PARAMS['h'], PARAMS['w'])
    print(f"✅ Modelo en {device}, estado inicial {x_init.shape}")
    print(f"✅ C_crit usado: {args.c_crit}")

    resultados = {}
    for clase in clases_a_procesar:
        stats = procesar_clase_R(
            clase,
            train_dataset,
            imagenes_por_clase,
            kblock,
            x_init,
            device,
            db_path,
            c_crit=args.c_crit,
            limite=args.limite,
            verbose=True,
        )
        resultados[clase] = stats

    print("\n📊 RESUMEN R_critico:")
    for clase, stats in sorted(resultados.items()):
        if stats:
            print(f"Clase {clase}: N={stats['n_images']}  mean={stats['mean']:.4f} ± {stats['std']:.4f}")

    # Guardar resumen JSON
    resumen_path = output_dir / 'resumen_R_critico.json'
    with open(resumen_path, 'w') as f:
        json.dump(resultados, f, indent=2)
    print(f"\n✅ Resumen R_critico guardado en: {resumen_path}")


if __name__ == '__main__':
    main()
