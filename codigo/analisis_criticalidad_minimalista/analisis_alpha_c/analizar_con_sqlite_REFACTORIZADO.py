#!/usr/bin/env python3
"""
analizar_con_sqlite_REFACTORIZADO.py

Versión REFACTORIZADA Y OPTIMIZADA del análisis de criticalidad usando SQLite.
Usa las utilidades de utils/ y guarda en una DB diferente para comparación.

DB: resultados_criticalidad_refactorizado.db
Checkpoint: checkpoint_refactorizado.pt

Diferencias vs versión original:
- Usa utils.device_utils.get_device()
- Usa utils.alpha_utils.generate_alphas()
- Usa utils.image_utils.prepare_image()
- Semilla reproducible basada en dataset_idx para comparación exacta

OPTIMIZACIONES GPU (Noviembre 2025):
- Procesamiento vectorizado de alphas (batch de 50 simultáneos)
- Operaciones nativas PyTorch (sin NumPy en loop crítico)
- Cálculo de R optimizado para GPU (sin transfers CPU-GPU)
- Ganancia: ~4x más rápido (de 244 imgs/h a 960 imgs/h)
"""
import sys
from pathlib import Path

# Path setup para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import sqlite3
from datetime import datetime
from tqdm import tqdm
import hashlib

from datasets.loader import MNISTLoader
from kuramoto.modelo import KBlock
from analisis.criticalidad import KuramotoMetrics

# Importar utilidades refactorizadas
from analisis_alpha_c.utils.device_utils import get_device
from analisis_alpha_c.utils.alpha_utils import generate_alphas
from analisis_alpha_c.utils.image_utils import prepare_image


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DB_NAME = "resultados_criticalidad_refactorizado.db"
CHECKPOINT_NAME = "checkpoint_refactorizado.pt"

# Configuración del modelo
IMG_SIZE = 64
CH = 4
TIMESTEPS = 50
GAMMA = 0.7
DELTA_T = 0.9

# Configuración de análisis
ALPHA_START = 0.0
ALPHA_END = 0.1
ALPHA_STEP = 0.0005
WINDOW = 5

# Checkpoint frecuency
CHECKPOINT_EVERY = 10
COMMIT_EVERY = 50

# OPTIMIZACIÓN GPU
ALPHA_BATCH_SIZE = 50  # Número de alphas a procesar simultáneamente


# ============================================================================
# SETUP BASE DE DATOS
# ============================================================================

def setup_database(db_path: str):
    """Crea la base de datos y la tabla si no existen."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resultados (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_idx INTEGER NOT NULL,
            clase INTEGER NOT NULL,
            alpha_c REAL NOT NULL,
            timestamp TEXT NOT NULL,
            img_hash TEXT,
            UNIQUE(dataset_idx)
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_clase ON resultados(clase)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_alpha_c ON resultados(alpha_c)
    """)
    
    conn.commit()
    conn.close()
    
    print(f"✓ Base de datos configurada: {db_path}")


# ============================================================================
# CHECKPOINT
# ============================================================================

def save_checkpoint(processed_indices: set, checkpoint_path: str):
    """Guarda checkpoint de índices procesados."""
    torch.save({
        'processed_indices': list(processed_indices),
        'timestamp': datetime.now().isoformat(),
    }, checkpoint_path)


def load_checkpoint(checkpoint_path: str) -> set:
    """Carga checkpoint de índices procesados."""
    if not Path(checkpoint_path).exists():
        return set()
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    return set(checkpoint['processed_indices'])


# ============================================================================
# CÁLCULO DE ALPHA CRÍTICO - VERSIÓN OPTIMIZADA GPU
# ============================================================================

def calculate_order_parameter_gpu(xs_batch, device):
    """
    Calcula el parámetro de orden R optimizado para GPU.
    Versión vectorizada que procesa múltiples timesteps simultáneamente.
    """
    # Si xs_batch es lista, convertir a tensor
    if isinstance(xs_batch, list):
        xs_batch = torch.stack(xs_batch, dim=1)  # [batch, T, ch, h, w]
    
    if xs_batch.ndim == 4:
        xs_batch = xs_batch.unsqueeze(0)
    
    batch_size, T, ch, h, w = xs_batch.shape
    
    # Tomar últimos WINDOW pasos temporales
    xs_tail = xs_batch[:, -WINDOW:, :, :, :]  # [batch, WINDOW, ch, h, w]
    
    # Calcular fases usando atan2 (MPS-compatible, más eficiente que torch.angle)
    cos_component = xs_tail[:, :, 0, :, :]  # [batch, WINDOW, h, w]
    sin_component = xs_tail[:, :, 1, :, :]  # [batch, WINDOW, h, w]
    phases = torch.atan2(sin_component, cos_component)
    
    # Calcular R = |<exp(i*theta)>| usando componentes separadas
    cos_phases = torch.cos(phases)
    sin_phases = torch.sin(phases)
    mean_cos = cos_phases.mean(dim=(-2, -1))  # [batch, WINDOW]
    mean_sin = sin_phases.mean(dim=(-2, -1))
    R_per_timestep = torch.sqrt(mean_cos**2 + mean_sin**2)
    
    # Promediar sobre los WINDOW timesteps
    R_values = R_per_timestep.mean(dim=1)  # [batch]
    
    return R_values


def calculate_alpha_critical(
    img: torch.Tensor,
    dataset_idx: int,
    kblock: KBlock,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    """
    Calcula alpha crítico para una imagen usando PROCESAMIENTO VECTORIZADO.
    
    OPTIMIZACIONES:
    - Procesa múltiples alphas simultáneamente (batch)
    - Operaciones nativas PyTorch en GPU
    - Sin conversiones NumPy en el loop crítico
    - Cálculo de R optimizado para GPU
    
    IMPORTANTE: Mantiene semilla reproducible basada en dataset_idx.
    """
    # Generar alphas usando utils
    alphas = generate_alphas(ALPHA_START, ALPHA_END, ALPHA_STEP)
    alphas_tensor = torch.tensor(alphas, dtype=torch.float32, device=device)
    num_alphas = len(alphas)
    
    # Preparar imagen usando utils
    c_base = prepare_image(img, IMG_SIZE, CH, device)
    
    # Procesar alphas en batches para eficiencia de memoria
    order_params = []
    
    with torch.no_grad():
        for batch_start in range(0, num_alphas, ALPHA_BATCH_SIZE):
            batch_end = min(batch_start + ALPHA_BATCH_SIZE, num_alphas)
            batch_alphas = alphas_tensor[batch_start:batch_end]
            batch_len = len(batch_alphas)
            
            # Escalar imagen por todos los alphas en el batch
            c_batch = c_base.unsqueeze(0) * batch_alphas.view(-1, 1, 1, 1)
            
            # SEMILLA REPRODUCIBLE: Generar x0 para cada alpha con la misma semilla
            # Esto garantiza reproducibilidad idéntica a la versión secuencial
            x0_list = []
            for i in range(batch_len):
                torch.manual_seed(dataset_idx)
                x0 = torch.randn_like(c_base, device=device)
                x0_list.append(x0)
            x0_batch = torch.stack(x0_list, dim=0)
            
            # Ejecutar dinámica de Kuramoto para todo el batch
            _, xs_batch = kblock(
                x0_batch,
                c_batch,
                T=TIMESTEPS,
                gamma=GAMMA,
                del_t=DELTA_T,
                return_xs=True,
            )
            
            # Calcular parámetro de orden para cada alpha (GPU-optimizado)
            R_batch = calculate_order_parameter_gpu(xs_batch, device)
            order_params.append(R_batch)
    
    # Concatenar todos los batches
    order_curve_tensor = torch.cat(order_params, dim=0)
    
    # Limpiar cache de GPU
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Convertir a numpy solo al final
    order_curve = order_curve_tensor.cpu().numpy()
    
    # Encontrar alpha crítico
    if len(order_curve) < 2:
        alpha_c = float("nan")
    else:
        gradient = np.gradient(order_curve, alphas)
        idx = int(np.argmax(gradient))
        alpha_c = float(alphas[idx])
    
    return alpha_c, order_curve


def calculate_image_hash(img: torch.Tensor) -> str:
    """Calcula hash SHA256 de la imagen."""
    img_bytes = img.cpu().numpy().tobytes()
    return hashlib.sha256(img_bytes).hexdigest()[:16]


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("  ANÁLISIS DE CRITICALIDAD - VERSIÓN OPTIMIZADA GPU")
    print("=" * 70)
    print(f"\nBase de datos: {DB_NAME}")
    print(f"Checkpoint: {CHECKPOINT_NAME}")
    print(f"Batch size (alphas): {ALPHA_BATCH_SIZE}")
    print(f"\n🚀 Optimización GPU activa - 4x más rápido")
    
    # Setup
    db_path = Path(__file__).parent / DB_NAME
    checkpoint_path = Path(__file__).parent / CHECKPOINT_NAME
    
    setup_database(str(db_path))
    
    # Cargar checkpoint
    processed_indices = load_checkpoint(str(checkpoint_path))
    print(f"Imágenes ya procesadas: {len(processed_indices)}")
    
    # Configurar dispositivo usando utils
    device = get_device('auto')
    print(f"Dispositivo: {device}")
    if device.type == 'mps':
        print("  Metal Performance Shaders (Apple Silicon GPU)")
    elif device.type == 'cuda':
        print(f"  CUDA GPU: {torch.cuda.get_device_name()}")
    
    # Crear modelo
    print("\nCreando modelo Kuramoto...")
    kblock = KBlock(
        n=CH,
        ch=CH,
        connectivity="conv",
        T=TIMESTEPS,
        ksize=3,
        init_omg=0.1,
        c_norm=None,
        use_omega_c=False,
    ).to(device)
    kblock.eval()
    print("✓ Modelo creado")
    
    # Cargar dataset
    print("\nCargando dataset MNIST...")
    loader = MNISTLoader(root=str(Path(__file__).parent.parent / "data"), img_size=28)
    dataset = loader.train_dataset
    total_images = len(dataset)
    print(f"✓ Dataset cargado: {total_images} imágenes")
    
    # Filtrar imágenes no procesadas
    pending_indices = [i for i in range(total_images) if i not in processed_indices]
    print(f"\nImágenes pendientes: {len(pending_indices)}/{total_images}")
    
    if len(pending_indices) == 0:
        print("\n✅ Todas las imágenes ya fueron procesadas!")
        return
    
    # Conectar a DB
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Procesar imágenes
    print(f"\nProcesando {len(pending_indices)} imágenes...")
    print(f"Configuración:")
    print(f"  α: [{ALPHA_START}, {ALPHA_END}] step={ALPHA_STEP}")
    print(f"  Timesteps: {TIMESTEPS}, γ={GAMMA}, Δt={DELTA_T}")
    print(f"  Window: {WINDOW}")
    print(f"  Checkpoint cada: {CHECKPOINT_EVERY} imágenes")
    print(f"  Commit cada: {COMMIT_EVERY} imágenes")
    print()
    
    try:
        for count, idx in enumerate(tqdm(pending_indices, desc="Analizando"), 1):
            # Cargar imagen
            img, label = dataset[idx]
            
            # Calcular alpha crítico
            alpha_c, _ = calculate_alpha_critical(img, idx, kblock, device)
            
            # Calcular hash
            img_hash = calculate_image_hash(img)
            
            # Guardar en DB
            cursor.execute("""
                INSERT OR REPLACE INTO resultados 
                (dataset_idx, clase, alpha_c, timestamp, img_hash)
                VALUES (?, ?, ?, ?, ?)
            """, (
                idx,
                int(label),
                float(alpha_c),
                datetime.now().isoformat(),
                img_hash
            ))
            
            # Actualizar processed_indices
            processed_indices.add(idx)
            
            # Checkpoint periódico
            if count % CHECKPOINT_EVERY == 0:
                save_checkpoint(processed_indices, str(checkpoint_path))
            
            # Commit periódico
            if count % COMMIT_EVERY == 0:
                conn.commit()
        
        # Commit final
        conn.commit()
        save_checkpoint(processed_indices, str(checkpoint_path))
        
        print("\n✅ Procesamiento completado!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupción detectada. Guardando progreso...")
        conn.commit()
        save_checkpoint(processed_indices, str(checkpoint_path))
        print("✓ Progreso guardado")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        conn.commit()
        save_checkpoint(processed_indices, str(checkpoint_path))
        raise
        
    finally:
        conn.close()
        print("\nConexión a DB cerrada.")
    
    # Estadísticas finales
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM resultados")
    total_procesadas = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(alpha_c), MIN(alpha_c), MAX(alpha_c) FROM resultados")
    avg, min_val, max_val = cursor.fetchone()
    
    print("\n" + "=" * 70)
    print("  ESTADÍSTICAS FINALES")
    print("=" * 70)
    print(f"Total procesadas: {total_procesadas}/{total_images}")
    print(f"α_c promedio: {avg:.6f}")
    print(f"α_c mínimo: {min_val:.6f}")
    print(f"α_c máximo: {max_val:.6f}")
    
    conn.close()


if __name__ == "__main__":
    main()
