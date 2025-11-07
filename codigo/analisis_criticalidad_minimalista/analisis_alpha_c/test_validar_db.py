#!/usr/bin/env python3
"""
test_validar_db.py

Valida que la refactorización produce EXACTAMENTE los mismos valores
que los ya calculados y guardados en la base de datos.

Este es el test DEFINITIVO de equivalencia.

Uso:
    python test_validar_db.py --num-samples 50 --verbose
"""
import sys
from pathlib import Path
import torch
import numpy as np
import sqlite3
import argparse
from tqdm import tqdm

# Asegurar que encuentra los módulos
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.loader import MNISTLoader
from kuramoto.modelo import KBlock
from analisis.criticalidad import KuramotoMetrics

# Importar las funciones refactorizadas
from analisis_alpha_c.utils.device_utils import get_device
from analisis_alpha_c.utils.alpha_utils import generate_alphas
from analisis_alpha_c.utils.image_utils import prepare_image


def recalcular_alpha_critico(
    img: torch.Tensor,
    label: int,
    kblock: KBlock,
    device: torch.device,
    img_size: int = 64,
    ch: int = 4,
    alpha_start: float = 0.0,
    alpha_end: float = 0.1,
    alpha_step: float = 0.0005,
    timesteps: int = 50,
    gamma: float = 0.7,
    delta_t: float = 0.9,
    window: int = 5,
) -> tuple[float, np.ndarray]:
    """
    Recalcula alpha crítico usando las utilidades refactorizadas.
    Debe dar EXACTAMENTE el mismo resultado que está en la DB.
    """
    # Generar alphas con utils
    alphas = generate_alphas(alpha_start, alpha_end, alpha_step)
    
    # Preparar imagen con utils
    c_base = prepare_image(img, img_size, ch, device)
    
    # Calcular curva R(α)
    order_values = []
    
    for alpha in alphas:
        c_scaled = c_base * float(alpha)
        x0 = torch.randn_like(c_base, device=device)
        
        # Agregar dimensión batch
        c_batch = c_scaled.unsqueeze(0)
        x0_batch = x0.unsqueeze(0)
        
        with torch.no_grad():
            _, xs = kblock(
                x0_batch,
                c_batch,
                T=timesteps,
                gamma=gamma,
                del_t=delta_t,
                return_xs=True,
            )
        
        r_series = KuramotoMetrics.order_parameter(xs)
        if len(r_series) == 0:
            order_values.append(0.0)
            continue
        
        tail = r_series[-window:] if window < len(r_series) else r_series
        order_values.append(float(np.mean(tail)))
    
    order_curve = np.asarray(order_values, dtype=np.float64)
    
    # Encontrar alpha crítico
    if len(order_curve) < 2:
        alpha_c = float("nan")
    else:
        gradient = np.gradient(order_curve, alphas)
        idx = int(np.argmax(gradient))
        alpha_c = float(alphas[idx])
    
    return alpha_c, order_curve


def validar_contra_db(
    db_path: str,
    num_samples: int = 50,
    verbose: bool = False,
    tolerance: float = 1e-9
):
    """
    Valida que recalcular imágenes de la DB produce los mismos resultados.
    """
    print("=" * 70)
    print("  VALIDACIÓN CONTRA BASE DE DATOS")
    print("=" * 70)
    print(f"\nDB: {db_path}")
    print(f"Muestras a validar: {num_samples}")
    print(f"Tolerancia: {tolerance}")
    
    # Conectar a la DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Verificar cuántas imágenes hay en la DB
    cursor.execute("SELECT COUNT(*) FROM resultados")
    total_en_db = cursor.fetchone()[0]
    
    print(f"Imágenes en DB: {total_en_db}")
    
    if total_en_db == 0:
        print("\n⚠️  ERROR: La base de datos está vacía!")
        return False
    
    # Limitar número de muestras
    num_samples = min(num_samples, total_en_db)
    
    # Obtener muestras aleatorias de la DB
    cursor.execute("""
        SELECT dataset_idx, clase, alpha_c
        FROM resultados
        ORDER BY RANDOM()
        LIMIT ?
    """, (num_samples,))
    
    muestras = cursor.fetchall()
    conn.close()
    
    print(f"\nValidando {len(muestras)} muestras aleatorias...\n")
    
    # Configurar dispositivo y modelo (igual que analizar_con_sqlite.py)
    device = get_device('auto')
    print(f"Dispositivo: {device}")
    
    kblock = KBlock(
        n=4,
        ch=4,
        connectivity="conv",
        T=50,
        ksize=3,
        init_omg=0.1,
        c_norm=None,
        use_omega_c=False,
    ).to(device)
    kblock.eval()
    
    # Cargar dataset MNIST
    loader = MNISTLoader(root=str(Path("../data")), img_size=28)
    dataset = loader.train_dataset
    
    # Validar cada muestra
    errores = []
    exactos = 0
    
    for idx, clase_db, alpha_c_db in tqdm(muestras, desc="Validando"):
        # Cargar imagen
        img, label = dataset[idx]
        
        # Verificar que la clase coincide
        if label != clase_db:
            print(f"\n⚠️  WARNING: Clase en DB ({clase_db}) != clase real ({label}) para idx={idx}")
            continue
        
        # Recalcular alpha crítico
        # IMPORTANTE: Usar la misma semilla para reproducibilidad
        torch.manual_seed(idx)  # Semilla basada en índice
        np.random.seed(idx)
        
        alpha_c_recalc, _ = recalcular_alpha_critico(
            img, label, kblock, device
        )
        
        # Comparar
        diff = abs(alpha_c_recalc - alpha_c_db)
        
        if diff < tolerance:
            exactos += 1
            if verbose:
                print(f"✓ idx={idx:5d} clase={clase_db} | DB: {alpha_c_db:.6f} | Recalc: {alpha_c_recalc:.6f} | diff: {diff:.2e}")
        else:
            errores.append({
                'idx': idx,
                'clase': clase_db,
                'db': alpha_c_db,
                'recalc': alpha_c_recalc,
                'diff': diff
            })
            print(f"✗ idx={idx:5d} clase={clase_db} | DB: {alpha_c_db:.6f} | Recalc: {alpha_c_recalc:.6f} | diff: {diff:.2e} ⚠️")
    
    # Resumen
    print("\n" + "=" * 70)
    print("  RESUMEN")
    print("=" * 70)
    print(f"Total validado: {len(muestras)}")
    print(f"✅ Exactos: {exactos} ({100*exactos/len(muestras):.1f}%)")
    print(f"❌ Diferentes: {len(errores)} ({100*len(errores)/len(muestras):.1f}%)")
    
    if len(errores) == 0:
        print("\n🎉 PERFECTO: Todos los valores coinciden con la DB!")
        print("   La refactorización produce resultados idénticos.")
        return True
    else:
        print(f"\n⚠️  ADVERTENCIA: {len(errores)} valores difieren de la DB")
        print("\nPosibles causas:")
        print("  1. Diferente semilla aleatoria usada en cálculo original")
        print("  2. Diferente inicialización de x0")
        print("  3. Cambio en el modelo KBlock entre ejecuciones")
        
        if len(errores) <= 10:
            print("\nErrores encontrados:")
            for e in errores:
                print(f"  idx={e['idx']:5d} clase={e['clase']} | "
                      f"DB: {e['db']:.6f} vs Recalc: {e['recalc']:.6f} | "
                      f"diff: {e['diff']:.2e}")
        
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Validar que la refactorización produce los mismos valores que la DB"
    )
    parser.add_argument(
        "--db",
        default="resultados_criticalidad.db",
        help="Ruta a la base de datos SQLite"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=50,
        help="Número de muestras a validar (default: 50)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mostrar todos los resultados"
    )
    parser.add_argument(
        "--tolerance", "-t",
        type=float,
        default=1e-9,
        help="Tolerancia para considerar valores iguales (default: 1e-9)"
    )
    
    args = parser.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"❌ ERROR: Base de datos no encontrada: {db_path}")
        sys.exit(1)
    
    success = validar_contra_db(
        str(db_path),
        num_samples=args.num_samples,
        verbose=args.verbose,
        tolerance=args.tolerance
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
