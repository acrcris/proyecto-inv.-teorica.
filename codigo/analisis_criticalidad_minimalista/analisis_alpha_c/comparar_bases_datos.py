#!/usr/bin/env python3
"""
comparar_bases_datos.py

Compara los resultados entre la base de datos ORIGINAL y la REFACTORIZADA.
Analiza diferencias, correlaciones y estadísticas.

Uso:
    python comparar_bases_datos.py --verbose
"""
import sys
from pathlib import Path
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse

# Path setup
sys.path.insert(0, str(Path(__file__).parent.parent))
from analisis_alpha_c.utils.plot_config import setup_matplotlib, save_figure


def cargar_resultados(db_path: str) -> dict:
    """Carga todos los resultados de una base de datos."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT dataset_idx, clase, alpha_c
        FROM resultados
        ORDER BY dataset_idx
    """)
    
    resultados = {}
    for idx, clase, alpha_c in cursor.fetchall():
        resultados[idx] = {
            'clase': clase,
            'alpha_c': alpha_c
        }
    
    conn.close()
    return resultados


def comparar_bases(db_original: str, db_refactorizado: str, verbose: bool = False):
    """Compara las dos bases de datos."""
    print("=" * 70)
    print("  COMPARACIÓN DE BASES DE DATOS")
    print("=" * 70)
    
    # Cargar resultados
    print(f"\nCargando DB original: {db_original}")
    original = cargar_resultados(db_original)
    print(f"  Imágenes: {len(original)}")
    
    print(f"\nCargando DB refactorizada: {db_refactorizado}")
    refactorizado = cargar_resultados(db_refactorizado)
    print(f"  Imágenes: {len(refactorizado)}")
    
    # Encontrar imágenes comunes
    indices_comunes = set(original.keys()) & set(refactorizado.keys())
    print(f"\nImágenes comunes: {len(indices_comunes)}")
    
    if len(indices_comunes) == 0:
        print("⚠️  No hay imágenes en común para comparar!")
        return
    
    # Extraer valores para comparación
    alpha_c_original = []
    alpha_c_refactor = []
    clases = []
    diferencias = []
    
    for idx in sorted(indices_comunes):
        orig = original[idx]
        refac = refactorizado[idx]
        
        alpha_c_original.append(orig['alpha_c'])
        alpha_c_refactor.append(refac['alpha_c'])
        clases.append(orig['clase'])
        diferencias.append(abs(orig['alpha_c'] - refac['alpha_c']))
    
    alpha_c_original = np.array(alpha_c_original)
    alpha_c_refactor = np.array(alpha_c_refactor)
    diferencias = np.array(diferencias)
    clases = np.array(clases)
    
    # Estadísticas
    print("\n" + "=" * 70)
    print("  ESTADÍSTICAS")
    print("=" * 70)
    
    print("\nα_c Original:")
    print(f"  Media: {np.mean(alpha_c_original):.6f}")
    print(f"  Std:   {np.std(alpha_c_original):.6f}")
    print(f"  Min:   {np.min(alpha_c_original):.6f}")
    print(f"  Max:   {np.max(alpha_c_original):.6f}")
    
    print("\nα_c Refactorizado:")
    print(f"  Media: {np.mean(alpha_c_refactor):.6f}")
    print(f"  Std:   {np.std(alpha_c_refactor):.6f}")
    print(f"  Min:   {np.min(alpha_c_refactor):.6f}")
    print(f"  Max:   {np.max(alpha_c_refactor):.6f}")
    
    print("\nDiferencias |α_c_orig - α_c_refac|:")
    print(f"  Media: {np.mean(diferencias):.6f}")
    print(f"  Std:   {np.std(diferencias):.6f}")
    print(f"  Min:   {np.min(diferencias):.6f}")
    print(f"  Max:   {np.max(diferencias):.6f}")
    print(f"  Mediana: {np.median(diferencias):.6f}")
    
    # Correlación
    correlacion = np.corrcoef(alpha_c_original, alpha_c_refactor)[0, 1]
    print(f"\nCorrelación de Pearson: {correlacion:.6f}")
    
    # Test estadístico
    t_stat, p_value = stats.ttest_rel(alpha_c_original, alpha_c_refactor)
    print(f"T-test pareado: t={t_stat:.4f}, p={p_value:.4e}")
    
    if p_value > 0.05:
        print("  → No hay diferencia significativa (p > 0.05)")
    else:
        print("  → Hay diferencia significativa (p < 0.05)")
    
    # Distribución de diferencias
    print(f"\nDistribución de diferencias:")
    percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
    for p in percentiles:
        val = np.percentile(diferencias, p)
        print(f"  p{p:3d}: {val:.6f}")
    
    # Análisis por clase
    print("\n" + "=" * 70)
    print("  DIFERENCIAS POR CLASE")
    print("=" * 70)
    
    for clase in range(10):
        mask = clases == clase
        if np.sum(mask) == 0:
            continue
        
        diffs_clase = diferencias[mask]
        print(f"\nClase {clase} (n={np.sum(mask)}):")
        print(f"  Diff media: {np.mean(diffs_clase):.6f}")
        print(f"  Diff max:   {np.max(diffs_clase):.6f}")
    
    # Gráficas
    print("\n" + "=" * 70)
    print("  GENERANDO GRÁFICAS")
    print("=" * 70)
    
    setup_matplotlib()
    
    # 1. Scatter plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Scatter
    axes[0, 0].scatter(alpha_c_original, alpha_c_refactor, alpha=0.3, s=10)
    axes[0, 0].plot([0, 0.1], [0, 0.1], 'r--', label='y=x')
    axes[0, 0].set_xlabel('α_c Original')
    axes[0, 0].set_ylabel('α_c Refactorizado')
    axes[0, 0].set_title(f'Correlación: {correlacion:.4f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histograma de diferencias
    axes[0, 1].hist(diferencias, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(diferencias), color='r', linestyle='--', 
                       label=f'Media: {np.mean(diferencias):.6f}')
    axes[0, 1].set_xlabel('|α_c_orig - α_c_refac|')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title('Distribución de Diferencias')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Diferencias vs índice
    indices_list = sorted(indices_comunes)
    axes[1, 0].plot(indices_list, diferencias, alpha=0.5, linewidth=0.5)
    axes[1, 0].set_xlabel('Índice de imagen')
    axes[1, 0].set_ylabel('|Diferencia|')
    axes[1, 0].set_title('Diferencias vs Índice')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot por clase
    diferencias_por_clase = [diferencias[clases == c] for c in range(10)]
    axes[1, 1].boxplot(diferencias_por_clase, labels=range(10))
    axes[1, 1].set_xlabel('Clase')
    axes[1, 1].set_ylabel('|Diferencia|')
    axes[1, 1].set_title('Diferencias por Clase')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'comparacion_original_vs_refactorizado.png')
    print("✓ Gráfica guardada: comparacion_original_vs_refactorizado.png")
    
    # Resumen final
    print("\n" + "=" * 70)
    print("  CONCLUSIÓN")
    print("=" * 70)
    
    if np.mean(diferencias) < 0.001:
        print("✅ Las diferencias son MÍNIMAS (< 0.001 en promedio)")
        print("   Las dos implementaciones son prácticamente equivalentes.")
    elif np.mean(diferencias) < 0.01:
        print("⚠️  Hay diferencias PEQUEÑAS (< 0.01 en promedio)")
        print("   Posiblemente debido a inicialización aleatoria.")
    else:
        print("❌ Hay diferencias SIGNIFICATIVAS (> 0.01 en promedio)")
        print("   Revisar la implementación refactorizada.")
    
    if verbose:
        print("\nPrimeros 20 valores para inspección:")
        print(f"{'Idx':>6} {'Clase':>6} {'Original':>10} {'Refactor':>10} {'Diff':>10}")
        print("-" * 60)
        for i, idx in enumerate(sorted(indices_comunes)[:20]):
            orig = original[idx]
            refac = refactorizado[idx]
            diff = abs(orig['alpha_c'] - refac['alpha_c'])
            print(f"{idx:6d} {orig['clase']:6d} {orig['alpha_c']:10.6f} "
                  f"{refac['alpha_c']:10.6f} {diff:10.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Comparar bases de datos original vs refactorizada"
    )
    parser.add_argument(
        "--db-original",
        default="resultados_criticalidad.db",
        help="Ruta a la DB original"
    )
    parser.add_argument(
        "--db-refactorizado",
        default="resultados_criticalidad_refactorizado.db",
        help="Ruta a la DB refactorizada"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mostrar información detallada"
    )
    
    args = parser.parse_args()
    
    # Verificar que existen las DBs
    if not Path(args.db_original).exists():
        print(f"❌ ERROR: DB original no encontrada: {args.db_original}")
        sys.exit(1)
    
    if not Path(args.db_refactorizado).exists():
        print(f"❌ ERROR: DB refactorizada no encontrada: {args.db_refactorizado}")
        sys.exit(1)
    
    comparar_bases(args.db_original, args.db_refactorizado, args.verbose)


if __name__ == "__main__":
    main()
