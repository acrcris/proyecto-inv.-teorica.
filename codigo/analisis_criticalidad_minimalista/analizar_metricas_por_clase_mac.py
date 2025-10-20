"""
Analiza métricas por clase a partir del archivo:
  resultados_kuramoto_TRAIN_MAC_60k/metricas_completas_TRAIN_MAC_60k.pt

Requisitos metodológicos:
- No promediar prematuramente series temporales.
- Extraer métricas por imagen y construir distribuciones por clase.

Métricas extraídas por imagen:
- label (clase 0..9)
- R_stationary = R_series[-1]
- PSD_slope
- DFA_alpha
- MI (mutual_info)
- Entropía de Shannon por canal (vector)
- Magnitud media por canal en estado estacionario (vector)

Salida:
- CSV con una fila por imagen (vectores guardados como JSON compactos)
- Resumen por clase (estadísticos robustos sin asumir normalidad)
"""

import os
import json
import argparse
from typing import Any, Dict

import torch
import numpy as np
import pandas as pd

RESULTS_DIR = "resultados_kuramoto_TRAIN_MAC_60k"
FINAL_FILE = os.path.join(RESULTS_DIR, "metricas_completas_TRAIN_MAC_60k.pt")
OUT_CSV = os.path.join(RESULTS_DIR, "metricas_por_imagen_MAC.csv")
RESUMEN_POR_CLASE = os.path.join(RESULTS_DIR, "resumen_metricas_por_clase_MAC.csv")


def _to_json_array(x: Any) -> str:
    if x is None:
        return json.dumps(None)
    arr = np.asarray(x).tolist()
    return json.dumps(arr, separators=(",", ":"))


def cargar_metricas() -> Dict[str, Any]:
    data = torch.load(FINAL_FILE)
    return data


def construir_dataframe(metricas: Dict[str, Any]) -> pd.DataFrame:
    filas = []
    for m in metricas['metricas']:
        fila = {
            'idx': m.get('idx'),
            'label': m.get('label'),
            'R_stationary': m.get('R_stationary', np.nan),
            'PSD_slope': m.get('PSD_slope', np.nan),
            'DFA_alpha': m.get('DFA_alpha', np.nan),
            'mutual_info': m.get('mutual_info', np.nan),
            'shannon_entropy_by_channel': _to_json_array(m.get('shannon_entropy_by_channel')),
            'mag_channel_mean_final': _to_json_array(m.get('mag_channel_mean_final')),
        }
        filas.append(fila)
    df = pd.DataFrame(filas)
    return df


def resumen_por_clase(df: pd.DataFrame) -> pd.DataFrame:
    # Para vectores (JSON), no calculamos estadísticos aquí; solo para escalares
    cols_escalar = ['R_stationary', 'PSD_slope', 'DFA_alpha', 'mutual_info']
    resumen = (
        df.groupby('label')[cols_escalar]
          .agg(['count', 'median', 'mean', 'std', 'min', 'max'])
    )
    # Aplanar columnas multiíndice
    resumen.columns = [f"{c[0]}_{c[1]}" for c in resumen.columns]
    resumen = resumen.reset_index()
    return resumen


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Cargando métricas desde:", FINAL_FILE)
    data = cargar_metricas()
    print(f"Total imágenes: {len(data['metricas']):,}")

    print("Construyendo DataFrame por imagen...")
    df = construir_dataframe(data)
    df.to_csv(OUT_CSV, index=False)
    print("✅ Guardado:", OUT_CSV)

    print("Construyendo resumen por clase (escalares)...")
    resumen = resumen_por_clase(df)
    resumen.to_csv(RESUMEN_POR_CLASE, index=False)
    print("✅ Guardado:", RESUMEN_POR_CLASE)

    print("\nSiguientes pasos sugeridos:")
    print(" - Visualizar distribuciones por clase (e.g., violin/box plot) para R_stationary, PSD_slope, DFA_alpha, MI")
    print(" - Para vectores por canal: expandir columnas JSON y graficar por canal y clase")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.parse_args()
    main()
