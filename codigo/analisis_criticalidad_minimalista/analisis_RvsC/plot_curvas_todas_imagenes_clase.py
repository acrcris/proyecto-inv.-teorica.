import sqlite3
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Uso: python plot_curvas_todas_imagenes_clase.py <db_path> <clase> [output.png]
if len(sys.argv) < 3:
    print("Uso: python plot_curvas_todas_imagenes_clase.py <db_path> <clase> [output.png]")
    sys.exit(1)

DB_PATH = Path(sys.argv[1])
CLASE = int(sys.argv[2])
OUTPUT = sys.argv[3] if len(sys.argv) > 3 else f"curvas_R_vs_C_todas_imagenes_clase{CLASE}.png"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute(f'SELECT c_values, r_values FROM clase_{CLASE}')
rows = cursor.fetchall()
conn.close()

if len(rows) == 0:
    print(f"No hay datos para clase {CLASE}")
    sys.exit(0)

plt.figure(figsize=(10, 7))
total_puntos = 0
for i, row in enumerate(rows):
    c_vals = pickle.loads(row[0])
    r_vals = pickle.loads(row[1])
    # Filtrar solo valores C entre 0 y 0.3
    mask = (c_vals >= 0.0) & (c_vals <= 0.3)
    c_filtered = c_vals[mask]
    r_filtered = r_vals[mask]
    if len(c_filtered) > 0:
        plt.plot(c_filtered, r_filtered, alpha=0.15, color='blue')
        total_puntos += len(c_filtered)

plt.title(f"Todas las curvas R vs C de la clase {CLASE} (n_curvas={len(rows):,}, n_puntos={total_puntos:,}) - Rango [0, 0.3]")
plt.xlabel("C (acoplamiento)")
plt.ylabel("R (parámetro de orden)")
plt.grid(True, alpha=0.3)
plt.xlim([0, 0.3])
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150)
plt.show()
print(f"✅ Gráfico guardado: {OUTPUT}")
