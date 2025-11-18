import sqlite3
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Uso: python plot_curvas_todas_clases_excepto9.py <db_path> [output.png]
if len(sys.argv) < 2:
    print("Uso: python plot_curvas_todas_clases_excepto9.py <db_path> [output.png]")
    sys.exit(1)

DB_PATH = Path(sys.argv[1])
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "curvas_todas_clases_excepto9.png"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

plt.figure(figsize=(12, 8))
total_curvas = 0

# Colores para cada clase
colores = plt.cm.tab10(np.linspace(0, 1, 10))

for clase in range(0, 9):  # Clases 0-8, excluyendo 9
    cursor.execute(f'SELECT c_values, r_values FROM clase_{clase}')
    rows = cursor.fetchall()
    
    if len(rows) == 0:
        continue
    
    for row in rows:
        c_vals = pickle.loads(row[0])
        r_vals = pickle.loads(row[1])
        # Filtrar solo valores C entre 0 y 0.4
        mask = (c_vals >= 0.0) & (c_vals <= 0.4)
        c_filtered = c_vals[mask]
        r_filtered = r_vals[mask]
        if len(c_filtered) > 0:
            plt.plot(c_filtered, r_filtered, alpha=0.08, color=colores[clase])
            total_curvas += 1

conn.close()

plt.title(f"Todas las curvas R vs C (clases 0-8) - Rango [0, 0.4]\n(n={total_curvas} imágenes)", fontsize=14, fontweight='bold')
plt.xlabel("C (acoplamiento)", fontsize=12)
plt.ylabel("R (parámetro de orden)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim([0, 0.4])
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150)
print(f"✅ Gráfico guardado: {OUTPUT}")
print(f"   Total de curvas: {total_curvas}")
