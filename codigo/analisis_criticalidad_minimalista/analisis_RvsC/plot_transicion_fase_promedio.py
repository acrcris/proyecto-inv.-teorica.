#!/usr/bin/env python3
"""
Genera gráfica de transición de fase promediando R sobre todas las imágenes.
Muestra el rango lineal alrededor de C_crítico = 0.1769
"""

import sqlite3
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Configuración
DB_PATH = Path("mnist_R_vs_C_COMBINED.db")
C_CRITICO = 0.1769
OUTPUT = "transicion_fase_sincronizacion.png"

# Rango de interés (zona lineal de transición)
C_MIN_PLOT = 0.0
C_MAX_PLOT = 0.25

print("Cargando datos de la base de datos...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Cargar todos los datos de todas las clases
all_c_values = []
all_r_values = []

for clase in range(10):
    print(f"  Cargando clase {clase}...")
    cursor.execute(f'SELECT c_values, r_values FROM clase_{clase}')
    rows = cursor.fetchall()
    
    for row in rows:
        c_vals = pickle.loads(row[0])
        r_vals = pickle.loads(row[1])
        
        # Filtrar por rango de interés
        mask = (c_vals >= C_MIN_PLOT) & (c_vals <= C_MAX_PLOT)
        c_filtered = c_vals[mask]
        r_filtered = r_vals[mask]
        
        if len(c_filtered) > 0:
            all_c_values.append(c_filtered)
            all_r_values.append(r_filtered)

conn.close()

print(f"\nTotal de curvas cargadas: {len(all_c_values)}")

# Interpolar todos los datos a una grilla común
print("Interpolando datos a grilla común...")
c_grid = np.linspace(C_MIN_PLOT, C_MAX_PLOT, 200)
r_interpolated = []

for c_vals, r_vals in zip(all_c_values, all_r_values):
    r_interp = np.interp(c_grid, c_vals, r_vals)
    r_interpolated.append(r_interp)

r_interpolated = np.array(r_interpolated)

# Calcular estadísticas
r_mean = np.mean(r_interpolated, axis=0)
r_std = np.std(r_interpolated, axis=0)

print(f"Rango de R promedio: [{r_mean.min():.4f}, {r_mean.max():.4f}]")

# Crear gráfica
print("\nGenerando gráfica...")
fig, ax = plt.subplots(figsize=(10, 7))

# Curva promedio
ax.plot(c_grid, r_mean, 'b-', linewidth=2.5, label='R promedio')

# Banda de desviación estándar
ax.fill_between(c_grid, r_mean - r_std, r_mean + r_std, 
                alpha=0.2, color='blue', label='±1 std')

# Configuración de ejes y etiquetas
ax.set_xlabel('Acoplamiento (C)', fontsize=14, fontweight='bold')
ax.set_ylabel('Parámetro de orden final R(T)', fontsize=14, fontweight='bold')
ax.set_title('Transición de fase de sincronización', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_xlim([C_MIN_PLOT, C_MAX_PLOT])
ax.set_ylim([0, 0.65])

# Añadir texto con estadísticas
textstr = f'N = {len(all_c_values):,} imágenes\nRango: [{C_MIN_PLOT}, {C_MAX_PLOT}]'
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUTPUT, dpi=300, bbox_inches='tight')
print(f"\n✅ Gráfico guardado: {OUTPUT}")

# Calcular y mostrar algunas estadísticas adicionales
idx_critico = np.argmin(np.abs(c_grid - C_CRITICO))
r_en_critico = r_mean[idx_critico]
print(f"\n📊 Estadísticas:")
print(f"   R promedio en C_crítico ({C_CRITICO}): {r_en_critico:.4f}")
print(f"   Desviación estándar en C_crítico: {r_std[idx_critico]:.4f}")

# Encontrar el punto de máxima pendiente (aproximación de C_crítico)
gradient = np.gradient(r_mean, c_grid)
idx_max_gradient = np.argmax(gradient)
c_max_gradient = c_grid[idx_max_gradient]
print(f"   C con máxima pendiente (aprox. C_c): {c_max_gradient:.4f}")
print(f"   Máxima pendiente: {gradient[idx_max_gradient]:.4f}")

plt.show()
