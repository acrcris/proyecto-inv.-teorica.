#!/usr/bin/env python3
"""
Script para verificar el contenido de mnist_critical_tot.db
"""

import sqlite3
import numpy as np

DB_PATH = "mnist_critical_tot.db"

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("📊 RESUMEN DE BASE DE DATOS COMBINADA")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"\n📦 Base de datos: {DB_PATH}\n")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

total_registros = 0
clases_completas = []
clases_parciales = []
clases_vacias = []

# Tamaños esperados por clase (MNIST)
tamaños_mnist = {
    0: 5923, 1: 6742, 2: 5958, 3: 6131, 4: 5842,
    5: 5421, 6: 5918, 7: 6265, 8: 5851, 9: 5949
}

print("Clase | Registros | Esperados | Completitud | Media C_c | Mediana")
print("─" * 70)

for clase in range(10):
    tabla = f"clase_{clase}"
    cursor.execute(f"SELECT COUNT(*) FROM {tabla}")
    count = cursor.fetchone()[0]
    total_registros += count
    
    esperado = tamaños_mnist[clase]
    porcentaje = (count / esperado * 100) if esperado > 0 else 0
    
    # Calcular estadísticas si hay datos
    if count > 0:
        cursor.execute(f"SELECT c_critical FROM {tabla}")
        valores = [row[0] for row in cursor.fetchall()]
        media = np.mean(valores)
        mediana = np.median(valores)
        
        if porcentaje >= 99:
            status = "✅"
            clases_completas.append(clase)
        else:
            status = "⏳"
            clases_parciales.append(clase)
        
        print(f"  {clase}   | {count:>9,} | {esperado:>9,} | {porcentaje:>6.1f}% {status} | {media:>9.4f} | {mediana:>7.4f}")
    else:
        status = "❌"
        clases_vacias.append(clase)
        print(f"  {clase}   | {count:>9,} | {esperado:>9,} | {porcentaje:>6.1f}% {status} |      VACÍA |    ---")

conn.close()

print("─" * 70)
print(f"TOTAL | {total_registros:>9,} |    60,000 | {(total_registros/60000*100):>6.1f}%")

print("\n" + "━" * 70)
print("📈 RESUMEN")
print("━" * 70)
print(f"✅ Clases completas: {len(clases_completas)} → {clases_completas}")
print(f"⏳ Clases parciales: {len(clases_parciales)} → {clases_parciales}")
print(f"❌ Clases vacías: {len(clases_vacias)} → {clases_vacias}")
print(f"\n📊 Total: {total_registros:,} / 60,000 imágenes ({(total_registros/60000*100):.1f}%)")
print("━" * 70)
