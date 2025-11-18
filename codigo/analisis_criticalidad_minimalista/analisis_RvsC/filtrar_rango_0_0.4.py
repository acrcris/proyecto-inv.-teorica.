#!/usr/bin/env python3
"""
Filtra la base de datos COMBINED para quedarse solo con datos en el rango [0.0, 0.4]
Crea una nueva base de datos con solo los datos filtrados.
"""

import sqlite3
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Rutas
DB_INPUT = Path("/Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/analisis_criticalidad_minimalista/analisis_RvsC/mnist_R_vs_C_COMBINED.db")
DB_OUTPUT = Path("/Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/analisis_criticalidad_minimalista/analisis_RvsC/mnist_R_vs_C_0_0.4.db")

# Rango deseado
C_MIN = 0.0
C_MAX = 0.4

def get_tables(db_path):
    """Obtiene la lista de tablas de clase (clase_0 a clase_9)"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'clase_%'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return sorted(tables)

def get_table_schema(db_path, table_name):
    """Obtiene el esquema SQL de una tabla"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    schema = cursor.fetchone()[0]
    conn.close()
    return schema

def filtrar_tabla(db_input, db_output, tabla):
    """Filtra una tabla para quedarse solo con datos en [C_MIN, C_MAX]"""
    # Conectar a ambas bases de datos
    conn_in = sqlite3.connect(db_input)
    conn_out = sqlite3.connect(db_output)
    cursor_in = conn_in.cursor()
    cursor_out = conn_out.cursor()
    
    # Obtener el esquema y crear la tabla en la DB de salida
    schema = get_table_schema(db_input, tabla)
    cursor_out.execute(schema)
    
    # Leer todos los registros
    cursor_in.execute(f"SELECT * FROM {tabla}")
    rows = cursor_in.fetchall()
    
    # Obtener nombres de columnas
    column_names = [description[0] for description in cursor_in.description]
    
    # Encontrar índices de las columnas relevantes
    id_idx = column_names.index('id') if 'id' in column_names else None
    image_idx_col = 'image_idx' if 'image_idx' in column_names else 'imagen_idx'
    image_idx = column_names.index(image_idx_col)
    c_values_idx = column_names.index('c_values')
    r_values_idx = column_names.index('r_values')
    timestamp_idx = column_names.index('timestamp')
    
    procesadas = 0
    descartadas = 0
    
    print(f"\n📋 Procesando {tabla}...")
    print(f"   Total de registros: {len(rows)}")
    
    for row in tqdm(rows, desc=f"   Filtrando", ncols=100):
        # Deserializar los arrays
        c_values = pickle.loads(row[c_values_idx])
        r_values = pickle.loads(row[r_values_idx])
        
        # Filtrar por rango
        mask = (c_values >= C_MIN) & (c_values <= C_MAX)
        c_filtered = c_values[mask]
        r_filtered = r_values[mask]
        
        # Si no hay datos en el rango, descartar
        if len(c_filtered) == 0:
            descartadas += 1
            continue
        
        # Serializar de nuevo
        c_blob = pickle.dumps(c_filtered)
        r_blob = pickle.dumps(r_filtered)
        
        # Insertar en la nueva DB (sin el id, se autogenera)
        cursor_out.execute(f'''
            INSERT INTO {tabla} ({image_idx_col}, c_values, r_values, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (row[image_idx], c_blob, r_blob, row[timestamp_idx]))
        
        procesadas += 1
    
    conn_out.commit()
    conn_in.close()
    conn_out.close()
    
    print(f"   ✅ Procesadas: {procesadas}")
    print(f"   ⚠️  Descartadas (fuera de rango): {descartadas}")
    
    return procesadas, descartadas

def main():
    print("=" * 70)
    print(f"FILTRADO DE BASE DE DATOS: RANGO [{C_MIN}, {C_MAX}]")
    print("=" * 70)
    print()
    
    # Verificar que existe la DB de entrada
    if not DB_INPUT.exists():
        print(f"❌ Error: No se encuentra {DB_INPUT}")
        return
    
    print(f"📂 Base de datos de entrada: {DB_INPUT.name}")
    print(f"   Tamaño: {DB_INPUT.stat().st_size / (1024*1024):.2f} MB")
    print()
    
    # Eliminar DB de salida si existe
    if DB_OUTPUT.exists():
        print(f"⚠️  Eliminando {DB_OUTPUT.name} existente...")
        DB_OUTPUT.unlink()
    
    print(f"📂 Base de datos de salida: {DB_OUTPUT.name}")
    print(f"📊 Rango de filtrado: C ∈ [{C_MIN}, {C_MAX}]")
    print()
    
    # Obtener tablas
    tables = get_tables(DB_INPUT)
    print(f"📋 Tablas encontradas: {tables}")
    print()
    
    # Procesar cada tabla
    total_procesadas = 0
    total_descartadas = 0
    summary = []
    
    for tabla in tables:
        procesadas, descartadas = filtrar_tabla(DB_INPUT, DB_OUTPUT, tabla)
        total_procesadas += procesadas
        total_descartadas += descartadas
        summary.append((tabla, procesadas, descartadas))
    
    # Resumen final
    print()
    print("=" * 70)
    print("✅ FILTRADO COMPLETADO")
    print("=" * 70)
    print()
    print(f"📊 RESUMEN:")
    print(f"   - Total procesadas: {total_procesadas:,}")
    print(f"   - Total descartadas: {total_descartadas:,}")
    print()
    print("📋 Detalle por clase:")
    for tabla, proc, desc in summary:
        print(f"   - {tabla}: {proc:,} procesadas, {desc:,} descartadas")
    print()
    
    # Tamaño del archivo de salida
    if DB_OUTPUT.exists():
        size_mb = DB_OUTPUT.stat().st_size / (1024 * 1024)
        print(f"💾 Tamaño de {DB_OUTPUT.name}: {size_mb:.2f} MB")
    
    print()
    print("✅ Base de datos filtrada guardada en:")
    print(f"   {DB_OUTPUT}")

if __name__ == "__main__":
    main()
