#!/usr/bin/env python3
"""
Combina mnist_R_vs_C_v2-8.db y mnist_R_vs_C_v2-9.db en mnist_R_vs_C.db
Selecciona la tabla con más registros de cada clase.
"""

import sqlite3
import os
from pathlib import Path

# Rutas de las bases de datos
DB1 = Path("/Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/analisis_criticalidad_minimalista/analisis_RvsC/mnist_R_vs_C_v2-8.db")
DB2 = Path("/Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/analisis_criticalidad_minimalista/analisis_RvsC/mnist_R_vs_C_v2-9.db")
DB_OUT = Path("/Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/analisis_criticalidad_minimalista/analisis_RvsC/mnist_R_vs_C.db")

def get_table_count(db_path, table_name):
    """Obtiene el número de registros en una tabla"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except sqlite3.OperationalError:
        return 0

def get_tables(db_path):
    """Obtiene la lista de tablas de una base de datos"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables

def copy_table(source_db, dest_db, table_name):
    """Copia una tabla completa de una DB a otra"""
    # Conectar a ambas bases de datos
    source_conn = sqlite3.connect(source_db)
    dest_conn = sqlite3.connect(dest_db)
    
    # Obtener el esquema de la tabla
    source_cursor = source_conn.cursor()
    source_cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    create_table_sql = source_cursor.fetchone()[0]
    
    # Crear la tabla en el destino
    dest_cursor = dest_conn.cursor()
    dest_cursor.execute(create_table_sql)
    
    # Copiar los datos
    source_cursor.execute(f"SELECT * FROM {table_name}")
    rows = source_cursor.fetchall()
    
    # Obtener nombres de columnas
    column_names = [description[0] for description in source_cursor.description]
    placeholders = ','.join(['?' for _ in column_names])
    
    dest_cursor.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", rows)
    dest_conn.commit()
    
    source_conn.close()
    dest_conn.close()
    
    return len(rows)

def main():
    print("=" * 70)
    print("COMBINACIÓN DE BASES DE DATOS - MNIST R vs C")
    print("=" * 70)
    print()
    
    # Verificar que existan las bases de datos
    if not DB1.exists():
        print(f"❌ Error: No se encuentra {DB1}")
        return
    if not DB2.exists():
        print(f"❌ Error: No se encuentra {DB2}")
        return
    
    print(f"📂 Base de datos 1: {DB1.name}")
    print(f"📂 Base de datos 2: {DB2.name}")
    print(f"📂 Base de datos de salida: {DB_OUT.name}")
    print()
    
    # Eliminar DB de salida si existe
    if DB_OUT.exists():
        print(f"⚠️  Eliminando {DB_OUT.name} existente...")
        os.remove(DB_OUT)
    
    # Obtener todas las tablas de ambas bases de datos
    tables_db1 = set(get_tables(DB1))
    tables_db2 = set(get_tables(DB2))
    all_tables = tables_db1.union(tables_db2)
    
    print(f"📊 Tablas en DB1: {sorted(tables_db1)}")
    print(f"📊 Tablas en DB2: {sorted(tables_db2)}")
    print(f"📊 Tablas totales a procesar: {sorted(all_tables)}")
    print()
    
    # Procesar cada tabla
    total_records = 0
    summary = []
    
    for table in sorted(all_tables):
        count1 = get_table_count(DB1, table) if table in tables_db1 else 0
        count2 = get_table_count(DB2, table) if table in tables_db2 else 0
        
        print(f"📋 Tabla: {table}")
        print(f"   - DB1 ({DB1.name}): {count1:,} registros")
        print(f"   - DB2 ({DB2.name}): {count2:,} registros")
        
        # Seleccionar la DB con más registros
        if count1 >= count2:
            source_db = DB1
            selected_count = count1
            source_name = "DB1"
        else:
            source_db = DB2
            selected_count = count2
            source_name = "DB2"
        
        if selected_count == 0:
            print(f"   ⚠️  Tabla vacía en ambas DBs, omitiendo...")
            print()
            continue
        
        print(f"   ✅ Seleccionada: {source_name} con {selected_count:,} registros")
        
        # Copiar la tabla
        copied = copy_table(source_db, DB_OUT, table)
        total_records += copied
        summary.append((table, source_name, copied))
        
        print(f"   ✅ Copiados {copied:,} registros")
        print()
    
    # Resumen final
    print("=" * 70)
    print("✅ COMBINACIÓN COMPLETADA")
    print("=" * 70)
    print()
    print(f"📊 RESUMEN:")
    print(f"   - Total de tablas combinadas: {len(summary)}")
    print(f"   - Total de registros: {total_records:,}")
    print()
    print("📋 Detalle por tabla:")
    for table, source, count in summary:
        print(f"   - {table}: {count:,} registros (desde {source})")
    print()
    
    # Tamaño del archivo
    if DB_OUT.exists():
        size_mb = DB_OUT.stat().st_size / (1024 * 1024)
        print(f"💾 Tamaño de {DB_OUT.name}: {size_mb:.2f} MB")
    
    print()
    print("✅ Base de datos combinada guardada en:")
    print(f"   {DB_OUT}")

if __name__ == "__main__":
    main()
