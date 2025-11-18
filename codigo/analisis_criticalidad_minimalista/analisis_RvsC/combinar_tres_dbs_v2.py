#!/usr/bin/env python3
"""
Combina tres bases de datos en una sola: mnist_R_vs_C_COMBINED.db
Selecciona la tabla con más registros de cada clase.
"""

import sqlite3
import os
from pathlib import Path

# Rutas de las bases de datos
DB1 = Path("/Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/analisis_criticalidad_minimalista/analisis_RvsC/mnist_R_vs_C_v2-10.db")
DB2 = Path("/Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/analisis_criticalidad_minimalista/analisis_RvsC/mnist_R_vs_C_v2-11.db")
DB3 = Path("/Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/analisis_criticalidad_minimalista/analisis_RvsC/mnist_R_vs_C_FINAL.db")
DB_OUT = Path("/Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/analisis_criticalidad_minimalista/analisis_RvsC/mnist_R_vs_C_COMBINED.db")

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
    print("COMBINACIÓN DE 3 BASES DE DATOS - MNIST R vs C")
    print("=" * 70)
    print()
    
    # Verificar que existan las bases de datos
    dbs = [DB1, DB2, DB3]
    db_names = ["v2-10", "v2-11", "FINAL"]
    
    for db, name in zip(dbs, db_names):
        if not db.exists():
            print(f"❌ Error: No se encuentra {db}")
            return
        print(f"✅ {name}: {db.stat().st_size / (1024*1024):.2f} MB")
    
    print()
    
    # Eliminar DB de salida si existe
    if DB_OUT.exists():
        print(f"⚠️  Eliminando {DB_OUT.name} existente...")
        os.remove(DB_OUT)
    
    print(f"📂 Base de datos de salida: {DB_OUT.name}")
    print()
    
    # Obtener todas las tablas de las tres bases de datos
    tables_db1 = set(get_tables(DB1))
    tables_db2 = set(get_tables(DB2))
    tables_db3 = set(get_tables(DB3))
    all_tables = tables_db1.union(tables_db2).union(tables_db3)
    
    print(f"📊 Tablas encontradas:")
    print(f"   - DB1 (v2-10): {sorted(tables_db1)}")
    print(f"   - DB2 (v2-11): {sorted(tables_db2)}")
    print(f"   - DB3 (FINAL): {sorted(tables_db3)}")
    print(f"   - Tablas totales a procesar: {sorted(all_tables)}")
    print()
    
    # Procesar cada tabla
    total_records = 0
    summary = []
    
    for table in sorted(all_tables):
        count1 = get_table_count(DB1, table) if table in tables_db1 else 0
        count2 = get_table_count(DB2, table) if table in tables_db2 else 0
        count3 = get_table_count(DB3, table) if table in tables_db3 else 0
        
        print(f"📋 Tabla: {table}")
        print(f"   - DB1 (v2-10): {count1:,} registros")
        print(f"   - DB2 (v2-11): {count2:,} registros")
        print(f"   - DB3 (FINAL): {count3:,} registros")
        
        # Seleccionar la DB con más registros
        counts = [(count1, DB1, "v2-10"), (count2, DB2, "v2-11"), (count3, DB3, "FINAL")]
        selected_count, source_db, source_name = max(counts, key=lambda x: x[0])
        
        if selected_count == 0:
            print(f"   ⚠️  Tabla vacía en todas las DBs, omitiendo...")
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
