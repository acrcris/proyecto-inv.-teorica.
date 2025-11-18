#!/usr/bin/env python3
"""
Script para combinar dos bases de datos SQLite específicas.
Toma las tablas con más datos de cada una y las une en una base de datos combinada.
"""

import sqlite3
import sys
from pathlib import Path

def contar_registros(db_path: Path, tabla: str) -> int:
    """Cuenta el número de registros en una tabla."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(f'SELECT COUNT(*) FROM {tabla}')
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except sqlite3.OperationalError:
        return 0

def obtener_tablas(db_path: Path) -> list:
    """Obtiene la lista de tablas en una base de datos."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tablas = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tablas

def copiar_tabla(origen_db: Path, destino_db: Path, tabla: str):
    """Copia una tabla completa de una base de datos a otra."""
    # Conectar a ambas bases de datos
    conn_origen = sqlite3.connect(str(origen_db))
    conn_destino = sqlite3.connect(str(destino_db))
    
    cursor_origen = conn_origen.cursor()
    cursor_destino = conn_destino.cursor()
    
    # Saltar tablas del sistema de SQLite
    if tabla == 'sqlite_sequence':
        conn_origen.close()
        conn_destino.close()
        return
    
    # Obtener la estructura de la tabla
    cursor_origen.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{tabla}'")
    result = cursor_origen.fetchone()
    if not result:
        conn_origen.close()
        conn_destino.close()
        return
    create_statement = result[0]
    
    # Crear tabla en destino (eliminar si existe)
    cursor_destino.execute(f'DROP TABLE IF EXISTS {tabla}')
    cursor_destino.execute(create_statement)
    
    # Copiar datos
    cursor_origen.execute(f'SELECT * FROM {tabla}')
    columnas = [desc[0] for desc in cursor_origen.description]
    placeholders = ','.join(['?'] * len(columnas))
    
    batch_size = 1000
    while True:
        rows = cursor_origen.fetchmany(batch_size)
        if not rows:
            break
        cursor_destino.executemany(
            f'INSERT INTO {tabla} VALUES ({placeholders})',
            rows
        )
    
    conn_destino.commit()
    conn_origen.close()
    conn_destino.close()

def main():
    # Configuración de las bases de datos
    db1_path = Path("mnist_R_vs_C_v2-8.db")
    db2_path = Path("mnist_R_vs_C_v2-9.db")
    output_path = Path("mnist_R_vs_C_combined.db")
    
    print("="*70)
    print("COMBINAR BASES DE DATOS R vs C")
    print("="*70)
    print()
    
    print(f"🔍 Analizando bases de datos...")
    print(f"   DB1: {db1_path}")
    print(f"   DB2: {db2_path}")
    print()
    
    # Verificar que existen
    if not db1_path.exists():
        print(f"❌ Error: No existe {db1_path}")
        sys.exit(1)
    
    if not db2_path.exists():
        print(f"❌ Error: No existe {db2_path}")
        sys.exit(1)
    
    # Obtener tablas de ambas bases de datos
    tablas_db1 = obtener_tablas(db1_path)
    tablas_db2 = obtener_tablas(db2_path)
    
    # Unión de todas las tablas
    todas_tablas = set(tablas_db1) | set(tablas_db2)
    
    print(f"📊 Tablas encontradas:")
    print(f"   DB1: {len(tablas_db1)} tablas")
    print(f"   DB2: {len(tablas_db2)} tablas")
    print(f"   Total únicas: {len(todas_tablas)} tablas")
    print()
    
    # Crear base de datos de salida
    if output_path.exists():
        print(f"⚠️  Archivo de salida ya existe, será sobrescrito: {output_path}")
        output_path.unlink()
    
    # Copiar la estructura inicial
    conn_output = sqlite3.connect(str(output_path))
    conn_output.close()
    
    print("🔄 Comparando y seleccionando tablas con más datos...")
    print()
    
    decisiones = []
    total_registros = 0
    
    for tabla in sorted(todas_tablas):
        # Saltar tablas del sistema
        if tabla == 'sqlite_sequence':
            continue
            
        count_db1 = contar_registros(db1_path, tabla)
        count_db2 = contar_registros(db2_path, tabla)
        
        # Decidir de dónde copiar
        if count_db1 == 0 and count_db2 == 0:
            print(f"⚠️  {tabla}: Ambas vacías, omitiendo")
            continue
        elif count_db1 >= count_db2:
            origen = db1_path
            count = count_db1
            origen_name = "v2-8"
        else:
            origen = db2_path
            count = count_db2
            origen_name = "v2-9"
        
        print(f"✅ {tabla}: {count:,} registros (desde {origen_name})")
        print(f"   v2-8: {count_db1:,} | v2-9: {count_db2:,}")
        
        # Copiar tabla
        copiar_tabla(origen, output_path, tabla)
        
        total_registros += count
        decisiones.append({
            'tabla': tabla,
            'origen': origen_name,
            'registros': count,
            'count_db1': count_db1,
            'count_db2': count_db2
        })
    
    print()
    print("="*70)
    print("✅ COMBINACIÓN COMPLETADA")
    print("="*70)
    print(f"📁 Base de datos combinada: {output_path}")
    print(f"📊 Tablas combinadas: {len(decisiones)}")
    print()
    
    # Resumen
    desde_db1 = sum(1 for d in decisiones if d['origen'] == 'v2-8')
    desde_db2 = sum(1 for d in decisiones if d['origen'] == 'v2-9')
    
    print("📈 Resumen:")
    print(f"   Total de registros: {total_registros:,}")
    print(f"   Tablas desde v2-8: {desde_db1}")
    print(f"   Tablas desde v2-9: {desde_db2}")
    
    # Tamaño del archivo
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   💾 Tamaño: {size_mb:.2f} MB")
    print()

if __name__ == '__main__':
    main()
