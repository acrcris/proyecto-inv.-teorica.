#!/usr/bin/env python3
"""
Script para combinar múltiples bases de datos parciales de C_crítico
Selecciona la tabla con más datos cuando hay duplicados
"""

import sqlite3
import os
from pathlib import Path
from collections import defaultdict

# Configuración
CARPETA_PARCIALES = "resultados_parciales"
DB_SALIDA = "mnist_c_critical_par.db"

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
        # La tabla no existe
        return 0

def get_all_tables(db_path):
    """Obtiene lista de todas las tablas en una DB"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables

def get_table_schema(db_path, table_name):
    """Obtiene el schema de una tabla"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    schema = cursor.fetchall()
    conn.close()
    return schema

def main():
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📦 COMBINAR BASES DE DATOS PARCIALES DE C_CRÍTICO")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    
    # Buscar todas las DBs parciales
    parciales_path = Path(CARPETA_PARCIALES)
    if not parciales_path.exists():
        print(f"❌ Error: No existe la carpeta {CARPETA_PARCIALES}")
        return
    
    db_files = list(parciales_path.glob("*.db"))
    if not db_files:
        print(f"❌ Error: No se encontraron archivos .db en {CARPETA_PARCIALES}")
        return
    
    print(f"🔍 Encontradas {len(db_files)} bases de datos parciales:")
    for db_file in db_files:
        size_mb = db_file.stat().st_size / (1024 * 1024)
        print(f"   • {db_file.name} ({size_mb:.2f} MB)")
    print()
    
    # Analizar todas las DBs y contar registros por tabla
    print("📊 Analizando contenido de cada base de datos...")
    print()
    
    # Diccionario: table_name -> [(db_path, count)]
    tablas_info = defaultdict(list)
    
    for db_file in db_files:
        tables = get_all_tables(str(db_file))
        print(f"   📁 {db_file.name}:")
        
        for table in tables:
            count = get_table_count(str(db_file), table)
            tablas_info[table].append((str(db_file), count))
            print(f"      └─ {table}: {count:,} registros")
        print()
    
    # Seleccionar la mejor fuente para cada tabla
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🎯 SELECCIONANDO MEJORES FUENTES")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    
    mejores_fuentes = {}
    total_registros = 0
    
    for table, sources in sorted(tablas_info.items()):
        # Encontrar la fuente con más registros
        mejor_fuente = max(sources, key=lambda x: x[1])
        mejores_fuentes[table] = mejor_fuente
        total_registros += mejor_fuente[1]
        
        db_name = Path(mejor_fuente[0]).name
        print(f"   {table}:")
        print(f"      ✅ Mejor fuente: {db_name} ({mejor_fuente[1]:,} registros)")
        
        # Mostrar alternativas si existen
        alternativas = [s for s in sources if s != mejor_fuente and s[1] > 0]
        if alternativas:
            for alt_db, alt_count in alternativas:
                alt_name = Path(alt_db).name
                print(f"      ⏭️  Descartando: {alt_name} ({alt_count:,} registros)")
        print()
    
    print(f"📈 Total de registros a combinar: {total_registros:,}")
    print()
    
    # Crear la base de datos combinada
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🔨 CREANDO BASE DE DATOS COMBINADA")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    
    # Eliminar DB de salida si existe
    if os.path.exists(DB_SALIDA):
        print(f"⚠️  Eliminando {DB_SALIDA} existente...")
        os.remove(DB_SALIDA)
    
    # Crear nueva DB
    conn_salida = sqlite3.connect(DB_SALIDA)
    cursor_salida = conn_salida.cursor()
    
    for table, (source_db, count) in mejores_fuentes.items():
        source_name = Path(source_db).name
        print(f"   📋 Copiando {table} desde {source_name}...")
        
        # Conectar a la DB de origen
        conn_origen = sqlite3.connect(source_db)
        
        # Obtener schema
        schema = get_table_schema(source_db, table)
        
        # Crear tabla en la DB de salida
        # Primero obtener el CREATE TABLE statement
        cursor_origen = conn_origen.cursor()
        cursor_origen.execute(
            f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'"
        )
        create_statement = cursor_origen.fetchone()[0]
        cursor_salida.execute(create_statement)
        
        # Copiar datos
        cursor_origen.execute(f"SELECT * FROM {table}")
        rows = cursor_origen.fetchall()
        
        # Obtener nombres de columnas
        column_names = [description[0] for description in cursor_origen.description]
        placeholders = ','.join(['?' for _ in column_names])
        
        cursor_salida.executemany(
            f"INSERT INTO {table} VALUES ({placeholders})",
            rows
        )
        
        conn_origen.close()
        print(f"      ✅ {len(rows):,} registros copiados")
    
    # Commit y cerrar
    conn_salida.commit()
    conn_salida.close()
    
    # Verificar resultado
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("✅ VERIFICACIÓN FINAL")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    
    conn_verificacion = sqlite3.connect(DB_SALIDA)
    cursor_verificacion = conn_verificacion.cursor()
    
    total_verificado = 0
    tablas_por_clase = {}
    
    for table in sorted(mejores_fuentes.keys()):
        cursor_verificacion.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor_verificacion.fetchone()[0]
        total_verificado += count
        
        # Extraer número de clase
        if table.startswith("clase_"):
            clase = int(table.split("_")[1])
            tablas_por_clase[clase] = count
    
    conn_verificacion.close()
    
    # Mostrar resumen por clase
    print("   📊 Registros por clase:")
    for clase in sorted(tablas_por_clase.keys()):
        count = tablas_por_clase[clase]
        print(f"      Clase {clase}: {count:,} imágenes")
    
    print()
    print(f"   📈 Total de registros: {total_verificado:,}")
    
    # Tamaño del archivo
    size_mb = os.path.getsize(DB_SALIDA) / (1024 * 1024)
    print(f"   💾 Tamaño del archivo: {size_mb:.2f} MB")
    print()
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"✨ ¡COMPLETADO! Base de datos combinada guardada en: {DB_SALIDA}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    
    # Comandos útiles
    print("💡 Comandos útiles:")
    print(f"   # Ver estructura")
    print(f"   sqlite3 {DB_SALIDA} '.tables'")
    print()
    print(f"   # Ver estadísticas de una clase")
    print(f"   sqlite3 {DB_SALIDA} 'SELECT COUNT(*), AVG(c_critical), MIN(c_critical), MAX(c_critical) FROM clase_0;'")
    print()

if __name__ == "__main__":
    main()
