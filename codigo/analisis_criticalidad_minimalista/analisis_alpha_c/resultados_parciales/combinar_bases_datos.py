#!/usr/bin/env python3
"""
Script para combinar múltiples bases de datos SQLite de resultados parciales.
Cuando hay duplicados (misma clase), se elige la tabla con más registros.
"""

import sqlite3
import os
import glob
from collections import defaultdict

def obtener_info_tablas(db_path):
    """
    Obtiene información de todas las tablas clase_X en una base de datos.
    
    Returns:
        dict: {clase_numero: num_registros}
    """
    info = {}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Obtener todas las tablas
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'clase_%'")
        tablas = cursor.fetchall()
        
        for (tabla,) in tablas:
            # Extraer número de clase
            try:
                clase = int(tabla.replace('clase_', ''))
                # Contar registros
                cursor.execute(f"SELECT COUNT(*) FROM {tabla}")
                count = cursor.fetchone()[0]
                info[clase] = count
            except ValueError:
                continue
        
        conn.close()
    except sqlite3.Error as e:
        print(f"⚠️  Error al leer {db_path}: {e}")
    
    return info

def copiar_tabla(src_db, dest_db, clase):
    """
    Copia una tabla clase_X de una base de datos a otra.
    """
    try:
        # Conectar a ambas bases de datos
        src_conn = sqlite3.connect(src_db)
        dest_conn = sqlite3.connect(dest_db)
        
        src_cursor = src_conn.cursor()
        dest_cursor = dest_conn.cursor()
        
        tabla = f"clase_{clase}"
        
        # Obtener estructura de la tabla
        src_cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{tabla}'")
        create_stmt = src_cursor.fetchone()
        
        if create_stmt:
            # Crear tabla en destino (si no existe)
            try:
                dest_cursor.execute(create_stmt[0])
            except sqlite3.OperationalError:
                # La tabla ya existe, eliminarla primero
                dest_cursor.execute(f"DROP TABLE IF EXISTS {tabla}")
                dest_cursor.execute(create_stmt[0])
            
            # Copiar datos
            src_cursor.execute(f"SELECT * FROM {tabla}")
            rows = src_cursor.fetchall()
            
            if rows:
                # Obtener nombres de columnas
                src_cursor.execute(f"PRAGMA table_info({tabla})")
                columns = [row[1] for row in src_cursor.fetchall()]
                placeholders = ','.join(['?' for _ in columns])
                
                # Insertar datos
                dest_cursor.executemany(
                    f"INSERT OR REPLACE INTO {tabla} VALUES ({placeholders})",
                    rows
                )
                dest_conn.commit()
                
                return len(rows)
        
        src_conn.close()
        dest_conn.close()
        return 0
        
    except sqlite3.Error as e:
        print(f"❌ Error al copiar tabla clase_{clase}: {e}")
        return 0

def main():
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🔄 COMBINADOR DE BASES DE DATOS PARCIALES")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    
    # Buscar todas las bases de datos .db en la carpeta actual
    db_files = sorted(glob.glob("*.db"))
    
    if not db_files:
        print("❌ No se encontraron archivos .db en la carpeta actual")
        return
    
    print(f"📂 Bases de datos encontradas: {len(db_files)}")
    for db in db_files:
        size_mb = os.path.getsize(db) / (1024 * 1024)
        print(f"   • {db} ({size_mb:.2f} MB)")
    print()
    
    # Analizar cada base de datos
    print("🔍 Analizando contenido de cada base de datos...")
    print()
    
    # Diccionario para guardar: {clase: [(db_path, num_registros), ...]}
    clases_info = defaultdict(list)
    
    for db_path in db_files:
        info = obtener_info_tablas(db_path)
        print(f"📊 {db_path}:")
        
        if not info:
            print(f"   ⚠️  Sin tablas válidas")
        else:
            for clase, count in sorted(info.items()):
                print(f"   • Clase {clase}: {count:,} registros")
                clases_info[clase].append((db_path, count))
        print()
    
    # Decidir qué tabla usar para cada clase
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📋 SELECCIÓN DE MEJORES TABLAS")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    
    seleccion = {}
    for clase in sorted(clases_info.keys()):
        candidatos = clases_info[clase]
        # Elegir la DB con más registros
        mejor_db, max_count = max(candidatos, key=lambda x: x[1])
        seleccion[clase] = (mejor_db, max_count)
        
        if len(candidatos) > 1:
            print(f"✅ Clase {clase}: {max_count:,} registros de {mejor_db}")
            print(f"   Descartando:")
            for db, count in candidatos:
                if db != mejor_db:
                    print(f"   • {db} ({count:,} registros)")
        else:
            print(f"✅ Clase {clase}: {max_count:,} registros de {mejor_db} (única fuente)")
    
    print()
    
    # Crear base de datos combinada
    output_db = "mnist_critical_tot.db"
    
    if os.path.exists(output_db):
        print(f"⚠️  {output_db} ya existe. Eliminando...")
        os.remove(output_db)
    
    print(f"🔨 Creando base de datos combinada: {output_db}")
    print()
    
    # Copiar tablas seleccionadas
    total_registros = 0
    for clase in sorted(seleccion.keys()):
        src_db, count = seleccion[clase]
        print(f"📥 Copiando Clase {clase} desde {src_db}...", end=" ")
        
        registros_copiados = copiar_tabla(src_db, output_db, clase)
        
        if registros_copiados > 0:
            print(f"✅ {registros_copiados:,} registros")
            total_registros += registros_copiados
        else:
            print(f"❌ Error")
    
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("✨ PROCESO COMPLETADO")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    
    if os.path.exists(output_db):
        size_mb = os.path.getsize(output_db) / (1024 * 1024)
        print(f"📦 Base de datos combinada: {output_db}")
        print(f"   • Tamaño: {size_mb:.2f} MB")
        print(f"   • Clases: {len(seleccion)}")
        print(f"   • Total registros: {total_registros:,}")
        print()
        
        # Verificar contenido final
        print("🔍 Verificando contenido final:")
        conn = sqlite3.connect(output_db)
        cursor = conn.cursor()
        for clase in sorted(seleccion.keys()):
            cursor.execute(f"SELECT COUNT(*) FROM clase_{clase}")
            count = cursor.fetchone()[0]
            print(f"   ✅ Clase {clase}: {count:,} registros")
        conn.close()
        print()
        print("✨ Todo listo! Puedes usar mnist_critical_tot.db ahora.")
    else:
        print("❌ Error: No se pudo crear la base de datos combinada")

if __name__ == "__main__":
    main()
