#!/usr/bin/env python3
"""
Script temporal para ejecutar el proceso de combinación
"""
import subprocess
import sys
import os
from pathlib import Path

# Cambiar al directorio de resultados parciales
script_dir = Path(__file__).parent
os.chdir(script_dir)

print("════════════════════════════════════════════════════════════════")
print("🔄 INICIANDO PROCESO DE COMBINACIÓN")
print("════════════════════════════════════════════════════════════════")
print()

# Paso 1: Combinar bases de datos
print("📦 Paso 1: Combinando bases de datos parciales...")
print("=" * 70)
result1 = subprocess.run([sys.executable, "combinar_bases_datos.py"], 
                        capture_output=False, text=True)

if result1.returncode == 0:
    print()
    print("✅ Combinación exitosa")
    print()
    
    # Paso 2: Verificar
    print("════════════════════════════════════════════════════════════════")
    print("🔍 Paso 2: Verificando base de datos combinada...")
    print("=" * 70)
    result2 = subprocess.run([sys.executable, "verificar_combinado.py"],
                           capture_output=False, text=True)
    
    if result2.returncode == 0:
        print()
        print("✅ Verificación exitosa")
        print()
        
        # Paso 3: Graficar
        print("════════════════════════════════════════════════════════════════")
        print("📊 Paso 3: Generando visualizaciones...")
        print("=" * 70)
        result3 = subprocess.run([sys.executable, "graficar_distribucion.py"],
                               capture_output=False, text=True)
        
        if result3.returncode == 0:
            print()
            print("════════════════════════════════════════════════════════════════")
            print("✨ PROCESO COMPLETADO EXITOSAMENTE")
            print("════════════════════════════════════════════════════════════════")
            
            # Verificar archivos generados
            db_file = Path("mnist_critical_tot.db")
            if db_file.exists():
                size_mb = db_file.stat().st_size / (1024 * 1024)
                print(f"\n📦 Base de datos generada: {db_file.name}")
                print(f"   💾 Tamaño: {size_mb:.2f} MB")
        else:
            print("\n⚠️  Error al generar gráficas")
    else:
        print("\n⚠️  Error al verificar la base de datos")
else:
    print("\n❌ Error al combinar las bases de datos")
    sys.exit(1)
