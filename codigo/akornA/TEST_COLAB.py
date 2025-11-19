#!/usr/bin/env python3
"""
Script de prueba para diagnosticar problemas en Colab
Ejecuta: python TEST_COLAB.py
"""

import sys
import os

print("="*60)
print("🔍 DIAGNÓSTICO COLAB")
print("="*60)

# 1. Verificar directorio actual
print(f"\n📍 Directorio actual: {os.getcwd()}")

# 2. Verificar Python path
print(f"\n🐍 Python executable: {sys.executable}")
print(f"📚 sys.path:")
for p in sys.path[:5]:
    print(f"   - {p}")

# 3. Verificar archivos críticos
print(f"\n📁 Archivos críticos:")
files_to_check = ['requirements.txt', 'train_classification.py', 'source/__init__.py']
for f in files_to_check:
    exists = "✅" if os.path.exists(f) else "❌"
    print(f"   {exists} {f}")

# 4. Intentar importar source
print(f"\n📦 Intentando importar módulo source...")
try:
    import source
    print(f"   ✅ source importado desde: {source.__file__}")
except ImportError as e:
    print(f"   ❌ Error al importar source: {e}")

# 5. Verificar train_classification.py
print(f"\n🔬 Verificando train_classification.py...")
try:
    with open('train_classification.py', 'r') as f:
        lines = f.readlines()
        print(f"   ✅ Archivo tiene {len(lines)} líneas")
        # Buscar línea de c_scale
        for i, line in enumerate(lines[:50], 1):
            if 'c_scale' in line.lower():
                print(f"   Línea {i}: {line.strip()}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*60)
print("✅ Diagnóstico completado")
print("="*60)
