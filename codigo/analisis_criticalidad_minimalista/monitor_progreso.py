#!/usr/bin/env python3
"""
Script: monitor_progreso.py

Monitorea el progreso del procesamiento en tiempo real.
Muestra estadísticas actualizadas mientras corre run_kuramoto_full_dataset.py

Uso: python monitor_progreso.py
"""

import os
import sys
import torch
import time
from datetime import datetime

CHECKPOINT_DIR = "resultados_kuramoto_full_dataset/checkpoints"

def obtener_ultimo_checkpoint():
    """Encuentra y carga el último checkpoint."""
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('checkpoint_')]
    if not checkpoints:
        return None
    
    checkpoints.sort()
    ultimo = checkpoints[-1]
    checkpoint_path = os.path.join(CHECKPOINT_DIR, ultimo)
    
    try:
        data = torch.load(checkpoint_path)
        return data
    except:
        return None

def mostrar_progreso():
    """Muestra el progreso actual."""
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("="*70)
    print("         MONITOR DE PROGRESO - ANÁLISIS FULL DATASET")
    print("="*70)
    print()
    
    data = obtener_ultimo_checkpoint()
    
    if data is None:
        print("⏳ Esperando inicio del procesamiento...")
        print("   (No se encontraron checkpoints todavía)")
        return
    
    metricas = data['metricas']
    last_idx = data['last_idx']
    timestamp = data.get('timestamp', 'Desconocido')
    
    total_esperado = 10000
    progreso = (last_idx + 1) / total_esperado * 100
    
    print(f"📊 PROGRESO GENERAL")
    print("─" * 70)
    print(f"Última actualización: {timestamp}")
    print(f"Imágenes procesadas:  {last_idx + 1:,} / {total_esperado:,}")
    print(f"Progreso:             {progreso:.2f}%")
    print()
    
    # Barra de progreso
    bar_length = 50
    filled = int(bar_length * progreso / 100)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"[{bar}] {progreso:.1f}%")
    print()
    
    # Estadísticas por clase
    print(f"📈 DISTRIBUCIÓN POR CLASE")
    print("─" * 70)
    for clase in range(10):
        count = sum(1 for m in metricas if m.get('label') == clase and m.get('success', False))
        bar_len = 30
        filled = int(bar_len * count / max(1, len(metricas) / 10))
        mini_bar = '▓' * filled + '░' * (bar_len - filled)
        print(f"  Clase {clase}: [{mini_bar}] {count:4,} imágenes")
    print()
    
    # Métricas recientes
    print(f"🔬 MÉTRICAS PROMEDIO (últimas 100 imágenes)")
    print("─" * 70)
    
    ultimas = [m for m in metricas[-100:] if m.get('success', False)]
    
    if ultimas:
        R_mean = sum(m['R_mean'] for m in ultimas) / len(ultimas)
        DFA_mean = sum(m.get('DFA_alpha', 0) for m in ultimas if 'DFA_alpha' in m) / len(ultimas)
        PSD_mean = sum(m.get('PSD_slope', 0) for m in ultimas if 'PSD_slope' in m) / len(ultimas)
        
        print(f"  R_mean:     {R_mean:.4f}")
        print(f"  DFA_alpha:  {DFA_mean:.4f}")
        print(f"  PSD_slope:  {PSD_mean:.4f}")
    else:
        print("  (Sin datos disponibles)")
    
    print()
    print("─" * 70)
    print("Presiona Ctrl+C para salir del monitor")
    print("El procesamiento continúa en segundo plano...")

def main():
    """Loop principal."""
    try:
        while True:
            mostrar_progreso()
            time.sleep(5)  # Actualizar cada 5 segundos
    except KeyboardInterrupt:
        print("\n\n✓ Monitor detenido. El procesamiento continúa.")
        sys.exit(0)

if __name__ == "__main__":
    main()
