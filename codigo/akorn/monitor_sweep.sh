#!/bin/bash

# Script para monitorear el progreso del sweep de C_scale

BASE_DIR="./runs/c_critico_sweep"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         MONITOR DE SWEEP C_SCALE                                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Procesos activos
echo "🔄 PROCESOS ACTIVOS:"
active_count=$(ps aux | grep "train_classification.py c_scale" | grep -v grep | wc -l | xargs)
ps aux | grep "train_classification.py c_scale" | grep -v grep | awk '{print "  PID " $2 ": c_scale_" $0}' | grep -o "c_scale_[0-9.]*" | head -3
echo "  Total: $active_count entrenamientos corriendo"
echo ""

# Progreso de logs
echo "📊 PROGRESO (últimas iteraciones):"
for log in ${BASE_DIR}/logs_c_scale_*.txt; do
    if [ -f "$log" ]; then
        c_val=$(basename "$log" | sed 's/logs_c_scale_//' | sed 's/.txt//')
        last_line=$(tail -1 "$log" 2>/dev/null | grep -o "[0-9]*it" | head -1)
        if [ ! -z "$last_line" ]; then
            echo "  C=$c_val: $last_line"
        fi
    fi
done | tail -10
echo ""

# Checkpoints guardados
echo "💾 CHECKPOINTS GUARDADOS:"
checkpoint_count=$(find ${BASE_DIR} -name "checkpoint_*.pth" 2>/dev/null | wc -l | xargs)
echo "  Total: $checkpoint_count checkpoints"
if [ $checkpoint_count -gt 0 ]; then
    find ${BASE_DIR} -name "checkpoint_*.pth" 2>/dev/null | head -5 | while read cp; do
        echo "  ✓ $(basename $(dirname "$cp"))/$(basename "$cp")"
    done
    if [ $checkpoint_count -gt 5 ]; then
        echo "  ... y $((checkpoint_count - 5)) más"
    fi
fi
echo ""

# Estado del sweep maestro
if [ -f "${BASE_DIR}/sweep_master.log" ]; then
    echo "📋 ÚLTIMO LOTE:"
    tail -15 "${BASE_DIR}/sweep_master.log" | grep -E "LOTE|Experimento|Iniciado|Esperando|completado" | tail -5
fi

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "🕐 $(date '+%H:%M:%S') - Usa 'watch -n 30 bash monitor_sweep.sh' para auto-actualizar"
echo "══════════════════════════════════════════════════════════════════"
