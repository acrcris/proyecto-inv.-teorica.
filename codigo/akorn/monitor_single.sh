#!/bin/bash
# Monitor simple para el entrenamiento individual

PID=68580
LOG_FILE="runs/c_critico_single/logs_c_scale_0.0062.txt"

echo "�� MONITOR ENTRENAMIENTO C_SCALE=0.0062"
echo "========================================"
date
echo ""

# Estado del proceso
if ps -p $PID > /dev/null 2>&1; then
    echo "✅ Proceso activo:"
    ps -p $PID -o pid,etime,pcpu,rss | tail -1 | awk '{printf "   PID: %s | Tiempo: %s | CPU: %s%% | RAM: %sMB\n", $1, $2, $3, int($4/1024)}'
else
    echo "❌ Proceso no encontrado (PID: $PID)"
fi

echo ""
echo "📊 Progreso reciente:"
tail -50 "$LOG_FILE" | grep -E "Epoch:" | tail -3 | sed 's/^/   /'

echo ""
echo "🔄 Últimas iteraciones:"
tail -5 "$LOG_FILE" | grep "it\]" | tail -1 | sed 's/^/   /'

echo ""
echo "========================================"
