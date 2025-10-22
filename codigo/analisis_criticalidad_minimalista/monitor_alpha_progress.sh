#!/bin/bash
# monitor_alpha_progress.sh
# Monitorea el progreso del análisis de alpha crítico

echo "============================================"
echo "  Monitor de Análisis Alpha Crítico"
echo "============================================"
echo ""

# Verificar si el proceso está corriendo
if [ -f analisis_alpha.pid ]; then
    PID=$(cat analisis_alpha.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Proceso activo (PID: $PID)"
    else
        echo "❌ Proceso terminado"
    fi
else
    echo "⚠️  Archivo PID no encontrado"
fi

echo ""
echo "--- Progreso ---"

# Contar imágenes procesadas (del CSV parcial)
if [ -f resultados_alpha/resultados_alpha_por_imagen_partial.csv ]; then
    PROCESSED=$(wc -l < resultados_alpha/resultados_alpha_por_imagen_partial.csv)
    PROCESSED=$((PROCESSED - 1))  # Restar header
    echo "Imágenes procesadas: $PROCESSED / 10002 ($(echo "scale=1; $PROCESSED * 100 / 10002" | bc)%)"
else
    echo "Sin archivo parcial aún"
fi

echo ""
echo "--- Últimas líneas del log ---"
tail -10 analisis_alpha_full.log 2>/dev/null || echo "Log no disponible"

echo ""
echo "--- Uso de CPU/Memoria ---"
if [ ! -z "$PID" ] && ps -p $PID > /dev/null 2>&1; then
    ps -p $PID -o %cpu,%mem,etime,cmd --no-headers
fi

echo ""
echo "Comando para ver log en tiempo real:"
echo "  tail -f analisis_alpha_full.log"
