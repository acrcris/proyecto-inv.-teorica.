#!/bin/bash
# Script para monitorear continuamente (alternativa a watch en macOS)

INTERVALO=${1:-60}  # Intervalo en segundos (default: 60)

echo "Monitoreando cada $INTERVALO segundos. Presiona Ctrl+C para detener."
echo ""

while true; do
    clear
    ./monitor_analisis.sh
    echo ""
    echo "⏱️  Próxima actualización en $INTERVALO segundos... (Ctrl+C para salir)"
    sleep $INTERVALO
done
