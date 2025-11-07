#!/bin/bash
# iniciar_refactorizado.sh
# Inicia el análisis con la versión REFACTORIZADA

echo "Iniciando análisis REFACTORIZADO..."
echo "DB: resultados_criticalidad_refactorizado.db"
echo "Checkpoint: checkpoint_refactorizado.pt"
echo ""

# Ejecutar en background con nohup
nohup python analizar_con_sqlite_REFACTORIZADO.py > analisis_refactorizado.log 2>&1 &

PID=$!
echo "Proceso iniciado con PID: $PID"
echo ""

sleep 2

# Verificar que se inició correctamente
if ps -p $PID > /dev/null; then
    echo "✅ Proceso corriendo correctamente"
    echo ""
    echo "Comandos útiles:"
    echo "  ./monitor_refactorizado.sh          - Ver estado"
    echo "  tail -f analisis_refactorizado.log  - Ver log en vivo"
    echo "  kill $PID                            - Detener proceso"
else
    echo "❌ Error: El proceso no se inició correctamente"
    echo "Ver log: cat analisis_refactorizado.log"
fi
