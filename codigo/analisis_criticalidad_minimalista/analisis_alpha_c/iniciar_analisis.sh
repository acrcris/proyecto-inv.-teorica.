#!/bin/bash
# Script para iniciar el análisis de criticalidad

cd "$(dirname "$0")"

echo "======================================================================="
echo "  INICIANDO ANÁLISIS DE CRITICALIDAD - MNIST (60,000 imágenes)"
echo "======================================================================="
echo ""

# Verificar si ya está corriendo
if pgrep -f "analizar_con_sqlite.py" > /dev/null; then
    PID=$(pgrep -f "analizar_con_sqlite.py")
    echo "⚠️  El proceso ya está corriendo (PID: $PID)"
    echo ""
    read -p "¿Deseas detenerlo y reiniciar? (s/N): " respuesta
    if [[ "$respuesta" =~ ^[Ss]$ ]]; then
        echo "Deteniendo proceso $PID..."
        kill $PID
        sleep 2
    else
        echo "Abortando. El proceso existente continúa corriendo."
        exit 0
    fi
fi

# Mostrar estado actual
if [ -f "resultados_criticalidad.db" ]; then
    echo "📊 Estado actual de la base de datos:"
    sqlite3 resultados_criticalidad.db << EOF
SELECT 
    '   Procesadas: ' || COUNT(*) || '/60,000 (' || ROUND(COUNT(*)*100.0/60000,2) || '%)' 
FROM resultados;
EOF
    echo ""
fi

# Iniciar proceso
echo "🚀 Iniciando análisis en background..."
nohup python analizar_con_sqlite.py > analisis_sqlite.log 2>&1 &
NEW_PID=$!

sleep 3

# Verificar que inició correctamente
if ps -p $NEW_PID > /dev/null 2>&1; then
    echo "✅ Proceso iniciado correctamente (PID: $NEW_PID)"
    echo ""
    echo "📝 Comandos útiles:"
    echo "   ./monitor_analisis.sh       # Ver estado actual"
    echo "   ./monitor_continuo.sh       # Monitoreo continuo"
    echo "   tail -f analisis_sqlite.log # Ver log en tiempo real"
    echo "   kill $NEW_PID               # Detener el proceso"
else
    echo "❌ Error al iniciar el proceso. Revisa analisis_sqlite.log"
    tail -20 analisis_sqlite.log
    exit 1
fi

echo ""
echo "======================================================================="
