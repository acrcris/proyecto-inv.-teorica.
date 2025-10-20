#!/bin/bash
# Script para monitorear el progreso del procesamiento

cd "$(dirname "$0")"

echo "========================================"
echo "📊 MONITOREO DE PROCESAMIENTO CORREGIDO"
echo "========================================"
echo ""

# Verificar si el proceso está corriendo
if [ -f "kuramoto_full.pid" ]; then
    PID=$(cat kuramoto_full.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Proceso activo (PID: $PID)"
        
        # Mostrar uso de CPU y memoria
        echo ""
        echo "📈 Recursos del proceso:"
        ps -p $PID -o %cpu,%mem,etime,cmd | tail -1
        echo ""
    else
        echo "❌ Proceso no está corriendo (PID en archivo: $PID)"
        echo ""
    fi
else
    echo "❌ No hay archivo PID"
    echo ""
fi

# Mostrar checkpoints guardados
echo "💾 Checkpoints guardados:"
if [ -d "resultados_kuramoto_full_dataset_CORRECTED/checkpoints" ]; then
    CHECKPOINTS=$(ls resultados_kuramoto_full_dataset_CORRECTED/checkpoints/*.pt 2>/dev/null | wc -l)
    echo "   Total: $CHECKPOINTS archivos"
    if [ $CHECKPOINTS -gt 0 ]; then
        ULTIMO=$(ls -t resultados_kuramoto_full_dataset_CORRECTED/checkpoints/*.pt 2>/dev/null | head -1)
        echo "   Último: $(basename $ULTIMO)"
        IMAGENES=$((CHECKPOINTS * 100))
        PROGRESO=$((IMAGENES * 100 / 10000))
        echo "   Progreso: $IMAGENES / 10,000 imágenes ($PROGRESO%)"
    fi
else
    echo "   No hay checkpoints aún"
fi
echo ""

# Mostrar últimas líneas del log
echo "📝 Últimas líneas del log:"
echo "----------------------------------------"
LOG_FILE=$(ls -t resultados_kuramoto_full_dataset_CORRECTED/run_*.log 2>/dev/null | head -1)
if [ ! -f "$LOG_FILE" ]; then
    LOG_FILE=$(ls -t resultados_kuramoto_full_dataset/run_*.log 2>/dev/null | head -1)
fi
if [ -f "$LOG_FILE" ]; then
    tail -15 "$LOG_FILE"
else
    echo "No hay archivo de log"
fi
echo "----------------------------------------"
echo ""

echo "💡 Comandos útiles:"
echo "   ./ver_progreso.sh        - Ver este resumen"
echo "   tail -f \$LOG_FILE       - Seguir log en tiempo real"
echo "   kill \$(cat kuramoto_full.pid) - Detener proceso"
echo ""
