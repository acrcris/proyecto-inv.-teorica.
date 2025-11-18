#!/bin/bash
# Script para ver el progreso del cálculo R vs C en tiempo real

echo "================================================"
echo "PROGRESO DEL CÁLCULO R vs C - Clase 9"
echo "================================================"
echo ""

# Verificar si el proceso está corriendo
if pgrep -f "calcular_r_vs_c.py" > /dev/null; then
    echo "✅ Proceso ACTIVO (PID: $(pgrep -f 'calcular_r_vs_c.py'))"
else
    echo "⚠️  Proceso NO está corriendo"
fi

echo ""
echo "📊 ÚLTIMAS LÍNEAS DEL LOG:"
echo "================================================"
tail -n 30 calcular_r_vs_c_clase9.log

echo ""
echo "================================================"
echo "Para ver el log completo: tail -f calcular_r_vs_c_clase9.log"
echo "Para detener el proceso: pkill -f calcular_r_vs_c.py"
echo "================================================"
