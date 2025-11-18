#!/bin/bash
# Script para ejecutar el procesamiento de C_crítico en background
# Procesa las clases en orden inverso: 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activar entorno virtual
source /Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/.venv/bin/activate

# Obtener timestamp para el log
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="procesamiento_${TIMESTAMP}.log"

echo "========================================"
echo "Iniciando procesamiento de C_crítico"
echo "========================================"
echo "Clases: 9, 8, 7, 6, 5, 4, 3, 2, 1, 0"
echo "Dispositivo: auto (MPS si disponible)"
echo "Log: $LOG_FILE"
echo "PID será guardado en: procesamiento.pid"
echo ""
echo "Para monitorear el progreso:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Para detener el proceso:"
echo "  kill \$(cat procesamiento.pid)"
echo "========================================"

# Ejecutar con caffeinate para evitar suspensión
# -d: prevenir suspensión del display
# -i: prevenir suspensión idle
# -m: prevenir suspensión del disco
# -s: prevenir suspensión del sistema
nohup caffeinate -dims python calcular_c_critico_local.py \
    --clases 9 8 7 6 5 4 3 2 1 0 \
    --device auto \
    > "$LOG_FILE" 2>&1 &

# Guardar PID
echo $! > procesamiento.pid

echo ""
echo "✅ Proceso iniciado en background"
echo "   PID: $(cat procesamiento.pid)"
echo "   Log: $LOG_FILE"
echo ""
echo "Comandos útiles:"
echo "  # Ver progreso en tiempo real"
echo "  tail -f $LOG_FILE"
echo ""
echo "  # Ver las últimas 50 líneas"
echo "  tail -50 $LOG_FILE"
echo ""
echo "  # Verificar que está corriendo"
echo "  ps aux | grep \$(cat procesamiento.pid)"
echo ""
echo "  # Detener el proceso"
echo "  kill \$(cat procesamiento.pid)"
echo ""
