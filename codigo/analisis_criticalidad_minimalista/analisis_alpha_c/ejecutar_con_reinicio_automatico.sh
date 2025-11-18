#!/bin/bash

# ============================================================================
# Script de ejecución con REINICIO AUTOMÁTICO
# ============================================================================
# Este script detecta cuando el proceso muere por semaphore leak y lo reinicia
# automáticamente hasta que todas las clases estén completadas.
# ============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activar entorno virtual
source /Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/.venv/bin/activate

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  PROCESAMIENTO C_CRÍTICO CON AUTO-REINICIO AUTOMÁTICO     ║${NC}"
echo -e "${GREEN}║  🌟 MODO: TODAS LAS CLASES (0-9)                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Parámetros - MODIFICADO PARA TODAS LAS CLASES
CLASES="0 1 2 3 4 5 6 7 8 9"  # Todas las clases
MAX_INTENTOS=100  # Máximo 100 reinicios (debería ser suficiente)
DB_PATH="resultados_c_critical/mnist_critical_tot.db"

# Función para verificar si una clase está completa
check_clase_completa() {
    clase=$1
    if [ ! -f "$DB_PATH" ]; then
        echo 0
        return
    fi
    
    # Contar imágenes procesadas
    count=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM clase_${clase};" 2>/dev/null || echo "0")
    
    # Total esperado por clase (aproximado, puedes ajustar)
    case $clase in
        0) total=5923 ;;
        1) total=6742 ;;
        2) total=5958 ;;
        3) total=6131 ;;
        4) total=5842 ;;
        5) total=5421 ;;
        6) total=5918 ;;
        7) total=6265 ;;
        8) total=5851 ;;
        9) total=5949 ;;
        *) total=6000 ;;
    esac
    
    if [ "$count" -ge "$total" ]; then
        echo 1
    else
        echo 0
    fi
}

# Función para verificar si todo está completo
check_todo_completo() {
    for clase in $CLASES; do
        completo=$(check_clase_completa $clase)
        if [ "$completo" -eq 0 ]; then
            return 1  # No está completo
        fi
    done
    return 0  # Todo completo
}

# Contador de reinicios
intentos=0

while [ $intentos -lt $MAX_INTENTOS ]; do
    intentos=$((intentos + 1))
    
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}   Intento #${intentos} de ${MAX_INTENTOS}${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # Verificar si ya está todo completo
    if check_todo_completo; then
        echo -e "${GREEN}✨ ¡TODO COMPLETADO! Todas las 60,000 imágenes procesadas.${NC}"
        exit 0
    fi
    
    # Crear timestamp para log
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="procesamiento_${TIMESTAMP}.log"
    
    echo -e "📝 Log: ${LOG_FILE}"
    echo -e "🚀 Iniciando procesamiento..."
    echo -e "🔋 Activando prevención de suspensión (caffeinate)..."
    echo ""
    
    # Ejecutar con caffeinate MEJORADO:
    # -d: Previene que el disco se duerma
    # -i: Previene que el sistema idle se duerma
    # -m: Previene que el disco se duerma cuando está cerrado
    # -s: Previene que el sistema se duerma (incluso si se cierra la tapa)
    # -u: Declara que user está activo (previene display sleep también)
    # Y capturar el código de salida
    caffeinate -dimsu python calcular_c_critico_local.py \
        --all \
        --device auto \
        > "$LOG_FILE" 2>&1
    
    EXIT_CODE=$?
    
    echo ""
    echo -e "⚠️  Proceso terminó con código: ${EXIT_CODE}"
    
    # Si el código es 0, terminó normalmente (éxito)
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✅ Proceso completado exitosamente.${NC}"
        break
    fi
    
    # Si fue señal de terminación (killed), reiniciar
    echo -e "${YELLOW}🔄 Reiniciando proceso en 5 segundos...${NC}"
    echo -e "${YELLOW}   (El auto-restart continuará desde el último checkpoint)${NC}"
    sleep 5
done

if [ $intentos -ge $MAX_INTENTOS ]; then
    echo ""
    echo -e "${RED}❌ Se alcanzó el máximo de intentos (${MAX_INTENTOS}).${NC}"
    echo -e "${RED}   Revisa los logs para más información.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           PROCESAMIENTO FINALIZADO CON ÉXITO               ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "📊 Para ver resultados:"
echo "   ./ver_estado.sh"
echo ""
