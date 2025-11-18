#!/bin/bash

# ============================================================================
# Script para regenerar gráficas automáticamente cada cierto tiempo
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Activar entorno virtual
source /Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/.venv/bin/activate

# Intervalo en segundos (default: 5 minutos)
INTERVAL=${1:-300}

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║        REGENERACIÓN AUTOMÁTICA DE GRÁFICAS DE DISTRIBUCIÓN           ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Intervalo de actualización: ${INTERVAL} segundos ($((INTERVAL/60)) minutos)${NC}"
echo -e "${YELLOW}Presiona Ctrl+C para detener${NC}"
echo ""

contador=1

while true; do
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}🔄 Actualización #${contador} - ${TIMESTAMP}${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Ejecutar script de generación de gráficas
    python graficar_distribucion_actual.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ Gráficas actualizadas correctamente${NC}"
    else
        echo ""
        echo -e "${YELLOW}⚠️  Hubo un problema al actualizar las gráficas${NC}"
    fi
    
    # Mostrar info de las gráficas
    if [ -f "resultados_c_critical/distribucion_c_critico_resumen.png" ]; then
        SIZE1=$(ls -lh resultados_c_critical/distribucion_c_critico_resumen.png | awk '{print $5}')
        TIME1=$(ls -l resultados_c_critical/distribucion_c_critico_resumen.png | awk '{print $6, $7, $8}')
        echo -e "   📊 distribucion_c_critico_resumen.png: ${SIZE1} (${TIME1})"
    fi
    
    if [ -f "resultados_c_critical/distribucion_c_critico_por_clase.png" ]; then
        SIZE2=$(ls -lh resultados_c_critical/distribucion_c_critico_por_clase.png | awk '{print $5}')
        TIME2=$(ls -l resultados_c_critical/distribucion_c_critico_por_clase.png | awk '{print $6, $7, $8}')
        echo -e "   📊 distribucion_c_critico_por_clase.png: ${SIZE2} (${TIME2})"
    fi
    
    echo ""
    echo -e "${YELLOW}⏳ Esperando ${INTERVAL} segundos hasta la próxima actualización...${NC}"
    
    contador=$((contador + 1))
    sleep $INTERVAL
done

