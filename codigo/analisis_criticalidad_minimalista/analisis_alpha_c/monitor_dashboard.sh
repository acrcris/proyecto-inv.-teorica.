#!/bin/bash

# ============================================================================
# Dashboard de Monitoreo en Tiempo Real
# ============================================================================
# Este script muestra un dashboard que se actualiza automáticamente
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Función para limpiar pantalla
clear_screen() {
    clear
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║            DASHBOARD DE MONITOREO - C_CRÍTICO MNIST                 ║"
    echo "║                  Actualización automática cada 10s                   ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Configuración
DB_PATH="resultados_c_critical/mnist_critical_tot.db"
INTERVAL=10  # segundos entre actualizaciones

echo ""
echo -e "${YELLOW}Iniciando dashboard de monitoreo...${NC}"
echo -e "${YELLOW}Presiona Ctrl+C para salir${NC}"
sleep 2

# Loop principal
while true; do
    clear_screen
    
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "${BLUE}📅 Última actualización: ${TIMESTAMP}${NC}"
    echo ""
    
    # ========================================================================
    # 1. ESTADO DEL PROCESO
    # ========================================================================
    echo -e "${MAGENTA}┌─ 🔍 ESTADO DEL PROCESO ────────────────────────────────────────────┐${NC}"
    
    if [ -f procesamiento.pid ]; then
        PID=$(cat procesamiento.pid)
        if ps -p $PID > /dev/null 2>&1; then
            ELAPSED=$(ps -p $PID -o etime= | xargs)
            CPU_MEM=$(ps -p $PID -o %cpu,%mem | tail -1)
            echo -e "${GREEN}│ ✅ Proceso ACTIVO${NC}"
            echo -e "│ PID: $PID | Tiempo: $ELAPSED | CPU/Mem: $CPU_MEM"
            
            # Verificar caffeinate
            if pgrep -f "caffeinate.*calcular_c_critico" > /dev/null 2>&1; then
                echo -e "${GREEN}│ ☕ Prevención de suspensión: ACTIVA${NC}"
            else
                echo -e "${YELLOW}│ ⚠️  Prevención de suspensión: INACTIVA${NC}"
            fi
        else
            echo -e "${RED}│ ❌ Proceso NO está corriendo (PID: $PID terminado)${NC}"
        fi
    else
        echo -e "${RED}│ ⚠️  No se encuentra procesamiento.pid${NC}"
    fi
    
    echo -e "${MAGENTA}└────────────────────────────────────────────────────────────────────┘${NC}"
    echo ""
    
    # ========================================================================
    # 2. PROGRESO POR CLASE
    # ========================================================================
    echo -e "${MAGENTA}┌─ 📊 PROGRESO POR CLASE ────────────────────────────────────────────┐${NC}"
    
    if [ -f "$DB_PATH" ]; then
        declare -A TOTALES
        TOTALES[0]=5923
        TOTALES[1]=6742
        TOTALES[2]=5958
        TOTALES[3]=6131
        TOTALES[4]=5842
        TOTALES[5]=5421
        TOTALES[6]=5918
        TOTALES[7]=6265
        TOTALES[8]=5851
        TOTALES[9]=5949
        
        TOTAL_PROCESADAS=0
        
        for i in {0..9}; do
            COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM clase_$i;" 2>/dev/null || echo "0")
            TOTAL=${TOTALES[$i]}
            TOTAL_PROCESADAS=$((TOTAL_PROCESADAS + COUNT))
            
            PERCENT=$(awk "BEGIN {printf \"%.1f\", ($COUNT/$TOTAL)*100}")
            
            # Barra de progreso
            BARS=$(awk "BEGIN {printf \"%.0f\", ($COUNT/$TOTAL)*10}")
            BAR=""
            for ((j=0; j<$BARS; j++)); do BAR="${BAR}█"; done
            for ((j=$BARS; j<10; j++)); do BAR="${BAR}░"; done
            
            if [ "$COUNT" -eq "$TOTAL" ]; then
                COLOR=$GREEN
                STATUS="✅"
            elif [ "$COUNT" -gt 0 ]; then
                COLOR=$YELLOW
                STATUS="⏳"
            else
                COLOR=$NC
                STATUS="⭕"
            fi
            
            printf "│ %s Clase %d: [%s] %5d/%d (%5.1f%%)${NC}\n" "$STATUS" "$i" "$BAR" "$COUNT" "$TOTAL" "$PERCENT"
        done
        
        # Total
        PERCENT_TOTAL=$(awk "BEGIN {printf \"%.2f\", ($TOTAL_PROCESADAS/60000)*100}")
        echo -e "${MAGENTA}├────────────────────────────────────────────────────────────────────┤${NC}"
        printf "│ ${BLUE}📈 TOTAL: %d / 60000 (%.2f%%)${NC}\n" "$TOTAL_PROCESADAS" "$PERCENT_TOTAL"
        
    else
        echo -e "│ ${RED}⚠️  Base de datos no encontrada${NC}"
    fi
    
    echo -e "${MAGENTA}└────────────────────────────────────────────────────────────────────┘${NC}"
    echo ""
    
    # ========================================================================
    # 3. ESTIMACIÓN DE TIEMPO
    # ========================================================================
    echo -e "${MAGENTA}┌─ ⏱️  TIEMPO Y VELOCIDAD ───────────────────────────────────────────┐${NC}"
    
    LATEST_LOG=$(ls -t procesamiento_*.log 2>/dev/null | head -1)
    
    if [ -n "$LATEST_LOG" ] && [ -f "$DB_PATH" ]; then
        SPEED=$(grep -oE '[0-9]+\.[0-9]+s/it' "$LATEST_LOG" | tail -1 | grep -oE '[0-9]+\.[0-9]+')
        
        if [ -n "$SPEED" ]; then
            REMAINING=$((60000 - TOTAL_PROCESADAS))
            HOURS=$(awk "BEGIN {printf \"%.1f\", ($REMAINING * $SPEED) / 3600}")
            MINUTES=$(awk "BEGIN {printf \"%.0f\", (($REMAINING * $SPEED) / 60) % 60}")
            
            echo -e "│ Velocidad: ${GREEN}~${SPEED}s/imagen${NC}"
            echo -e "│ Restantes: ${YELLOW}${REMAINING} imágenes${NC}"
            echo -e "│ Tiempo estimado: ${CYAN}~${HOURS}h ${MINUTES}min${NC}"
            
            # ETA
            TOTAL_MINUTES=$(awk "BEGIN {printf \"%.0f\", ($REMAINING * $SPEED) / 60}")
            if [ "$TOTAL_MINUTES" -gt 0 ]; then
                END_TIMESTAMP=$(date -v+${TOTAL_MINUTES}M "+%H:%M:%S" 2>/dev/null)
                if [ -n "$END_TIMESTAMP" ]; then
                    echo -e "│ ETA: ${GREEN}${END_TIMESTAMP}${NC}"
                fi
            fi
        else
            echo -e "│ ${YELLOW}⏳ Calculando velocidad...${NC}"
        fi
    else
        echo -e "│ ${YELLOW}⏳ Esperando datos...${NC}"
    fi
    
    echo -e "${MAGENTA}└────────────────────────────────────────────────────────────────────┘${NC}"
    echo ""
    
    # ========================================================================
    # 4. ÚLTIMAS ACTUALIZACIONES DEL LOG
    # ========================================================================
    echo -e "${MAGENTA}┌─ 📝 ÚLTIMAS ACTUALIZACIONES ───────────────────────────────────────┐${NC}"
    
    if [ -n "$LATEST_LOG" ]; then
        echo -e "│ ${BLUE}Log: $(basename "$LATEST_LOG")${NC}"
        echo -e "${MAGENTA}├────────────────────────────────────────────────────────────────────┤${NC}"
        
        # Últimas 3 líneas relevantes
        grep -E "Clase [0-9].*\|.*\|" "$LATEST_LOG" | tail -3 | while IFS= read -r line; do
            echo "│ ${line:0:68}"
        done
        
        if [ $(grep -c "Clase [0-9]" "$LATEST_LOG") -eq 0 ]; then
            echo -e "│ ${YELLOW}⏳ Sin actualizaciones aún${NC}"
        fi
    else
        echo -e "│ ${YELLOW}⚠️  No hay log disponible${NC}"
    fi
    
    echo -e "${MAGENTA}└────────────────────────────────────────────────────────────────────┘${NC}"
    echo ""
    
    # ========================================================================
    # Footer
    # ========================================================================
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}💡 Actualización automática en ${INTERVAL}s | Ctrl+C para salir${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Esperar antes de la siguiente actualización
    sleep $INTERVAL
done

