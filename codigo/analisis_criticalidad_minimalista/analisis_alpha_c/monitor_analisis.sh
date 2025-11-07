#!/bin/bash
# Script para monitorear el progreso del análisis de criticalidad

# Colores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

clear
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}    MONITOR DE ANÁLISIS DE CRITICALIDAD - MNIST (60,000 imágenes)${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# 1. Estado del proceso
echo -e "${GREEN}📊 Estado del Proceso:${NC}"
if pgrep -f "analizar_con_sqlite.py" > /dev/null; then
    PID=$(pgrep -f "analizar_con_sqlite.py")
    echo "✅ Proceso activo (PID: $PID)"
    ps -p $PID -o etime,pcpu,pmem,rss | tail -n 1 | awk '{printf "   Tiempo: %s | CPU: %s%% | RAM: %s%% | RSS: %.2f MB\n", $1, $2, $3, $4/1024}'
else
    echo "❌ Proceso no está corriendo"
fi
echo ""

# 2. Progreso desde la base de datos
echo -e "${GREEN}📈 Progreso Global:${NC}"
sqlite3 resultados_criticalidad.db << EOF
.mode column
.headers on
SELECT 
    COUNT(*) as "Total Procesadas",
    ROUND(COUNT(*) * 100.0 / 60000, 2) || '%' as "Progreso",
    60000 - COUNT(*) as "Restantes"
FROM resultados;
EOF
echo ""

# 3. Progreso por clase
echo -e "${GREEN}🎯 Distribución por Clase:${NC}"
sqlite3 resultados_criticalidad.db << EOF
.mode column
.headers on
SELECT 
    clase as "Clase",
    COUNT(*) as "Imgs",
    ROUND(AVG(alpha_c), 6) as "α_c Promedio",
    ROUND(MIN(alpha_c), 6) as "α_c Min",
    ROUND(MAX(alpha_c), 6) as "α_c Max"
FROM resultados 
GROUP BY clase 
ORDER BY clase;
EOF
echo ""

# 4. Velocidad de procesamiento
echo -e "${GREEN}⚡ Velocidad:${NC}"
sqlite3 resultados_criticalidad.db << EOF
SELECT 
    'Última hora: ' || COUNT(*) || ' imgs' as Metrica
FROM resultados 
WHERE datetime(timestamp) > datetime('now', '-1 hour');

SELECT 
    'Últimas 10 mins: ' || COUNT(*) || ' imgs' as Metrica
FROM resultados 
WHERE datetime(timestamp) > datetime('now', '-10 minutes');
EOF
echo ""

# 5. Últimas líneas del log
echo -e "${GREEN}📝 Últimas Actualizaciones (log):${NC}"
tail -3 analisis_sqlite.log 2>/dev/null | grep "Procesando" || echo "No hay actualizaciones recientes en el log"
echo ""

# 6. Estimación de tiempo restante
echo -e "${YELLOW}⏱️  Estimación de Tiempo:${NC}"
TOTAL=$(sqlite3 resultados_criticalidad.db "SELECT COUNT(*) FROM resultados;")
FIRST_TIME=$(sqlite3 resultados_criticalidad.db "SELECT MIN(timestamp) FROM resultados;")
LAST_TIME=$(sqlite3 resultados_criticalidad.db "SELECT MAX(timestamp) FROM resultados;")

if [ "$TOTAL" -gt 10 ]; then
    python3 << EOF
from datetime import datetime
total = $TOTAL
first = datetime.fromisoformat('$FIRST_TIME')
last = datetime.fromisoformat('$LAST_TIME')
elapsed = (last - first).total_seconds()
rate = total / elapsed if elapsed > 0 else 0
remaining = 60000 - total
eta_seconds = remaining / rate if rate > 0 else 0
eta_hours = eta_seconds / 3600
eta_days = eta_hours / 24
print(f"   Velocidad: {rate*3600:.1f} imgs/hora ({elapsed/total:.1f} seg/img)")
print(f"   Tiempo restante: ~{eta_days:.1f} días ({eta_hours:.1f} horas)")
EOF
fi

echo ""
echo -e "${BLUE}======================================================================${NC}"
echo "Ejecuta: watch -n 60 ./monitor_analisis.sh  # Para actualizar cada minuto"
echo -e "${BLUE}======================================================================${NC}"
