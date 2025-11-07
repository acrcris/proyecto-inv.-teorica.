#!/bin/bash
# monitor_refactorizado.sh
# Monitor para el proceso de análisis REFACTORIZADO

DB="resultados_criticalidad_refactorizado.db"
LOG="analisis_refactorizado.log"

# Verificar si el proceso está corriendo
PID=$(pgrep -f "analizar_con_sqlite_REFACTORIZADO.py")

if [ -z "$PID" ]; then
    echo "❌ Proceso NO está corriendo"
else
    echo "✅ Proceso activo (PID: $PID)"
    
    # Información del proceso
    ps -p $PID -o etime,pcpu,pmem,rss | tail -1 | awk '{
        printf "   Tiempo: %s | CPU: %s%% | RAM: %.2f MB\n", $1, $2, $4/1024
    }'
fi

echo ""

# Verificar base de datos
if [ -f "$DB" ]; then
    echo "📊 Base de Datos: $DB"
    ls -lh "$DB" | awk '{print "   Tamaño:", $5}'
    
    # Contar registros
    TOTAL=$(sqlite3 "$DB" "SELECT COUNT(*) FROM resultados;" 2>/dev/null || echo "0")
    echo "   Imágenes procesadas: $TOTAL/60000 ($(awk "BEGIN {printf \"%.2f\", $TOTAL/60000*100}")%)"
    
    # Estadísticas de alpha_c
    STATS=$(sqlite3 "$DB" "SELECT AVG(alpha_c), MIN(alpha_c), MAX(alpha_c) FROM resultados;" 2>/dev/null)
    if [ ! -z "$STATS" ]; then
        echo "   α_c: $STATS" | awk -F'|' '{printf "   α_c promedio: %.6f | min: %.6f | max: %.6f\n", $1, $2, $3}'
    fi
else
    echo "⚠️  Base de datos no encontrada: $DB"
fi

echo ""

# Verificar log
if [ -f "$LOG" ]; then
    echo "📝 Últimas líneas del log:"
    tail -5 "$LOG" | sed 's/^/   /'
else
    echo "⚠️  Log no encontrado: $LOG"
fi
