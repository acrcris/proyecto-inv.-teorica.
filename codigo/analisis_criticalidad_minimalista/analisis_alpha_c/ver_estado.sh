#!/bin/bash
# Script para verificar el estado del procesamiento de C_crítico

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 ESTADO DEL PROCESAMIENTO DE C_CRÍTICO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Verificar si el proceso está corriendo
echo ""
echo "🔍 1. ESTADO DEL PROCESO"
echo "────────────────────────────────────────────────────────────────────────"

if [ -f procesamiento.pid ]; then
    PID=$(cat procesamiento.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "   ✅ Proceso ACTIVO"
        echo "   PID: $PID"
        
        # Tiempo de ejecución
        ELAPSED=$(ps -p $PID -o etime= | xargs)
        echo "   Tiempo de ejecución: $ELAPSED"
        
        # Uso de CPU y memoria
        CPU_MEM=$(ps -p $PID -o %cpu,%mem | tail -1)
        echo "   CPU y Memoria: $CPU_MEM"
    else
        echo "   ❌ Proceso NO está corriendo"
        echo "   Último PID: $PID (terminado)"
    fi
else
    echo "   ⚠️  No se encuentra archivo procesamiento.pid"
    echo "   El proceso nunca se inició o el archivo fue eliminado"
fi

# 2. Último log activo
echo ""
echo "📝 2. LOG ACTIVO"
echo "────────────────────────────────────────────────────────────────────────"

LATEST_LOG=$(ls -t procesamiento_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "   Log: $LATEST_LOG"
    echo "   Tamaño: $(du -h "$LATEST_LOG" | cut -f1)"
    echo ""
    echo "   Últimas 5 líneas:"
    tail -5 "$LATEST_LOG" | sed 's/^/   │ /'
else
    echo "   ⚠️  No se encontraron logs"
fi

# 3. Progreso en la base de datos
echo ""
echo "💾 3. PROGRESO EN BASE DE DATOS"
echo "────────────────────────────────────────────────────────────────────────"

DB_PATH="resultados_c_critical/mnist_critical_tot.db"

if [ -f "$DB_PATH" ]; then
    echo "   Base de datos: $DB_PATH"
    echo "   Tamaño: $(du -h "$DB_PATH" | cut -f1)"
    echo ""
    
    # Contar imágenes procesadas por clase
    echo "   Imágenes procesadas por clase:"
    for i in {9..0}; do
        COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM clase_$i;" 2>/dev/null || echo "0")
        
        # Determinar total de imágenes por clase (aproximado de MNIST)
        case $i in
            0) TOTAL=5923 ;;
            1) TOTAL=6742 ;;
            2) TOTAL=5958 ;;
            3) TOTAL=6131 ;;
            4) TOTAL=5842 ;;
            5) TOTAL=5421 ;;
            6) TOTAL=5918 ;;
            7) TOTAL=6265 ;;
            8) TOTAL=5851 ;;
            9) TOTAL=5949 ;;
        esac
        
        PERCENT=$(awk "BEGIN {printf \"%.1f\", ($COUNT/$TOTAL)*100}")
        
        # Barra de progreso visual
        BARS=$(awk "BEGIN {printf \"%.0f\", ($COUNT/$TOTAL)*20}")
        BAR=""
        for ((j=0; j<$BARS; j++)); do BAR="${BAR}█"; done
        for ((j=$BARS; j<20; j++)); do BAR="${BAR}░"; done
        
        if [ "$COUNT" -eq "$TOTAL" ]; then
            STATUS="✅"
        elif [ "$COUNT" -gt 0 ]; then
            STATUS="⏳"
        else
            STATUS="⭕"
        fi
        
        printf "   %s Clase %d: [%s] %5d/%d (%5.1f%%)\n" "$STATUS" "$i" "$BAR" "$COUNT" "$TOTAL" "$PERCENT"
    done
    
    # Total general
    echo ""
    TOTAL_PROCESADAS=$(sqlite3 "$DB_PATH" "SELECT SUM(cnt) FROM (
        SELECT COUNT(*) as cnt FROM clase_0 UNION ALL
        SELECT COUNT(*) FROM clase_1 UNION ALL
        SELECT COUNT(*) FROM clase_2 UNION ALL
        SELECT COUNT(*) FROM clase_3 UNION ALL
        SELECT COUNT(*) FROM clase_4 UNION ALL
        SELECT COUNT(*) FROM clase_5 UNION ALL
        SELECT COUNT(*) FROM clase_6 UNION ALL
        SELECT COUNT(*) FROM clase_7 UNION ALL
        SELECT COUNT(*) FROM clase_8 UNION ALL
        SELECT COUNT(*) FROM clase_9
    );" 2>/dev/null || echo "0")
    
    TOTAL_MNIST=60000
    PERCENT_TOTAL=$(awk "BEGIN {printf \"%.2f\", ($TOTAL_PROCESADAS/$TOTAL_MNIST)*100}")
    
    echo "   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "   📈 TOTAL: $TOTAL_PROCESADAS / $TOTAL_MNIST imágenes ($PERCENT_TOTAL%)"
    
else
    echo "   ⚠️  Base de datos no encontrada: $DB_PATH"
fi

# 4. Estadísticas de las clases completadas
echo ""
echo "📊 4. ESTADÍSTICAS DE CLASES COMPLETADAS"
echo "────────────────────────────────────────────────────────────────────────"

if [ -f "$DB_PATH" ]; then
    COMPLETED=false
    for i in {9..0}; do
        COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM clase_$i;" 2>/dev/null || echo "0")
        
        case $i in
            0) TOTAL=5923 ;;
            1) TOTAL=6742 ;;
            2) TOTAL=5958 ;;
            3) TOTAL=6131 ;;
            4) TOTAL=5842 ;;
            5) TOTAL=5421 ;;
            6) TOTAL=5918 ;;
            7) TOTAL=6265 ;;
            8) TOTAL=5851 ;;
            9) TOTAL=5949 ;;
        esac
        
        if [ "$COUNT" -eq "$TOTAL" ]; then
            COMPLETED=true
            STATS=$(sqlite3 "$DB_PATH" "SELECT 
                printf('%.4f', AVG(c_critical)),
                printf('%.4f', MIN(c_critical)),
                printf('%.4f', MAX(c_critical)),
                printf('%.4f', 
                    (SELECT c_critical FROM clase_$i ORDER BY c_critical LIMIT 1 OFFSET (SELECT COUNT(*)/2 FROM clase_$i))
                )
            FROM clase_$i;" 2>/dev/null)
            
            if [ -n "$STATS" ]; then
                IFS='|' read -r MEAN MIN MAX MEDIAN <<< "$STATS"
                echo "   ✅ Clase $i: μ=$MEAN, min=$MIN, max=$MAX, med=$MEDIAN"
            fi
        fi
    done
    
    if [ "$COMPLETED" = false ]; then
        echo "   ⏳ Ninguna clase completada aún"
    fi
else
    echo "   ⚠️  Base de datos no disponible"
fi

# 5. Estimación de tiempo restante
echo ""
echo "⏱️  5. ESTIMACIÓN DE TIEMPO"
echo "────────────────────────────────────────────────────────────────────────"

if [ -f "$DB_PATH" ] && [ -n "$LATEST_LOG" ]; then
    # Obtener velocidad promedio del log (segundos/imagen)
    SPEED=$(grep -oE '[0-9]+\.[0-9]+s/it' "$LATEST_LOG" | tail -1 | grep -oE '[0-9]+\.[0-9]+')
    
    if [ -n "$SPEED" ]; then
        REMAINING=$((60000 - TOTAL_PROCESADAS))
        HOURS=$(awk "BEGIN {printf \"%.1f\", ($REMAINING * $SPEED) / 3600}")
        MINUTES=$(awk "BEGIN {printf \"%.0f\", (($REMAINING * $SPEED) / 60) % 60}")
        
        echo "   Velocidad actual: ~${SPEED}s por imagen"
        echo "   Imágenes restantes: $REMAINING"
        echo "   Tiempo estimado: ~${HOURS} horas y ${MINUTES} minutos"
        
        # Estimar hora de finalización
        if command -v gdate &> /dev/null; then
            # macOS con coreutils instalado
            END_TIMESTAMP=$(gdate -d "+${HOURS} hours +${MINUTES} minutes" "+%Y-%m-%d %H:%M:%S" 2>/dev/null)
        else
            # macOS nativo
            TOTAL_MINUTES=$(awk "BEGIN {printf \"%.0f\", ($REMAINING * $SPEED) / 60}")
            END_TIMESTAMP=$(date -v+${TOTAL_MINUTES}M "+%Y-%m-%d %H:%M:%S" 2>/dev/null)
        fi
        
        if [ -n "$END_TIMESTAMP" ]; then
            echo "   Finalización estimada: $END_TIMESTAMP"
        fi
        
        # Calcular ETA más preciso
        if [ "$REMAINING" -gt 0 ]; then
            PERCENT_COMPLETE=$(awk "BEGIN {printf \"%.2f\", ($TOTAL_PROCESADAS/60000)*100}")
            echo "   Progreso general: ${PERCENT_COMPLETE}% completado"
        fi
    else
        echo "   ⚠️  No se pudo calcular velocidad actual"
        echo "   Revisar el log para más detalles: $LATEST_LOG"
    fi
else
    echo "   ⚠️  No hay suficiente información para estimar"
fi

# 6. Monitoreo de recursos del sistema
echo ""
echo "💻 6. USO DE RECURSOS DEL SISTEMA"
echo "────────────────────────────────────────────────────────────────────────"

if [ -f procesamiento.pid ]; then
    PID=$(cat procesamiento.pid)
    if ps -p $PID > /dev/null 2>&1; then
        # Uso detallado de CPU y memoria
        PROCESS_INFO=$(ps -p $PID -o %cpu,%mem,rss,vsz | tail -1)
        read CPU_USAGE MEM_PERCENT RSS VSZ <<< "$PROCESS_INFO"
        
        RSS_MB=$(awk "BEGIN {printf \"%.1f\", $RSS/1024}")
        VSZ_MB=$(awk "BEGIN {printf \"%.1f\", $VSZ/1024}")
        
        echo "   CPU: ${CPU_USAGE}%"
        echo "   Memoria: ${MEM_PERCENT}% (RSS: ${RSS_MB} MB, VSZ: ${VSZ_MB} MB)"
        
        # Verificar si está usando GPU/MPS
        if command -v nvidia-smi &> /dev/null; then
            echo ""
            echo "   GPU (NVIDIA):"
            nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
            awk '{printf "   └─ Utilización: %s%%, Memoria: %s/%s MB\n", $1, $2, $3}'
        fi
        
        # Estado de caffeinate (prevención de suspensión)
        if pgrep -f "caffeinate.*calcular_c_critico" > /dev/null 2>&1; then
            echo "   ☕ Prevención de suspensión: ACTIVA (caffeinate)"
        else
            echo "   ⚠️  Prevención de suspensión: INACTIVA"
        fi
    fi
fi

# Espacio en disco
DISK_USAGE=$(df -h . | tail -1 | awk '{print $5}')
DISK_AVAIL=$(df -h . | tail -1 | awk '{print $4}')
echo ""
echo "   Disco disponible: $DISK_AVAIL (usado: $DISK_USAGE)"

# 7. Historial reciente de progreso
echo ""
echo "📈 7. HISTORIAL DE PROGRESO RECIENTE"
echo "────────────────────────────────────────────────────────────────────────"

if [ -f "$LATEST_LOG" ]; then
    # Extraer líneas de progreso del log
    echo "   Últimas actualizaciones de progreso:"
    grep -E "Clase [0-9].*\|.*\|" "$LATEST_LOG" | tail -5 | sed 's/^/   │ /'
    
    if [ $(grep -c "Clase [0-9]" "$LATEST_LOG") -eq 0 ]; then
        echo "   ⏳ Aún no hay progreso registrado en el log actual"
    fi
else
    echo "   ⚠️  No hay log disponible"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 COMANDOS ÚTILES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -n "$LATEST_LOG" ]; then
    echo "   # Ver log en tiempo real"
    echo "   tail -f $LATEST_LOG"
    echo ""
fi

echo "   # Monitoreo automático cada 30 segundos"
echo "   watch -n 30 './ver_estado.sh'"
echo ""
echo "   # Ver estado completo de todas las clases"
echo "   ./ver_estado_completo.sh"
echo ""
echo "   # Consultar clase específica (ej: clase 5)"
echo "   sqlite3 $DB_PATH \"SELECT COUNT(*), AVG(c_critical), MIN(c_critical), MAX(c_critical) FROM clase_5;\""
echo ""
echo "   # Ver últimas imágenes procesadas de una clase"
echo "   sqlite3 $DB_PATH \"SELECT * FROM clase_0 ORDER BY ROWID DESC LIMIT 5;\""
echo ""
echo "   # Ver distribución de C_crítico para una clase"
echo "   sqlite3 $DB_PATH \"SELECT ROUND(c_critical, 1) as c_range, COUNT(*) FROM clase_5 GROUP BY c_range ORDER BY c_range;\""
echo ""

if [ -f procesamiento.pid ]; then
    PID=$(cat procesamiento.pid)
    echo "   # Detener el proceso gracefully"
    echo "   kill $PID"
    echo ""
    echo "   # Forzar detención del proceso"
    echo "   kill -9 $PID"
    echo ""
fi

echo "   # Verificar todos los logs disponibles"
echo "   ls -lth procesamiento_*.log | head -10"
echo ""
echo "   # Buscar errores en logs"
echo "   grep -i error procesamiento_*.log"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🔄 Para actualizar este estado automáticamente:"
echo "   watch -n 30 './ver_estado.sh'     # Actualiza cada 30 segundos"
echo "   watch -n 60 './ver_estado.sh'     # Actualiza cada 1 minuto"
echo ""
