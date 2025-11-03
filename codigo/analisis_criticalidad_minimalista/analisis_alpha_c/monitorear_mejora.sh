#!/bin/bash
# monitorear_mejora.sh
# Compara el progreso del proceso optimizado vs el anterior

DB_PATH="resultados_criticalidad_refactorizado.db"

echo "========================================================================"
echo "MONITOR DE MEJORA - VERSIÓN OPTIMIZADA"
echo "========================================================================"
echo ""

# Función para contar imágenes
contar_imagenes() {
    python3 -c "
import sqlite3
conn = sqlite3.connect('$DB_PATH')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM resultados')
count = cursor.fetchone()[0]
conn.close()
print(count)
"
}

# Función para calcular velocidad
calcular_velocidad() {
    local inicio=$1
    local fin=$2
    local tiempo=$3
    local diff=$((fin - inicio))
    local velocidad=$(echo "scale=1; $diff / ($tiempo / 3600)" | bc)
    echo $velocidad
}

# Contar inicial
INICIAL=$(contar_imagenes)
TIEMPO_INICIO=$(date +%s)

echo "📊 Progreso inicial: $INICIAL / 60,000 imágenes"
echo "⏱️  Iniciando monitoreo cada 60 segundos..."
echo ""
echo "Presiona Ctrl+C para detener"
echo ""
echo "Tiempo | Procesadas | Nuevas | Velocidad (imgs/h) | Tiempo restante"
echo "------------------------------------------------------------------------"

while true; do
    sleep 60
    
    ACTUAL=$(contar_imagenes)
    TIEMPO_ACTUAL=$(date +%s)
    TIEMPO_TRANSCURRIDO=$((TIEMPO_ACTUAL - TIEMPO_INICIO))
    
    NUEVAS=$((ACTUAL - INICIAL))
    VELOCIDAD=$(calcular_velocidad $INICIAL $ACTUAL $TIEMPO_TRANSCURRIDO)
    
    # Calcular tiempo restante
    RESTANTES=$((60000 - ACTUAL))
    if [ $(echo "$VELOCIDAD > 0" | bc) -eq 1 ]; then
        HORAS_RESTANTES=$(echo "scale=1; $RESTANTES / $VELOCIDAD" | bc)
    else
        HORAS_RESTANTES="N/A"
    fi
    
    # Formatear tiempo transcurrido
    MINUTOS=$((TIEMPO_TRANSCURRIDO / 60))
    
    printf "%3d min | %5d | %5d | %15.0f | %8s horas\n" \
        $MINUTOS $ACTUAL $NUEVAS $VELOCIDAD $HORAS_RESTANTES
done
