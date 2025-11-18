#!/bin/bash

# Script para prevenir suspensión del Mac durante el procesamiento
# Este script corre en paralelo al procesamiento principal

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔋 PREVENCIÓN DE SUSPENSIÓN ACTIVA"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "   Este script mantendrá el Mac despierto mientras:"
echo "   • calcular_c_critico_local.py esté corriendo"
echo "   • O ejecutar_con_reinicio_automatico.sh esté activo"
echo ""
echo "   Para detener: kill \$(cat prevenir_suspension.pid)"
echo ""

# Guardar PID
echo $$ > prevenir_suspension.pid

# Verificar si caffeinate está disponible
if ! command -v caffeinate &> /dev/null; then
    echo "❌ Error: caffeinate no está disponible"
    exit 1
fi

# Verificar si pmset está disponible
if ! command -v pmset &> /dev/null; then
    echo "⚠️  Advertencia: pmset no está disponible (requiere permisos)"
fi

# Contador de ciclos
CICLOS=0

echo "🚀 Iniciando monitoreo..."
echo ""

# Loop infinito que mantiene el Mac despierto
while true; do
    CICLOS=$((CICLOS + 1))
    
    # Verificar si algún proceso relevante está corriendo
    if ps aux | grep -E "calcular_c_critico_local.py|ejecutar_con_reinicio" | grep -v grep > /dev/null; then
        ESTADO="🟢 ACTIVO"
        
        # Mostrar info cada 60 ciclos (aprox. cada 5 minutos)
        if [ $((CICLOS % 60)) -eq 0 ]; then
            TIEMPO=$(date "+%H:%M:%S")
            echo "[$TIEMPO] $ESTADO - Ciclo #$CICLOS - Mac mantenido despierto"
            
            # Mostrar estado de energía si pmset está disponible
            if command -v pmset &> /dev/null; then
                SLEEP_INFO=$(pmset -g assertions | grep -i "PreventUserIdleSystemSleep\|PreventSystemSleep" | head -2)
                if [ ! -z "$SLEEP_INFO" ]; then
                    echo "   Assertions activas:"
                    echo "$SLEEP_INFO" | sed 's/^/   /'
                fi
            fi
        fi
    else
        ESTADO="🔴 INACTIVO"
        TIEMPO=$(date "+%H:%M:%S")
        echo "[$TIEMPO] $ESTADO - No se detectaron procesos activos"
        echo "   Esperando 30 segundos antes de terminar..."
        
        # Esperar 30 segundos por si el proceso está reiniciando
        sleep 30
        
        # Verificar nuevamente
        if ! ps aux | grep -E "calcular_c_critico_local.py|ejecutar_con_reinicio" | grep -v grep > /dev/null; then
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "✅ Procesamiento finalizado. Deteniendo prevención de suspensión."
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            rm -f prevenir_suspension.pid
            exit 0
        fi
    fi
    
    # Usar caffeinate con todas las opciones por 5 segundos
    # Esto crea una "assertion" que previene la suspensión
    caffeinate -dimsu -t 5 &
    
    # Esperar 5 segundos antes del siguiente ciclo
    sleep 5
done
