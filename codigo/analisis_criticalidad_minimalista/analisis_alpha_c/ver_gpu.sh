#!/bin/bash
# Script para monitorear el uso de GPU (Metal) en Mac

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎮 MONITOREO DE GPU (Metal Performance Shaders)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. Información del proceso Python
echo "🔍 1. PROCESO DE PYTHON"
echo "────────────────────────────────────────────────────────────────────────"

if [ -f procesamiento.pid ]; then
    PID=$(cat procesamiento.pid)
    
    if ps -p $PID > /dev/null 2>&1; then
        echo "   PID: $PID ✅"
        
        # Información detallada del proceso
        ps -p $PID -o pid,ppid,%cpu,%mem,etime,command | tail -1 | while read line; do
            echo "   $line"
        done
        
        echo ""
        
        # Threads del proceso
        THREADS=$(ps -M -p $PID 2>/dev/null | wc -l)
        echo "   Threads activos: $((THREADS - 1))"
        
    else
        echo "   ❌ Proceso no está corriendo (PID: $PID)"
        PID=""
    fi
else
    echo "   ⚠️  No se encuentra procesamiento.pid"
    PID=""
fi

# 2. Uso general del sistema (GPU indirecta via powermetrics)
echo ""
echo "⚡ 2. USO DE GPU (Metal)"
echo "────────────────────────────────────────────────────────────────────────"
echo "   Obteniendo datos de GPU... (esto toma ~3 segundos)"

# sudo powermetrics requiere contraseña, pero podemos usar alternativas
# Intentar obtener info de GPU usando system_profiler
GPU_INFO=$(system_profiler SPDisplaysDataType 2>/dev/null | grep -A 3 "Chipset Model")

if [ -n "$GPU_INFO" ]; then
    echo "$GPU_INFO" | sed 's/^/   /'
else
    echo "   ⚠️  No se pudo obtener información de GPU"
fi

# 3. Activity Monitor via ioreg (GPU Engine utilization)
echo ""
echo "   Consultando Activity Monitor para GPU..."

# En Macs con Apple Silicon, podemos ver la frecuencia de la GPU
if [ -x "$(command -v ioreg)" ]; then
    GPU_FREQ=$(ioreg -l | grep -i "gpu" | grep -i "freq" | head -3)
    if [ -n "$GPU_FREQ" ]; then
        echo ""
        echo "   Frecuencia de GPU:"
        echo "$GPU_FREQ" | sed 's/^/   /'
    fi
fi

# 4. Memoria GPU (estimación indirecta)
echo ""
echo "💾 3. MEMORIA DEL SISTEMA (compartida con GPU en Apple Silicon)"
echo "────────────────────────────────────────────────────────────────────────"

# En Apple Silicon, GPU y CPU comparten memoria unificada
vm_stat | grep -E "Pages (free|active|inactive|speculative|wired)" | while read line; do
    echo "   $line"
done

echo ""
MEMORY_PRESSURE=$(memory_pressure 2>&1 | head -5)
if [ -n "$MEMORY_PRESSURE" ]; then
    echo "   Presión de memoria:"
    echo "$MEMORY_PRESSURE" | sed 's/^/   /'
fi

# 5. Temperatura (si está disponible)
echo ""
echo "🌡️  4. TEMPERATURA Y THROTTLING"
echo "────────────────────────────────────────────────────────────────────────"

# Intentar obtener temperatura con powermetrics (requiere sudo)
echo "   Para ver temperatura detallada ejecuta:"
echo "   sudo powermetrics --samplers smc -i 1 -n 1"
echo ""

# Alternativa: usar osx-cpu-temp si está instalado
if [ -x "$(command -v osx-cpu-temp)" ]; then
    TEMP=$(osx-cpu-temp)
    echo "   Temperatura CPU: $TEMP"
fi

# 6. Procesos usando GPU (via ps)
echo ""
echo "🔥 5. PROCESOS INTENSIVOS (Top 10 CPU)"
echo "────────────────────────────────────────────────────────────────────────"

ps aux | head -1
ps aux | grep -v "grep" | sort -rn -k 3 | head -10

# 7. Información específica del proceso Python si está corriendo
if [ -n "$PID" ]; then
    echo ""
    echo "🐍 6. DETALLES DEL PROCESO PYTHON"
    echo "────────────────────────────────────────────────────────────────────────"
    
    # Archivos abiertos (descriptores)
    FD_COUNT=$(lsof -p $PID 2>/dev/null | wc -l)
    echo "   Archivos abiertos: $FD_COUNT"
    
    # Network connections
    NET_CONN=$(lsof -p $PID -i 2>/dev/null | wc -l)
    echo "   Conexiones de red: $((NET_CONN - 1))"
    
    # Memoria detallada
    echo ""
    echo "   Uso de memoria detallado:"
    ps -p $PID -o pid,vsz,rss,pmem | tail -1 | while read pid vsz rss pmem; do
        VSZ_MB=$((vsz / 1024))
        RSS_MB=$((rss / 1024))
        echo "   Virtual: ${VSZ_MB}MB, Resident: ${RSS_MB}MB, %Mem: ${pmem}%"
    done
fi

# 8. Comando avanzado para monitoreo continuo
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 COMANDOS AVANZADOS DE MONITOREO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "   # Monitoreo en tiempo real (requiere contraseña sudo)"
echo "   sudo powermetrics --samplers gpu_power -i 1000"
echo ""
echo "   # Ver todos los procesos del usuario ordenados por CPU"
echo "   top -U \$USER"
echo ""
echo "   # Activity Monitor desde terminal"
echo "   open -a 'Activity Monitor'"
echo ""
echo "   # Monitoreo continuo del proceso Python"
echo "   watch -n 5 'ps -p \$(cat procesamiento.pid) -o pid,%cpu,%mem,etime'"
echo ""
echo "   # Ver uso de GPU con asitop (si está instalado)"
echo "   # brew install asitop"
echo "   # sudo asitop"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
