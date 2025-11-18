#!/bin/bash

# Script para monitorear uso de MPS (Metal Performance Shaders) en Apple Silicon

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎮 MONITOREO DE USO MPS/GPU (Apple Silicon M3)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. Verificar proceso Python
echo "🔍 1. PROCESO PYTHON"
echo "────────────────────────────────────────────────────────────────────────"
PID=$(cat procesamiento.pid 2>/dev/null)
if [ ! -z "$PID" ]; then
    ps -p $PID -o pid,ppid,%cpu,%mem,etime,command 2>/dev/null | head -2
    if [ $? -ne 0 ]; then
        echo "   ❌ Proceso no encontrado (PID: $PID)"
    fi
else
    echo "   ⚠️  No se encontró archivo procesamiento.pid"
fi
echo ""

# 2. Verificar que MPS esté disponible
echo "🔧 2. VERIFICACIÓN DE MPS"
echo "────────────────────────────────────────────────────────────────────────"
python3 -c "import torch; print(f'   PyTorch version: {torch.__version__}'); print(f'   MPS disponible: {torch.backends.mps.is_available()}'); print(f'   MPS construido: {torch.backends.mps.is_built()}')" 2>/dev/null
echo ""

# 3. Uso actual de GPU/MPS
echo "⚡ 3. USO DE GPU/METAL (requiere sudo)"
echo "────────────────────────────────────────────────────────────────────────"
echo "   Obteniendo métricas de GPU... (esto toma ~1 segundo)"
GPU_DATA=$(sudo powermetrics --samplers gpu_power -n 1 -i 1000 2>/dev/null)

# Extraer información relevante
GPU_FREQ=$(echo "$GPU_DATA" | grep "GPU HW active frequency" | awk '{print $5, $6}')
GPU_ACTIVE=$(echo "$GPU_DATA" | grep "GPU HW active residency" | awk -F':' '{print $2}' | awk '{print $1}')
GPU_IDLE=$(echo "$GPU_DATA" | grep "GPU idle residency" | awk '{print $4}')
GPU_POWER=$(echo "$GPU_DATA" | grep "GPU Power" | awk '{print $3, $4}')

echo "   📊 Frecuencia activa: $GPU_FREQ"
echo "   🔥 Residencia activa: $GPU_ACTIVE"
echo "   😴 Residencia idle: $GPU_IDLE"
echo "   ⚡ Consumo energético: $GPU_POWER"
echo ""

# Calcular uso efectivo
ACTIVE_NUM=$(echo "$GPU_ACTIVE" | sed 's/%//')
if (( $(echo "$ACTIVE_NUM > 90" | bc -l) )); then
    echo "   ✅ GPU/MPS en USO ALTO (>90%)"
elif (( $(echo "$ACTIVE_NUM > 50" | bc -l) )); then
    echo "   ⚠️  GPU/MPS en uso moderado (50-90%)"
else
    echo "   ❌ GPU/MPS en uso bajo (<50%)"
fi
echo ""

# 4. Verificar script está usando device correcto
echo "🎯 4. CONFIGURACIÓN DEL SCRIPT"
echo "────────────────────────────────────────────────────────────────────────"
if [ -f "calcular_c_critico_local.py" ]; then
    DEVICE_LINE=$(grep -n "device = " calcular_c_critico_local.py | head -1)
    echo "   $DEVICE_LINE"
    echo ""
    echo "   Para verificar en el log que está usando MPS:"
    echo "   grep 'Usando device' procesamiento_20251116_075040.log | tail -1"
fi
echo ""

# 5. Temperatura (si es posible)
echo "🌡️  5. TEMPERATURA DEL SISTEMA"
echo "────────────────────────────────────────────────────────────────────────"
echo "   Obteniendo temperaturas... (esto toma ~1 segundo)"
TEMP_DATA=$(sudo powermetrics --samplers smc -n 1 -i 1000 2>/dev/null | grep -i "CPU die temperature\|GPU die temperature\|Package" | head -3)
if [ ! -z "$TEMP_DATA" ]; then
    echo "$TEMP_DATA" | sed 's/^/   /'
else
    echo "   No disponible (requiere permisos adicionales)"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 NOTAS IMPORTANTES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "   • En Apple Silicon, la GPU (MPS) comparte memoria con CPU"
echo "   • Alto uso de CPU (~60%) + GPU activa (>90%) = MPS funcionando bien"
echo "   • La frecuencia se ajusta automáticamente según la carga"
echo "   • Bajo consumo (<500mW) es normal para operaciones de tensor"
echo ""
echo "   Para monitoreo continuo:"
echo "   watch -n 5 './ver_uso_mps.sh'"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
