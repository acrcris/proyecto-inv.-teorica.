#!/bin/bash

# Script de monitoreo para ejecución en Mac (Apple M3)
# Versión: resultados_kuramoto_TRAIN_MAC_60k

RESULTS_DIR="resultados_kuramoto_TRAIN_MAC_60k"
CHECKPOINT_DIR="${RESULTS_DIR}/checkpoints"
FINAL_FILE="${RESULTS_DIR}/metricas_completas_TRAIN_MAC_60k.pt"

echo "======================================================================"
echo "PROGRESO - ANÁLISIS KURAMOTO TRAIN SET (60k) - APPLE M3"
echo "======================================================================"
echo ""

# Verificar si existe el directorio
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "⚠️  Directorio de checkpoints no encontrado: $CHECKPOINT_DIR"
    echo "   El proceso aún no ha iniciado o hay un error."
    exit 1
fi

# Contar checkpoints
NUM_CHECKPOINTS=$(ls -1 "$CHECKPOINT_DIR"/checkpoint_*.pt 2>/dev/null | wc -l | tr -d ' ')
IMGS_PROCESADAS=$((NUM_CHECKPOINTS * 100))
TOTAL_IMGS=60000
PROGRESO_PCT=$(echo "scale=1; $IMGS_PROCESADAS * 100 / $TOTAL_IMGS" | bc)

echo "📊 PROGRESO:"
echo "   Checkpoints guardados: $NUM_CHECKPOINTS / 600"
echo "   Imágenes procesadas:   $IMGS_PROCESADAS / $TOTAL_IMGS"
echo "   Progreso:              $PROGRESO_PCT%"
echo ""

# Calcular imágenes restantes
IMGS_RESTANTES=$((TOTAL_IMGS - IMGS_PROCESADAS))

if [ $IMGS_RESTANTES -gt 0 ]; then
    # Estimar tiempo restante (velocidad estimada: 3.5 img/s para Apple M3)
    SEGUNDOS_RESTANTES=$(echo "scale=0; $IMGS_RESTANTES / 3.5" | bc)
    HORAS_RESTANTES=$(echo "scale=1; $SEGUNDOS_RESTANTES / 3600" | bc)
    MINUTOS_RESTANTES=$(echo "scale=0; $SEGUNDOS_RESTANTES / 60" | bc)
    
    echo "⏱️  ESTIMACIÓN:"
    echo "   Imágenes restantes:  $IMGS_RESTANTES"
    echo "   Tiempo restante:     ~$HORAS_RESTANTES horas ($MINUTOS_RESTANTES min)"
    echo "   Velocidad estimada:  ~3.5 img/s (Apple M3 MPS)"
    echo ""
fi

# Verificar último checkpoint
if [ $NUM_CHECKPOINTS -gt 0 ]; then
    ULTIMO_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/checkpoint_*.pt 2>/dev/null | head -1)
    if [ -n "$ULTIMO_CHECKPOINT" ]; then
        TIMESTAMP=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$ULTIMO_CHECKPOINT" 2>/dev/null || stat -c "%y" "$ULTIMO_CHECKPOINT" 2>/dev/null | cut -d'.' -f1)
        TAMANIO=$(du -h "$ULTIMO_CHECKPOINT" | cut -f1)
        echo "💾 ÚLTIMO CHECKPOINT:"
        echo "   Archivo:   $(basename "$ULTIMO_CHECKPOINT")"
        echo "   Timestamp: $TIMESTAMP"
        echo "   Tamaño:    $TAMANIO"
        echo ""
    fi
fi

# Verificar espacio en disco
ESPACIO_LIBRE=$(df -h . | tail -1 | awk '{print $4}')
echo "💿 ESPACIO EN DISCO:"
echo "   Disponible: $ESPACIO_LIBRE"
echo ""

# Verificar si el proceso está corriendo
if pgrep -f "run_kuramoto_TRAIN_MAC.py" > /dev/null; then
    PID=$(pgrep -f "run_kuramoto_TRAIN_MAC.py")
    echo "✅ PROCESO ACTIVO:"
    echo "   PID: $PID"
    echo "   Comando: ps -p $PID -o pid,etime,%cpu,%mem,command"
    ps -p $PID -o pid,etime,%cpu,%mem,command 2>/dev/null || echo "   (Info no disponible)"
    echo ""
else
    echo "⚠️  PROCESO NO DETECTADO"
    echo "   El script run_kuramoto_TRAIN_MAC.py no está corriendo"
    echo ""
    
    # Verificar si ya terminó
    if [ -f "$FINAL_FILE" ]; then
        echo "✅ PROCESAMIENTO COMPLETO"
        echo "   Archivo final encontrado: $FINAL_FILE"
        TAMANIO_FINAL=$(du -h "$FINAL_FILE" | cut -f1)
        echo "   Tamaño: $TAMANIO_FINAL"
        echo ""
    fi
fi

# Tamaño total del directorio de resultados
if [ -d "$RESULTS_DIR" ]; then
    TAMANIO_TOTAL=$(du -sh "$RESULTS_DIR" | cut -f1)
    echo "📁 TAMAÑO TOTAL RESULTADOS:"
    echo "   $TAMANIO_TOTAL en $RESULTS_DIR/"
    echo ""
fi

echo "======================================================================"
echo "COMANDOS ÚTILES:"
echo "======================================================================"
echo "  Ver log en tiempo real:"
echo "    tail -f kuramoto_train_mac.log"
echo ""
echo "  Ver uso de recursos:"
echo "    top -pid \$(pgrep -f run_kuramoto_TRAIN_MAC.py)"
echo ""
echo "  Detener proceso:"
echo "    kill \$(cat kuramoto_train_mac.pid)"
echo ""
echo "======================================================================"
