#!/bin/bash

# Script para entrenar un solo experimento con C_scale específico
# Optimizado para Apple Silicon con MPS

set -e

# Parámetros del experimento
C_SCALE=0.0062
EXP_NAME="c_scale_${C_SCALE}"
LOG_DIR="runs/c_critico_single"
LOG_FILE="${LOG_DIR}/logs_${EXP_NAME}.txt"
PID_FILE="${LOG_DIR}/pid_${EXP_NAME}.txt"

# Configuración del modelo (optimizada)
DATA="mnist"
EPOCHS=15
BATCH_SIZE=128
LR=0.0001
N=2
CH=64
T=10
L=2
KSIZES="7 5"
GAMMA=0.7
CHECKPOINT_EVERY=5

# Crear directorio de logs
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "  ENTRENAMIENTO INDIVIDUAL"
echo "=========================================="
echo "Experimento: $EXP_NAME"
echo "C_scale: $C_SCALE"
echo "Dataset: $DATA"
echo "Épocas: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Checkpoints cada: $CHECKPOINT_EVERY épocas"
echo ""
echo "Configuración del modelo:"
echo "  • n=$N, ch=$CH"
echo "  • T=$T, L=$L"
echo "  • ksizes=$KSIZES"
echo "  • gamma=$GAMMA"
echo ""
echo "Optimizaciones:"
echo "  • Device: auto (MPS si disponible)"
echo "  • caffeinate: evita suspensión de Mac"
echo "  • Log: $LOG_FILE"
echo ""
echo "=========================================="
echo ""

# Ejecutar con caffeinate para evitar suspensión
echo "→ Iniciando entrenamiento..."
caffeinate -i /Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/.venv/bin/python train_classification.py "$EXP_NAME" \
    --data "$DATA" \
    --epochs $EPOCHS \
    --batchsize $BATCH_SIZE \
    --lr $LR \
    --n $N \
    --ch $CH \
    --T $T \
    --L $L \
    --ksizes $KSIZES \
    --gamma $GAMMA \
    --c_scale $C_SCALE \
    --checkpoint_every $CHECKPOINT_EVERY \
    --adveval_freq 10 \
    --device auto \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo $PID > "$PID_FILE"
echo "  ✓ Iniciado (PID: $PID)"
echo "  ✓ Log: $LOG_FILE"
echo ""
echo "=========================================="
echo ""
echo "Para monitorear el progreso:"
echo "  tail -f $LOG_FILE | grep 'Epoch:'"
echo ""
echo "Para ver el estado:"
echo "  ps -p $PID -o pid,etime,pcpu,rss"
echo ""
echo "Para detener:"
echo "  kill $PID"
echo ""
