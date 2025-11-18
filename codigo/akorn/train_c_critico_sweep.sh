#!/bin/bash

# Script para entrenar AKOrN con diferentes valores de C (acoplamiento externo)
# Basado en los centros de bins de C_crítico encontrados en el análisis

# Valores de C a probar (centros de bins del análisis de distribución)
C_VALUES=(
    0.0062 0.0186 0.031 0.0434 0.0558 0.0682 0.0806 0.093 0.1054 
    0.1178 0.1302 0.1426 0.155 0.1674 0.1798 0.1922 0.2046 0.217 
    0.2294 0.2418 0.2542 0.2666 0.279 0.2914 0.3038 0.3162
)

# Configuración base (matching notebook parameters)
BASE_DIR="./runs/c_critico_sweep"
EPOCHS=50
BATCH_SIZE=128
LR=0.0001
DATA="mnist"

# Parámetros del modelo (ajustados para AKOrN - solo soporta n par)
N=2  # AKOrN solo soporta n par (2, 4, 6, ...)
CH=64  # Divisible por 2
T=30
L=3
KSIZES="7 5 3"
GAMMA=0.7

echo "=========================================="
echo "ENTRENAMIENTO SWEEP DE C_SCALE"
echo "=========================================="
echo "Total de experimentos: ${#C_VALUES[@]}"
echo "Épocas por experimento: $EPOCHS"
echo "Dataset: $DATA"
echo ""

# Contador de progreso
total=${#C_VALUES[@]}
current=0

# Entrenar con cada valor de C
for c_val in "${C_VALUES[@]}"; do
    current=$((current + 1))
    exp_name="c_scale_${c_val}"
    
    echo ""
    echo "=========================================="
    echo "Experimento $current/$total"
    echo "C_scale = $c_val"
    echo "Experimento: $exp_name"
    echo "=========================================="
    echo ""
    
    # Ejecutar entrenamiento en background
    python train_classification.py "$exp_name" \
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
        --c_scale $c_val \
        --checkpoint_every 25 \
        --adveval_freq 10 \
        --device auto \
        > "${BASE_DIR}/logs_${exp_name}.txt" 2>&1 &
    
    # Guardar PID
    echo $! > "${BASE_DIR}/pid_${exp_name}.txt"
    
    echo "✓ Entrenamiento iniciado en background (PID: $!)"
    echo "  Ver logs en: ${BASE_DIR}/logs_${exp_name}.txt"
    
    # Esperar un poco entre lanzamientos para evitar sobrecarga
    sleep 2
done

echo ""
echo "=========================================="
echo "TODOS LOS ENTRENAMIENTOS INICIADOS"
echo "=========================================="
echo "Total: $total experimentos en background"
echo ""
echo "Para monitorear el progreso:"
echo "  tail -f ${BASE_DIR}/logs_c_scale_*.txt"
echo ""
echo "Para ver los procesos activos:"
echo "  ps aux | grep train_classification.py"
echo ""
echo "Para detener todos los entrenamientos:"
echo "  kill \$(cat ${BASE_DIR}/pid_*.txt)"
echo "=========================================="
