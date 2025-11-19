#!/bin/bash

# Script para entrenar AKOrN con diferentes valores de C de forma SECUENCIAL
# Ejecuta 3 entrenamientos en paralelo y espera a que terminen antes de lanzar los siguientes

# Valores de C a probar (centros de bins del análisis de distribución)
C_VALUES=(
    0.0062 0.0186 0.031 0.0434 0.0558 0.0682 0.0806 0.093 0.1054 
    0.1178 0.1302 0.1426 0.155 0.1674 0.1798 0.1922 0.2046 0.217 
    0.2294 0.2418 0.2542 0.2666 0.279 0.2914 0.3038 0.3162
)

# Configuración base
BASE_DIR="./runs/c_critico_sweep"
mkdir -p "$BASE_DIR"

# Parámetros optimizados para MPS (Apple Silicon)
EPOCHS=15          # Reducido de 50 para tiempos razonables
BATCH_SIZE=128     # Optimizado para rendimiento en MPS
LR=0.0001
DATA="mnist"

# Parámetros del modelo (n=2 para compatibilidad con OmegaLayer)
N=2
CH=64
T=10              # Reducido de 30 para acelerar cómputo
L=2               # Reducido de 3 para menos parámetros
KSIZES="7 5"      # Ajustado a L=2
GAMMA=0.7

# Checkpoints frecuentes
CHECKPOINT_EVERY=5  # Cada 5 épocas (antes 10)

# Número de entrenamientos en paralelo
PARALLEL_JOBS=3

echo "=========================================="
echo "ENTRENAMIENTO SECUENCIAL DE C_SCALE"
echo "=========================================="
echo "Total de experimentos: ${#C_VALUES[@]}"
echo "Épocas por experimento: $EPOCHS"
echo "Checkpoints cada: $CHECKPOINT_EVERY épocas"
echo "Jobs en paralelo: $PARALLEL_JOBS"
echo "Dataset: $DATA"
echo ""

# Contador de progreso
total=${#C_VALUES[@]}
current=0
batch=0

# Procesar en lotes de PARALLEL_JOBS
for ((i=0; i<$total; i+=PARALLEL_JOBS)); do
    batch=$((batch + 1))
    echo ""
    echo "=========================================="
    echo "LOTE $batch - Iniciando $PARALLEL_JOBS entrenamientos"
    echo "=========================================="
    
    # Array para guardar PIDs del lote actual
    pids=()
    
    # Lanzar PARALLEL_JOBS entrenamientos
    for ((j=0; j<PARALLEL_JOBS; j++)); do
        idx=$((i + j))
        if [ $idx -ge $total ]; then
            break
        fi
        
        c_val=${C_VALUES[$idx]}
        current=$((idx + 1))
        exp_name="c_scale_${c_val}"
        
        echo ""
        echo "→ Experimento $current/$total: C_scale = $c_val"
        
        # Ejecutar entrenamiento en background
        /Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/.venv/bin/python train_classification.py "$exp_name" \
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
            --checkpoint_every $CHECKPOINT_EVERY \
            --adveval_freq 10 \
            --device auto \
            > "${BASE_DIR}/logs_${exp_name}.txt" 2>&1 &
        
        pid=$!
        pids+=($pid)
        echo $pid > "${BASE_DIR}/pid_${exp_name}.txt"
        echo "  ✓ Iniciado (PID: $pid)"
    done
    
    # Esperar a que todos los jobs del lote terminen
    echo ""
    echo "⏳ Esperando a que terminen los $((${#pids[@]})) entrenamientos del lote $batch..."
    for pid in "${pids[@]}"; do
        wait $pid
        exit_code=$?
        echo "  ✓ Proceso $pid terminado (exit code: $exit_code)"
    done
    
    echo ""
    echo "✅ Lote $batch completado"
done

echo ""
echo "=========================================="
echo "TODOS LOS ENTRENAMIENTOS COMPLETADOS"
echo "=========================================="
echo "Total: $total experimentos ejecutados"
echo ""
echo "Resultados guardados en: ${BASE_DIR}/"
echo "  - Checkpoints: ${BASE_DIR}/c_scale_*/checkpoint_*.pth"
echo "  - Logs: ${BASE_DIR}/logs_c_scale_*.txt"
echo "  - TensorBoard: tensorboard --logdir ${BASE_DIR}"
echo "=========================================="
