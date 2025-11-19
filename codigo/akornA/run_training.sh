#!/bin/bash

# Script para ejecutar entrenamiento de AKOrN en MNIST
# Con caffeinate para evitar suspensión del Mac

cd /Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/akorn
source /Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/.venv/bin/activate

echo "🚀 Iniciando entrenamiento con caffeinate..."
echo "📂 Directorio: $(pwd)"
echo "🐍 Python: $(which python)"
echo "⏰ Inicio: $(date)"
echo ""
echo "Los checkpoints se guardarán en: runs/mnist_maxima_precision/"
echo "Logs en: runs/mnist_maxima_precision/training.log"
echo ""

# Ejecutar con caffeinate para evitar suspensión
caffeinate -dims python train_classification.py mnist_maxima_precision \
    --data mnist \
    --epochs 200 \
    --n 4 \
    --L 3 \
    --T 5 \
    --ch 128 \
    --batchsize 128 \
    --lr 0.0001 \
    --device mps \
    --checkpoint_every 25 \
    --adveval_freq 10 \
    --pgdeval_freq 50 \
    2>&1 | tee runs/mnist_maxima_precision/training.log

echo ""
echo "⏰ Fin: $(date)"
