#!/bin/bash
# Script para entrenar AKOrN en MNIST
# Diferentes configuraciones experimentales

set -e

echo "=========================================="
echo "ENTRENAMIENTO AKORN EN MNIST"
echo "=========================================="

# Asegurarse de que estamos en el directorio correcto
cd /Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/akorn

# Activar el entorno virtual si existe
if [ -f ".venv/bin/activate" ]; then
    echo "Activando entorno virtual..."
    source .venv/bin/activate
fi

# Verificar que el dataset esté descargado
echo ""
echo "[1/2] Verificando dataset MNIST..."
python3 setup_mnist.py

if [ $? -ne 0 ]; then
    echo "✗ Error configurando MNIST. Abortando."
    exit 1
fi

echo ""
echo "[2/2] Configuración completada. Eligiendo experimento..."
echo ""
echo "Experimentos disponibles:"
echo "  1) mnist_baseline_small    - Modelo pequeño para pruebas rápidas"
echo "  2) mnist_baseline_medium   - Modelo mediano (recomendado)"
echo "  3) mnist_baseline_large    - Modelo grande para máxima precisión"
echo "  4) mnist_kuramoto_debug    - Debug: usar con T más alto"
echo ""
read -p "Selecciona experimento (1-4): " choice

case $choice in
    1)
        echo "Ejecutando: mnist_baseline_small"
        python3 train_classification.py mnist_baseline_small \
            --data mnist \
            --epochs 20 \
            --checkpoint_every 10 \
            --lr 0.0001 \
            --batchsize 128 \
            --n 2 \
            --L 2 \
            --T 2 \
            --ch 32 \
            --adveval_freq 5
        ;;
    2)
        echo "Ejecutando: mnist_baseline_medium"
        python3 train_classification.py mnist_baseline_medium \
            --data mnist \
            --epochs 100 \
            --checkpoint_every 25 \
            --lr 0.0001 \
            --batchsize 128 \
            --n 3 \
            --L 2 \
            --T 3 \
            --ch 64 \
            --adveval_freq 10
        ;;
    3)
        echo "Ejecutando: mnist_baseline_large"
        python3 train_classification.py mnist_baseline_large \
            --data mnist \
            --epochs 200 \
            --checkpoint_every 50 \
            --lr 0.0001 \
            --batchsize 256 \
            --n 4 \
            --L 3 \
            --T 4 \
            --ch 128 \
            --adveval_freq 20
        ;;
    4)
        echo "Ejecutando: mnist_kuramoto_debug (T=10 para sincronización completa)"
        python3 train_classification.py mnist_kuramoto_debug \
            --data mnist \
            --epochs 50 \
            --checkpoint_every 10 \
            --lr 0.0001 \
            --batchsize 64 \
            --n 2 \
            --L 2 \
            --T 10 \
            --ch 64 \
            --adveval_freq 5
        ;;
    *)
        echo "Opción inválida. Abortando."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✓ Entrenamiento completado"
echo "=========================================="
echo ""
echo "Resultados guardados en: runs/mnist_*/"
echo "Visualizar con TensorBoard:"
echo "  tensorboard --logdir=runs/mnist_baseline_medium"
echo ""
