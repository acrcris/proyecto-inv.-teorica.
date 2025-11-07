#!/bin/bash
# iniciar_analisis_optimizado.sh
#
# Script para iniciar el análisis de criticalidad OPTIMIZADO

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activar entorno virtual
source /home/crperezp/proyectos/ProyectoInvTeorica/Proyecto-Inv.-teorica./codigo/.venv/bin/activate 2>/dev/null || \
source ../../.venv/bin/activate 2>/dev/null || \
echo "⚠️  No se pudo activar el entorno virtual automáticamente"

# Base de datos optimizada
DB_PATH="resultados_criticalidad_optimizado.db"
LOG_FILE="analisis_optimizado.log"

echo "=============================================================="
echo "INICIANDO ANÁLISIS DE CRITICALIDAD - VERSIÓN OPTIMIZADA GPU"
echo "=============================================================="
echo "Base de datos: $DB_PATH"
echo "Log: $LOG_FILE"
echo ""
echo "Configuración optimizada:"
echo "  - Procesamiento vectorizado de alphas (batch=50)"
echo "  - Operaciones nativas PyTorch (sin NumPy)"
echo "  - Cálculo de fases optimizado para MPS"
echo "  - Sin transferencias GPU↔CPU"
echo ""
echo "Mejoras esperadas:"
echo "  - 15-60x más rápido que versión original"
echo "  - De ~14 seg/img a ~1 seg/img"
echo "  - De ~24 días a ~1-3 días para 60k imágenes"
echo "=============================================================="
echo ""

# Verificar GPU
python3 -c "
import torch
if torch.backends.mps.is_available():
    print('✅ MPS (Apple Silicon GPU) disponible')
elif torch.cuda.is_available():
    print(f'✅ CUDA GPU disponible: {torch.cuda.get_device_name()}')
else:
    print('⚠️  Solo CPU disponible (será más lento)')
" || echo "⚠️  Error verificando GPU"

echo ""
read -p "¿Deseas continuar? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelado"
    exit 1
fi

# Ejecutar en background con nohup
echo ""
echo "Iniciando proceso en background..."
nohup python3 analizar_con_sqlite_OPTIMIZADO.py \
    --device auto \
    --db_path "$DB_PATH" \
    --data_root ../../data \
    --alpha_batch_size 50 \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "✅ Proceso iniciado con PID: $PID"
echo ""
echo "Comandos útiles:"
echo "  Ver progreso:  tail -f $LOG_FILE"
echo "  Ver últimas 50 líneas:  tail -50 $LOG_FILE"
echo "  Detener proceso:  kill $PID"
echo "  Estado del proceso:  ps -p $PID"
echo ""
echo "Base de datos: $DB_PATH"
echo "=============================================================="

# Esperar un momento y mostrar inicio
sleep 3
echo ""
echo "Primeras líneas del log:"
head -30 "$LOG_FILE"
