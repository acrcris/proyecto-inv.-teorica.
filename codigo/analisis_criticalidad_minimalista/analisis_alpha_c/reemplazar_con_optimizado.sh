#!/bin/bash
# reemplazar_con_optimizado.sh
# Reemplaza el proceso actual con la versión optimizada, manteniendo el progreso

echo "========================================================================"
echo "REEMPLAZAR PROCESO CON VERSIÓN OPTIMIZADA"
echo "========================================================================"
echo ""

# Directorio base
BASE_DIR="/Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/analisis_criticalidad_minimalista"
cd "$BASE_DIR/analisis_alpha_c" || exit 1

# Verificar base de datos existente
DB_PATH="resultados_criticalidad_refactorizado.db"

if [ ! -f "$DB_PATH" ]; then
    echo "❌ Error: No se encuentra la base de datos $DB_PATH"
    exit 1
fi

# Contar imágenes ya procesadas
PROCESADAS=$(python3 -c "
import sqlite3
conn = sqlite3.connect('$DB_PATH')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM resultados')
count = cursor.fetchone()[0]
print(count)
conn.close()
")

echo "✅ Base de datos encontrada: $DB_PATH"
echo "✅ Imágenes ya procesadas: $PROCESADAS / 60,000"
echo ""

# Detener proceso anterior si existe
echo "Verificando procesos anteriores..."
OLD_PID=$(ps aux | grep "[a]nalizar_con_sqlite" | grep -v "OPTIMIZADO" | awk '{print $2}')

if [ ! -z "$OLD_PID" ]; then
    echo "⚠️  Deteniendo proceso anterior (PID: $OLD_PID)..."
    kill $OLD_PID
    sleep 2
    
    # Verificar si se detuvo
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "⚠️  Proceso no se detuvo, forzando..."
        kill -9 $OLD_PID
        sleep 1
    fi
    echo "✅ Proceso anterior detenido"
else
    echo "✅ No hay procesos anteriores corriendo"
fi
echo ""

# Verificar GPU
echo "Verificando GPU..."
GPU_STATUS=$(python3 -c "import torch; print('MPS' if torch.backends.mps.is_available() else 'CPU')")
echo "✅ Dispositivo: $GPU_STATUS"
echo ""

# Iniciar versión optimizada
echo "========================================================================"
echo "INICIANDO VERSIÓN OPTIMIZADA"
echo "========================================================================"
echo ""
echo "Configuración:"
echo "  Base de datos: $DB_PATH (continúa desde imagen $PROCESADAS)"
echo "  Batch size: 50 alphas simultáneos"
echo "  Dispositivo: $GPU_STATUS"
echo "  Log: analisis_optimizado_continuacion.log"
echo ""

# Crear comando
PYTHON_CMD="python3 analizar_con_sqlite_OPTIMIZADO.py \
    --db_path $DB_PATH \
    --device auto \
    --alpha_batch_size 50 \
    --data_root ../data"

# Ejecutar en background
nohup $PYTHON_CMD > analisis_optimizado_continuacion.log 2>&1 &
NEW_PID=$!

# Esperar un momento para verificar que inició
sleep 3

if ps -p $NEW_PID > /dev/null 2>&1; then
    echo "✅ Proceso optimizado iniciado exitosamente"
    echo "   PID: $NEW_PID"
    echo ""
    echo "========================================================================"
    echo "MONITOREO"
    echo "========================================================================"
    echo ""
    echo "Ver progreso en tiempo real:"
    echo "  tail -f analisis_optimizado_continuacion.log"
    echo ""
    echo "Ver últimas 50 líneas:"
    echo "  tail -50 analisis_optimizado_continuacion.log"
    echo ""
    echo "Verificar que está corriendo:"
    echo "  ps -p $NEW_PID"
    echo ""
    echo "Detener el proceso:"
    echo "  kill $NEW_PID"
    echo ""
    echo "Velocidad esperada: ~960 imgs/hora (vs ~244 anterior)"
    echo "Tiempo restante estimado: $(python3 -c "print(f'{(60000-$PROCESADAS)/960:.1f} horas')")"
    echo ""
    
    # Mostrar primeras líneas del log
    echo "Primeras líneas del log:"
    echo "------------------------------------------------------------------------"
    head -20 analisis_optimizado_continuacion.log
    
else
    echo "❌ Error: El proceso no se inició correctamente"
    echo "Ver log para detalles:"
    echo "  cat analisis_optimizado_continuacion.log"
    exit 1
fi
