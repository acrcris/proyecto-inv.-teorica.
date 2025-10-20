#!/bin/bash

# Script de monitoreo para TRAIN SET (60,000 imágenes)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MONITOREO - KURAMOTO SOBRE MNIST TRAINING SET (60,000 imágenes)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Verificar si el proceso está corriendo
if [ -f "kuramoto_train.pid" ]; then
    PID=$(cat kuramoto_train.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Proceso activo (PID: $PID)"
        echo ""
        
        # Información del proceso
        echo "📊 Información del proceso:"
        ps -p $PID -o pid,user,%cpu,%mem,etime,cmd --no-headers | \
            awk '{printf "   PID: %s\n   Usuario: %s\n   CPU: %s%%\n   Memoria: %s%%\n   Tiempo: %s\n", $1, $2, $3, $4, $5}'
        echo ""
    else
        echo "⚠️  Proceso no está corriendo (PID $PID no encontrado)"
        echo ""
    fi
else
    echo "⚠️  Archivo kuramoto_train.pid no encontrado"
    echo "   El proceso no se inició o fue eliminado"
    echo ""
fi

# Verificar checkpoints
echo "💾 Checkpoints guardados:"
CHECKPOINT_DIR="resultados_kuramoto_TRAIN_FULL_60k/checkpoints"

if [ -d "$CHECKPOINT_DIR" ]; then
    NUM_CHECKPOINTS=$(ls -1 $CHECKPOINT_DIR/checkpoint_*.pt 2>/dev/null | wc -l)
    
    if [ $NUM_CHECKPOINTS -gt 0 ]; then
        echo "   Total: $NUM_CHECKPOINTS checkpoints"
        
        # Último checkpoint
        ULTIMO=$(ls -1t $CHECKPOINT_DIR/checkpoint_*.pt 2>/dev/null | head -1)
        if [ ! -z "$ULTIMO" ]; then
            NOMBRE=$(basename "$ULTIMO")
            # Extraer número del checkpoint (checkpoint_XXXXX.pt)
            NUM=$(echo "$NOMBRE" | grep -o '[0-9]\+' | head -1)
            IMGS_PROCESADAS=$((NUM + 1))
            PROGRESO=$(echo "scale=1; $IMGS_PROCESADAS * 100 / 60000" | bc)
            
            echo "   Último: $NOMBRE"
            echo "   Tamaño: $(du -h "$ULTIMO" | cut -f1)"
            echo "   Fecha: $(stat -c '%y' "$ULTIMO" | cut -d'.' -f1)"
            echo ""
            echo "📈 Progreso estimado:"
            echo "   Imágenes procesadas: ~$IMGS_PROCESADAS / 60,000"
            echo "   Completado: ~${PROGRESO}%"
            
            # Calcular tiempo restante
            if [ $IMGS_PROCESADAS -gt 0 ]; then
                VELOCIDAD=4.5  # img/s estimada
                IMGS_RESTANTES=$((60000 - IMGS_PROCESADAS))
                SEGUNDOS_RESTANTES=$(echo "scale=0; $IMGS_RESTANTES / $VELOCIDAD" | bc)
                HORAS=$(echo "scale=1; $SEGUNDOS_RESTANTES / 3600" | bc)
                echo "   Tiempo estimado restante: ~${HORAS} horas"
            fi
        fi
    else
        echo "   No hay checkpoints guardados aún"
    fi
else
    echo "   ⚠️  Directorio de checkpoints no existe"
fi

echo ""

# Verificar logs
echo "📜 Últimas líneas del log:"
LOG_FILE="kuramoto_train.log"

if [ -f "$LOG_FILE" ]; then
    # Intentar mostrar últimas líneas del log
    echo "   (Mostrando últimas 5 líneas con información)"
    tail -100 "$LOG_FILE" 2>/dev/null | grep -E "Procesando|img/s|Checkpoint|Clase" | tail -5
elif [ -f "nohup.out" ]; then
    echo "   (Mostrando de nohup.out)"
    tail -100 "nohup.out" 2>/dev/null | grep -E "Procesando|img/s|Checkpoint|Clase" | tail -5
else
    echo "   ⚠️  No se encontró archivo de log"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 Comandos útiles:"
echo "   ./ver_progreso_TRAIN.sh  - Ver este resumen"
echo "   tail -f kuramoto_train.log - Seguir log en tiempo real"
echo "   kill \$(cat kuramoto_train.pid) - Detener proceso"
echo "   ls -lh $CHECKPOINT_DIR/ - Listar checkpoints"
echo ""
