#!/bin/bash

# Script para monitorear el estado de los entrenamientos

LOG_DIR="./runs/c_critico_sweep"
PID_DIR="$LOG_DIR"

echo "=========================================="
echo "MONITOR DE ENTRENAMIENTOS - C_SCALE SWEEP"
echo "=========================================="
echo ""

# Contar procesos activos
echo "📊 Estado de procesos:"
active_count=0
completed_count=0
failed_count=0

for pid_file in "$PID_DIR"/pid_*.txt; do
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            active_count=$((active_count + 1))
        else
            # Verificar si completó exitosamente
            exp_name=$(basename "$pid_file" | sed 's/pid_//' | sed 's/.txt//')
            log_file="$LOG_DIR/logs_${exp_name}.txt"
            if [ -f "$log_file" ]; then
                if grep -q "Evaluating EMA model" "$log_file" 2>/dev/null; then
                    completed_count=$((completed_count + 1))
                else
                    failed_count=$((failed_count + 1))
                fi
            fi
        fi
    fi
done

total=$((active_count + completed_count + failed_count))

echo "  ✅ Activos:     $active_count / 26"
echo "  🏁 Completados: $completed_count / 26"
echo "  ❌ Fallidos:    $failed_count / 26"
echo ""

# Mostrar progreso de cada experimento activo
echo "=========================================="
echo "📈 PROGRESO DETALLADO (Activos)"
echo "=========================================="
echo ""

for pid_file in "$PID_DIR"/pid_*.txt; do
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            exp_name=$(basename "$pid_file" | sed 's/pid_//' | sed 's/.txt//')
            c_val=$(echo "$exp_name" | sed 's/c_scale_//')
            log_file="$LOG_DIR/logs_${exp_name}.txt"
            
            echo "🔄 C_scale = $c_val (PID: $pid)"
            
            if [ -f "$log_file" ]; then
                # Extraer última época
                last_epoch=$(grep "Epoch " "$log_file" | tail -1 | grep -oE "Epoch [0-9]+" | grep -oE "[0-9]+")
                
                # Extraer últimos accuracy
                last_train_acc=$(grep "Train Acc:" "$log_file" | tail -1 | grep -oE "[0-9]+\.[0-9]+%" | head -1)
                last_test_acc=$(grep "Test Acc:" "$log_file" | tail -1 | grep -oE "[0-9]+\.[0-9]+%" | head -1)
                
                if [ -n "$last_epoch" ]; then
                    echo "   Época: $last_epoch/50"
                    if [ -n "$last_train_acc" ] && [ -n "$last_test_acc" ]; then
                        echo "   Train Acc: $last_train_acc | Test Acc: $last_test_acc"
                    fi
                else
                    echo "   Estado: Inicializando..."
                fi
            else
                echo "   Estado: Sin log disponible"
            fi
            echo ""
        fi
    fi
done

# Mostrar resumen de completados
if [ $completed_count -gt 0 ]; then
    echo "=========================================="
    echo "✅ ENTRENAMIENTOS COMPLETADOS"
    echo "=========================================="
    echo ""
    
    for pid_file in "$PID_DIR"/pid_*.txt; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if ! ps -p "$pid" > /dev/null 2>&1; then
                exp_name=$(basename "$pid_file" | sed 's/pid_//' | sed 's/.txt//')
                log_file="$LOG_DIR/logs_${exp_name}.txt"
                
                if [ -f "$log_file" ] && grep -q "Evaluating EMA model" "$log_file" 2>/dev/null; then
                    c_val=$(echo "$exp_name" | sed 's/c_scale_//')
                    
                    # Extraer mejor accuracy
                    best_test=$(grep "Test Acc:" "$log_file" | grep -oE "[0-9]+\.[0-9]+%" | sort -rn | head -1)
                    
                    echo "✓ C_scale = $c_val - Mejor Test Acc: ${best_test:-N/A}"
                fi
            fi
        fi
    done
    echo ""
fi

# Mostrar fallos si los hay
if [ $failed_count -gt 0 ]; then
    echo "=========================================="
    echo "❌ ENTRENAMIENTOS FALLIDOS/INCOMPLETOS"
    echo "=========================================="
    echo ""
    
    for pid_file in "$PID_DIR"/pid_*.txt; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if ! ps -p "$pid" > /dev/null 2>&1; then
                exp_name=$(basename "$pid_file" | sed 's/pid_//' | sed 's/.txt//')
                log_file="$LOG_DIR/logs_${exp_name}.txt"
                
                if [ -f "$log_file" ] && ! grep -q "Evaluating EMA model" "$log_file" 2>/dev/null; then
                    c_val=$(echo "$exp_name" | sed 's/c_scale_//')
                    
                    # Buscar error
                    error_msg=$(grep -i "error\|exception\|traceback" "$log_file" | tail -1)
                    
                    echo "✗ C_scale = $c_val"
                    if [ -n "$error_msg" ]; then
                        echo "  Error: ${error_msg:0:80}..."
                    fi
                    echo ""
                fi
            fi
        fi
    done
fi

echo "=========================================="
echo "COMANDOS ÚTILES"
echo "=========================================="
echo ""
echo "Ver log específico:"
echo "  tail -f $LOG_DIR/logs_c_scale_0.2046.txt"
echo ""
echo "Ver todos los logs activos:"
echo "  tail -f $LOG_DIR/logs_c_scale_*.txt"
echo ""
echo "Detener todos los entrenamientos:"
echo "  kill \$(cat $PID_DIR/pid_*.txt 2>/dev/null)"
echo ""
echo "Re-ejecutar este monitor:"
echo "  watch -n 30 ./monitor_entrenamientos.sh"
echo ""
echo "=========================================="
