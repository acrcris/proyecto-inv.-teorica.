#!/bin/bash
# Script para combinar las bases de datos parciales en mnist_critical_tot.db

cd "$(dirname "$0")"

echo "════════════════════════════════════════════════════════════════"
echo "🔄 PROCESO DE COMBINACIÓN DE BASES DE DATOS"
echo "════════════════════════════════════════════════════════════════"
echo ""

# 1. Combinar bases de datos
echo "📦 Paso 1: Combinando bases de datos parciales..."
echo ""
python3 combinar_bases_datos.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Combinación exitosa"
    echo ""
    
    # 2. Verificar la base de datos combinada
    echo "════════════════════════════════════════════════════════════════"
    echo "🔍 Paso 2: Verificando base de datos combinada..."
    echo ""
    python3 verificar_combinado.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Verificación exitosa"
        echo ""
        
        # 3. Generar gráficas
        echo "════════════════════════════════════════════════════════════════"
        echo "📊 Paso 3: Generando visualizaciones..."
        echo ""
        python3 graficar_distribucion.py
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "════════════════════════════════════════════════════════════════"
            echo "✨ PROCESO COMPLETADO EXITOSAMENTE"
            echo "════════════════════════════════════════════════════════════════"
            echo ""
            echo "📦 Archivos generados:"
            echo "   • mnist_critical_tot.db (base de datos combinada)"
            echo "   • distribucion_c_critico_resumen.png"
            echo "   • distribucion_c_critico_por_clase.png"
            echo ""
            
            # Mostrar tamaño del archivo
            if [ -f "mnist_critical_tot.db" ]; then
                SIZE=$(du -h mnist_critical_tot.db | cut -f1)
                echo "   💾 Tamaño de la base de datos: $SIZE"
            fi
            echo ""
        else
            echo ""
            echo "⚠️  Error al generar gráficas"
        fi
    else
        echo ""
        echo "⚠️  Error al verificar la base de datos"
    fi
else
    echo ""
    echo "❌ Error al combinar las bases de datos"
    exit 1
fi
