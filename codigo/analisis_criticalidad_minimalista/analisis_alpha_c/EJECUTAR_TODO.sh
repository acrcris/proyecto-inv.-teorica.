#!/bin/bash

# ============================================================================
# SCRIPT MAESTRO: Preparación y Ejecución Completa
# ============================================================================
# Este script:
# 1. Combina las bases de datos parciales en mnist_critical_tot.db
# 2. Mueve la DB combinada a resultados_c_critical/
# 3. Muestra el estado actual
# 4. Lanza el procesamiento de todas las clases faltantes
# ============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         PREPARACIÓN Y PROCESAMIENTO COMPLETO                 ║${NC}"
echo -e "${CYAN}║         Cálculo de C_crítico - TODAS LAS CLASES              ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Activar entorno virtual
echo -e "${YELLOW}🔧 Activando entorno virtual...${NC}"
source /Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/.venv/bin/activate
echo -e "${GREEN}✅ Entorno virtual activado${NC}"
echo ""

# ============================================================================
# PASO 1: Combinar bases de datos parciales
# ============================================================================

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  PASO 1: Combinar bases de datos parciales${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ -f "resultados_c_critical/mnist_critical_tot.db" ]; then
    echo -e "${YELLOW}⚠️  mnist_critical_tot.db ya existe en resultados_c_critical/${NC}"
    echo -e "${YELLOW}   Se usará la base de datos existente.${NC}"
    echo ""
else
    echo -e "${YELLOW}📦 Combinando bases de datos parciales...${NC}"
    
    cd resultados_parciales
    
    if [ ! -f "_ejecutar_proceso.py" ]; then
        echo -e "${RED}❌ Error: No se encontró _ejecutar_proceso.py${NC}"
        exit 1
    fi
    
    python3 _ejecutar_proceso.py
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Bases de datos combinadas exitosamente${NC}"
        echo ""
        
        if [ -f "mnist_critical_tot.db" ]; then
            echo -e "${YELLOW}📦 Moviendo mnist_critical_tot.db a resultados_c_critical/${NC}"
            mv mnist_critical_tot.db ../resultados_c_critical/
            echo -e "${GREEN}✅ Base de datos movida${NC}"
        else
            echo -e "${RED}❌ Error: No se generó mnist_critical_tot.db${NC}"
            exit 1
        fi
    else
        echo -e "${RED}❌ Error al combinar bases de datos${NC}"
        exit 1
    fi
    
    cd ..
fi

echo ""

# ============================================================================
# PASO 2: Verificar estado actual
# ============================================================================

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  PASO 2: Verificar estado actual${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

chmod +x ver_estado_completo.sh
./ver_estado_completo.sh

# ============================================================================
# PASO 3: Confirmar ejecución
# ============================================================================

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  PASO 3: Iniciar procesamiento${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${YELLOW}⚠️  El procesamiento completará TODAS las clases faltantes.${NC}"
echo -e "${YELLOW}   Este proceso puede tomar varias horas.${NC}"
echo ""
echo -e "📋 Características:"
echo -e "   • Reinicio automático si el proceso muere"
echo -e "   • Prevención de suspensión (caffeinate)"
echo -e "   • Logs detallados por cada ejecución"
echo -e "   • Resume desde el último checkpoint"
echo ""

read -p "¿Deseas iniciar el procesamiento ahora? (s/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Ss]$ ]]; then
    echo ""
    echo -e "${GREEN}🚀 Iniciando procesamiento...${NC}"
    echo ""
    
    chmod +x ejecutar_con_reinicio_automatico.sh
    ./ejecutar_con_reinicio_automatico.sh
else
    echo ""
    echo -e "${YELLOW}⏸️  Procesamiento cancelado por el usuario.${NC}"
    echo ""
    echo -e "Para iniciar manualmente más tarde:"
    echo -e "  ${CYAN}./ejecutar_con_reinicio_automatico.sh${NC}"
    echo ""
fi

