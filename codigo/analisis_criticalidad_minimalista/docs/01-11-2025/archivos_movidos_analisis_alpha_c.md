# 📂 Archivos Movidos a analisis_alpha_c

**Fecha:** 2 de noviembre de 2025, 01:07 AM

## ✅ Archivos Reubicados

Los siguientes archivos se movieron exitosamente a `codigo/analisis_criticalidad_minimalista/analisis_alpha_c/`:

### 🗄️ Base de Datos
- **`resultados_criticalidad.db`** (192 KB)
  - 1,579 imágenes procesadas (2.6% de 60,000)
  - Periodo: 2025-11-01 06:44 hasta 2025-11-02 01:03
  - Todas las 10 clases representadas

### 📝 Logs
- **`analisis_sqlite.log`** (145 KB)
  - Log completo del análisis
  - Incluye progreso detallado con tqdm

### 🛠️ Scripts de Monitoreo
- **`monitor_analisis.sh`** (ejecutable)
  - Muestra estado del proceso, progreso global, distribución por clase
  - Calcula velocidad y tiempo estimado
  
- **`monitor_continuo.sh`** (ejecutable)
  - Actualización automática cada N segundos (default: 60)
  
- **`iniciar_analisis.sh`** (ejecutable) ⭐ **NUEVO**
  - Script automatizado para iniciar/reiniciar el análisis
  - Detecta procesos en ejecución
  - Muestra estado actual antes de iniciar

### 📚 Archivos Existentes
- **`utils/`** - Módulos compartidos (Fase 1 de refactorización)
- **`MIGRACION.md`** - Guía de migración de scripts
- **`ejemplo_uso_utils.py`** - Ejemplos de uso de utils/
- **`README.md`** - Documentación actualizada con nuevas ubicaciones

---

## 🚀 Cómo Usar (desde codigo/analisis_criticalidad_minimalista/analisis_alpha_c)

### Opción 1: Script Automatizado (Recomendado)
```bash
cd codigo/analisis_criticalidad_minimalista/analisis_alpha_c
./iniciar_analisis.sh
```

### Opción 2: Manual
```bash
cd codigo/analisis_criticalidad_minimalista/analisis_alpha_c
nohup python analizar_con_sqlite.py > analisis_sqlite.log 2>&1 &
```

### Monitoreo
```bash
# Ver estado actual
./monitor_analisis.sh

# Monitoreo continuo (actualiza cada 60 seg)
./monitor_continuo.sh

# Monitoreo continuo cada 30 segundos
./monitor_continuo.sh 30

# Ver log en tiempo real
tail -f analisis_sqlite.log
```

### Consultas Directas a la Base de Datos
```bash
# Progreso total
sqlite3 resultados_criticalidad.db "SELECT COUNT(*), ROUND(COUNT(*)*100.0/60000,2)||'%' FROM resultados;"

# Estadísticas por clase
sqlite3 resultados_criticalidad.db "SELECT clase, COUNT(*), AVG(alpha_c), MIN(alpha_c), MAX(alpha_c) FROM resultados GROUP BY clase;"

# Últimas 10 imágenes procesadas
sqlite3 resultados_criticalidad.db "SELECT * FROM resultados ORDER BY timestamp DESC LIMIT 10;"
```

---

## ⚙️ Estado del Proceso

Al momento de la reubicación:
- ✅ **Proceso detenido** para permitir el movimiento seguro
- ✅ **Base de datos intacta** - verificada post-movimiento
- ✅ **Todos los scripts actualizados** y listos para usar
- 🔄 **Listo para reiniciar** desde la nueva ubicación

**Próximo paso:** Ejecutar `./iniciar_analisis.sh` para continuar el análisis desde 1,579/60,000 imágenes.

---

## 📊 Progreso Actual

| Métrica | Valor |
|---------|-------|
| Imágenes procesadas | 1,579/60,000 (2.6%) |
| Velocidad promedio | ~187.8 imgs/hora (~19.2 seg/img) |
| Tiempo estimado restante | ~13 días |
| Inicio del análisis | 2025-11-01 06:44:42 |
| Última actualización | 2025-11-02 01:03:49 |

### Distribución por Clase
| Clase | Imágenes | α_c Promedio | α_c Min | α_c Max |
|-------|----------|--------------|---------|---------|
| 0 | 150 | 0.00041 | 0.0 | 0.001 |
| 1 | 185 | 0.00130 | 0.0005 | 0.0045 |
| 2 | 145 | 0.00065 | 0.0 | 0.002 |
| 3 | 156 | 0.00068 | 0.0 | 0.002 |
| 4 | 160 | 0.00087 | 0.0 | 0.0025 |
| 5 | 147 | 0.00083 | 0.0 | 0.0025 |
| 6 | 153 | 0.00069 | 0.0 | 0.002 |
| 7 | 169 | 0.00086 | 0.0 | 0.003 |
| 8 | 139 | 0.00062 | 0.0 | 0.002 |
| 9 | 158 | 0.00081 | 0.0 | 0.0025 |

---

## 🔍 Notas Técnicas

- El script `analizar_con_sqlite.py` usa rutas relativas, por lo que funciona correctamente desde su nueva ubicación
- La base de datos SQLite es portable - puede moverse sin problemas mientras no esté en uso
- Los scripts de monitoreo se actualizaron para trabajar desde el directorio `analisis_alpha_c`
- El proceso puede reanudarse en cualquier momento - detecta automáticamente imágenes ya procesadas
