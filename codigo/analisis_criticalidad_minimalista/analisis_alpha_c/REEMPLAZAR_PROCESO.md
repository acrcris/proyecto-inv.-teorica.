# Reemplazar Proceso Actual con Versión Optimizada

## 🎯 Situación Actual

- **Base de datos**: `resultados_criticalidad_refactorizado.db`
- **Progreso**: 5,650 / 60,000 imágenes procesadas (9.4%)
- **Versión**: Refactorizada (244 imgs/hora)
- **Tiempo restante**: ~223 horas (~9.3 días)

## 🚀 Estrategia de Reemplazo

En lugar de empezar desde cero, vamos a:

1. ✅ **Mantener la misma base de datos** (no perder las 5,650 imágenes ya procesadas)
2. ✅ **Detener el proceso antiguo** (si está corriendo)
3. ✅ **Iniciar la versión optimizada** que continúe desde la imagen 5,650
4. ✅ **Acelerar el procesamiento** de 244 → 960 imgs/hora

### Ganancia

- **Tiempo restante con versión antigua**: ~9.3 días
- **Tiempo restante con versión optimizada**: ~2.4 días
- **Ahorro de tiempo**: ~7 días ⏱️

## 📝 Instrucciones

### Paso 1: Reemplazar el Proceso

```bash
cd /Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/analisis_criticalidad_minimalista/analisis_alpha_c
./reemplazar_con_optimizado.sh
```

Este script:
- Verifica el progreso actual (5,650 imágenes)
- Detiene el proceso antiguo (si existe)
- Inicia la versión optimizada usando la misma DB
- Muestra el PID del nuevo proceso

### Paso 2: Monitorear el Progreso

**Opción A: Ver log en tiempo real**
```bash
tail -f analisis_optimizado_continuacion.log
```

**Opción B: Monitorear velocidad cada minuto**
```bash
./monitorear_mejora.sh
```

Esto mostrará una tabla actualizada:
```
Tiempo | Procesadas | Nuevas | Velocidad (imgs/h) | Tiempo restante
------------------------------------------------------------------------
  1 min |  5666 |    16 |             960 |     56.6 horas
  2 min |  5698 |    48 |            1440 |     37.7 horas
  3 min |  5714 |    64 |            1280 |     42.4 horas
```

### Paso 3: Verificar Mejora

Después de ~10 minutos, deberías ver:
- **Velocidad estable**: ~900-1000 imgs/hora (vs 244 anterior)
- **Aceleración**: ~4x más rápido
- **Tiempo restante**: ~2.4 días (vs 9.3 días)

## ⚙️ Configuración del Proceso Optimizado

- **Base de datos**: `resultados_criticalidad_refactorizado.db` (la misma)
- **Batch size**: 50 alphas simultáneos
- **Dispositivo**: MPS (Apple Silicon GPU)
- **Commits**: Cada 50 imágenes
- **Log**: `analisis_optimizado_continuacion.log`

## 🛠️ Comandos Útiles

**Ver estado del proceso:**
```bash
ps aux | grep analizar_con_sqlite_OPTIMIZADO
```

**Contar imágenes procesadas:**
```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('resultados_criticalidad_refactorizado.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM resultados')
print(f'Procesadas: {cursor.fetchone()[0]} / 60,000')
conn.close()
"
```

**Detener el proceso (si necesario):**
```bash
# Obtener PID del log o con ps
kill <PID>
```

**Reiniciar si se detuvo:**
```bash
./reemplazar_con_optimizado.sh
```

## 📊 Comparación de Rendimiento

| Métrica | Versión Anterior | Versión Optimizada | Mejora |
|---------|------------------|-------------------|--------|
| Velocidad | 244 imgs/h | 960 imgs/h | **3.9x** |
| Tiempo/img | 14.7 seg | 3.75 seg | **74% más rápido** |
| Tiempo restante | 9.3 días | 2.4 días | **Ahorra 7 días** |
| GPU usage | ~20-30% | ~60-80% | **Mejor utilización** |

## ✅ Ventajas de Esta Estrategia

1. **No pierdes progreso**: Las 5,650 imágenes ya procesadas se mantienen
2. **Transición suave**: Detiene el viejo e inicia el nuevo automáticamente
3. **Misma base de datos**: Resultados compatibles y comparables
4. **Velocidad inmediata**: Empieza a procesar 4x más rápido de inmediato
5. **Monitoreo fácil**: Scripts de monitoreo en tiempo real

## 🚨 Notas Importantes

- El script detecta automáticamente si hay un proceso corriendo y lo detiene
- La versión optimizada salta automáticamente las imágenes ya procesadas
- Los resultados son idénticos, solo cambia la velocidad de procesamiento
- Si hay algún error, simplemente ejecuta `./reemplazar_con_optimizado.sh` de nuevo

## 🎯 Resultado Esperado

Después de ejecutar `./reemplazar_con_optimizado.sh`:

```
✅ Imágenes ya procesadas: 5,650 / 60,000
✅ Proceso anterior detenido (si existía)
✅ Proceso optimizado iniciado (PID: XXXXX)
✅ Velocidad esperada: ~960 imgs/hora
⏱️  Tiempo restante estimado: ~56 horas (2.4 días)
```

---

**¿Listo para reemplazar?** Ejecuta:
```bash
./reemplazar_con_optimizado.sh
```
