# FASE 2.1 - ANÁLISIS COMPLETO DEL TEST SET DE MNIST

## 📋 Plan de Ejecución

Este plan procesa las **10,000 imágenes completas** del test set de MNIST.

### 🎯 Objetivo
Validar estadísticamente los hallazgos de la Fase 1 con una muestra robusta.

---

## ⚡ INICIO RÁPIDO

### Opción 1: Ejecutar en foreground (ver progreso en vivo)
```bash
cd /home/crperezp/proyectos/ProyectoInvTeorica/Proyecto-Inv.-teorica./codigo/analisis_criticalidad_minimalista
source .venv/bin/activate
python run_kuramoto_full_dataset.py
```

### Opción 2: Ejecutar en background (recomendado para procesos largos)
```bash
cd /home/crperezp/proyectos/ProyectoInvTeorica/Proyecto-Inv.-teorica./codigo/analisis_criticalidad_minimalista
source .venv/bin/activate

# Ejecutar en background con nohup
nohup python run_kuramoto_full_dataset.py > kuramoto_full.log 2>&1 &

# Guardar el PID para poder monitorear
echo $! > kuramoto_full.pid
```

### Opción 3: Monitorear progreso mientras corre en background
```bash
# Terminal 1: Ejecutar procesamiento
nohup python run_kuramoto_full_dataset.py > kuramoto_full.log 2>&1 &

# Terminal 2: Monitorear progreso
python monitor_progreso.py
```

---

## 📊 ESTIMACIONES

- **Total de imágenes**: 10,000
- **Tiempo por imagen**: ~1-2 segundos
- **Tiempo total estimado**: 3-6 horas
- **Checkpoints cada**: 100 imágenes
- **Espacio en disco**: ~500 MB (solo métricas, sin estados completos)

---

## 🔄 CARACTERÍSTICAS DEL SISTEMA

### Sistema de Checkpoints
- Se guarda un checkpoint cada 100 imágenes
- Si el proceso se interrumpe, se puede reanudar automáticamente
- Los checkpoints se guardan en: `resultados_kuramoto_full_dataset/checkpoints/`

### Sistema de Recuperación
- Al reiniciar `run_kuramoto_full_dataset.py`, detecta automáticamente el último checkpoint
- Reanuda desde la última imagen procesada
- No se pierde progreso

---

## 📁 ESTRUCTURA DE RESULTADOS

```
resultados_kuramoto_full_dataset/
├── checkpoints/
│   ├── checkpoint_00100.pt
│   ├── checkpoint_00200.pt
│   └── ...
├── metricas/
│   └── (reservado para futuras extensiones)
├── metricas_completas.pt          # Resultado final con todas las métricas
└── analisis_estadistico/
    ├── estadisticas_por_clase.csv
    ├── tests_significancia.csv
    ├── ranking_criticalidad.csv
    ├── distribuciones_boxplot.pdf
    ├── distribuciones_violin.pdf
    ├── medias_por_clase.pdf
    └── resumen_ejecutivo.txt
```

---

## 🛠️ COMANDOS ÚTILES

### Verificar progreso (desde los logs)
```bash
tail -f kuramoto_full.log
```

### Ver último checkpoint
```bash
ls -lht resultados_kuramoto_full_dataset/checkpoints/ | head
```

### Verificar proceso activo
```bash
# Si guardaste el PID
ps -p $(cat kuramoto_full.pid)

# O buscar el proceso
ps aux | grep run_kuramoto_full_dataset
```

### Detener el proceso
```bash
# Si guardaste el PID
kill $(cat kuramoto_full.pid)

# O buscar y matar
pkill -f run_kuramoto_full_dataset
```

### Reanudar después de interrupción
```bash
# Simplemente vuelve a ejecutar, detectará el checkpoint automáticamente
python run_kuramoto_full_dataset.py
```

---

## 📈 DESPUÉS DEL PROCESAMIENTO

Una vez que `run_kuramoto_full_dataset.py` termine, ejecuta el análisis estadístico:

```bash
python analizar_estadisticas_full_dataset.py
```

Este script generará:
- ✅ Estadísticas descriptivas por clase
- ✅ Tests de significancia (ANOVA)
- ✅ Ranking de candidatos críticos
- ✅ 6 visualizaciones en PDF
- ✅ Resumen ejecutivo

---

## ⚠️ CONSIDERACIONES

### Recursos Necesarios
- **RAM**: ~4 GB disponibles
- **CPU**: Cualquier CPU moderna (no requiere GPU)
- **Disco**: ~500 MB libres
- **Tiempo**: 3-6 horas continuas

### Recomendaciones
1. Ejecutar en horario nocturno si es posible
2. Usar `nohup` para evitar interrupciones por cierre de sesión
3. Monitorear los primeros 100-200 imágenes para verificar que todo funciona
4. Los checkpoints permiten pausar y reanudar en cualquier momento

### Si algo sale mal
- Los checkpoints protegen tu progreso
- Puedes reanudar desde el último checkpoint
- Revisa `kuramoto_full.log` para diagnóstico
- Los errores en imágenes individuales no detienen el proceso completo

---

## 🎯 PRÓXIMOS PASOS (DESPUÉS DE FASE 2.1)

Una vez completado el análisis estadístico, estarás listo para:

- **Fase 2.2**: Construcción de redes funcionales
- **Fase 2.3**: Optimización de parámetros
- **Fase 2.4**: Validación estadística avanzada

---

## 📞 REFERENCIAS

- Script principal: `run_kuramoto_full_dataset.py`
- Script de análisis: `analizar_estadisticas_full_dataset.py`
- Monitor: `monitor_progreso.py`
- Documentación Fase 1: `OBJETIVO_GENERAL_Y_RESULTADOS.md`

---

**Fecha de creación**: Octubre 19, 2025  
**Autor**: Cristian Pérez  
**Fase**: 2.1 - Análisis Completo MNIST
