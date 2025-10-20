# OPCIÓN C: TRAINING SET COMPLETO (60,000 IMÁGENES)

**Dataset**: MNIST Training Set  
**Tamaño**: 60,000 imágenes (~6,000 por clase)  
**Objetivo**: Máxima robustez estadística para publicación de alto impacto  
**Fecha**: Octubre 20, 2025

---

## 📊 CARACTERÍSTICAS

### Ventajas
✅ **Máxima robustez estadística**
- N ≈ 6,000 muestras por clase
- 6,060,000 observaciones temporales totales
- Intervalo de confianza extremadamente estrecho

✅ **Análisis exhaustivo**
- Detección de subpoblaciones
- Análisis de variabilidad completa
- Curvas de distribución suaves

✅ **Publicación de alto impacto**
- Dataset estándar completo
- Reproducibilidad garantizada
- Resultados definitivos

### Especificaciones Técnicas
- **Imágenes**: 60,000 (vs 10,000 del test set)
- **Pasos temporales**: 101 por imagen
- **Observaciones totales**: 6,060,000
- **Tiempo estimado**: 12-30 horas (dependiendo de GPU)
- **Espacio en disco**: ~3 GB
- **Checkpoints**: 600 archivos (cada 100 imágenes)

---

## 🚀 CÓMO EJECUTAR

### 1. Preparación

**IMPORTANTE**: Espera a que termine el procesamiento del test set actual (10,000 imágenes) antes de iniciar este.

Verifica que el proceso actual haya terminado:
```bash
ps -p $(cat kuramoto_full.pid)
```

Si sigue corriendo, espera o cancélalo si prefieres empezar con el training set:
```bash
kill $(cat kuramoto_full.pid)
```

### 2. Dar permisos de ejecución

```bash
chmod +x ver_progreso_TRAIN.sh
```

### 3. Iniciar procesamiento

**Opción A: En background (RECOMENDADO)**
```bash
nohup python3 run_kuramoto_TRAIN_full.py > kuramoto_train.log 2>&1 &
echo $! > kuramoto_train.pid
```

**Opción B: En una sesión screen**
```bash
screen -S kuramoto_train
python3 run_kuramoto_TRAIN_full.py
# Ctrl+A, D para detach
```

**Opción C: En una sesión tmux**
```bash
tmux new -s kuramoto_train
python3 run_kuramoto_TRAIN_full.py
# Ctrl+B, D para detach
```

### 4. Monitorear progreso

```bash
# Ver resumen de progreso
./ver_progreso_TRAIN.sh

# Seguir log en tiempo real
tail -f kuramoto_train.log

# Ver checkpoints guardados
ls -lh resultados_kuramoto_TRAIN_FULL_60k/checkpoints/

# Verificar proceso
ps aux | grep run_kuramoto_TRAIN
```

---

## 📈 ESTIMACIONES

### Tiempo de Procesamiento

Basado en velocidad observada de ~4.5 img/s:

| Etapa | Imágenes | Tiempo | Completado |
|-------|----------|--------|------------|
| 0-10k | 10,000 | ~37 min | 16.7% |
| 10k-20k | 10,000 | ~37 min | 33.3% |
| 20k-30k | 10,000 | ~37 min | 50.0% |
| 30k-40k | 10,000 | ~37 min | 66.7% |
| 40k-50k | 10,000 | ~37 min | 83.3% |
| 50k-60k | 10,000 | ~37 min | 100% |
| **TOTAL** | **60,000** | **~3.7 hrs** | - |

**Nota**: Este tiempo asume velocidad constante de 4.5 img/s. En la práctica:
- Velocidad mínima esperada: 3.5 img/s → ~4.8 horas
- Velocidad máxima esperada: 5.0 img/s → ~3.3 horas

### Espacio en Disco

| Componente | Tamaño |
|------------|--------|
| Checkpoints (600 archivos) | ~2.5 GB |
| Archivo final | ~3.0 GB |
| Logs | ~50 MB |
| **TOTAL** | **~5.5 GB** |

**Recomendación**: Asegúrate de tener al menos 10 GB libres.

### Checkpoints

- **Frecuencia**: Cada 100 imágenes
- **Total**: 600 checkpoints
- **Primer checkpoint**: checkpoint_00099.pt (100 imágenes)
- **Último checkpoint**: checkpoint_59999.pt (60,000 imágenes)
- **Tamaño promedio**: ~4 MB por checkpoint

---

## 🔍 VERIFICACIÓN DE PROGRESO

### Indicadores de Progreso

1. **Checkpoints guardados**
   ```bash
   ls resultados_kuramoto_TRAIN_FULL_60k/checkpoints/ | wc -l
   ```
   Divide el número entre 600 para ver el % de progreso

2. **Tamaño del directorio**
   ```bash
   du -sh resultados_kuramoto_TRAIN_FULL_60k/
   ```
   Cuando llegue a ~3 GB, está casi completo

3. **Último checkpoint**
   ```bash
   ls -lt resultados_kuramoto_TRAIN_FULL_60k/checkpoints/ | head -2
   ```
   El nombre del archivo indica cuántas imágenes se han procesado

### Estimación de Tiempo Restante

Si tienes N checkpoints:
```
Imágenes procesadas = N × 100
Progreso = (N × 100) / 60000 × 100%
Imágenes restantes = 60000 - (N × 100)
Tiempo restante = Imágenes restantes / 4.5 / 3600 horas
```

Ejemplo: Si tienes 200 checkpoints (20,000 imágenes procesadas):
- Progreso: 33.3%
- Restantes: 40,000 imágenes
- Tiempo: ~2.5 horas

---

## 🛠️ TROUBLESHOOTING

### Problema 1: Proceso se detuvo

**Síntoma**: El proceso no aparece en `ps`

**Solución**: El script tiene sistema de checkpoints, puedes reiniciar:
```bash
nohup python3 run_kuramoto_TRAIN_full.py > kuramoto_train.log 2>&1 &
echo $! > kuramoto_train.pid
```
Automáticamente reanudará desde el último checkpoint.

### Problema 2: Memoria insuficiente

**Síntoma**: Proceso termina con error de memoria

**Solución**: Reduce el batch interno o libera memoria:
```bash
# Liberar caché
sync; echo 3 > /proc/sys/vm/drop_caches

# Verificar memoria disponible
free -h
```

### Problema 3: Disco lleno

**Síntoma**: Error de escritura

**Solución**: Limpia espacio o mueve checkpoints antiguos:
```bash
# Ver espacio disponible
df -h

# Comprimir checkpoints antiguos
cd resultados_kuramoto_TRAIN_FULL_60k/checkpoints
tar -czf checkpoints_parte1.tar.gz checkpoint_000*.pt
rm checkpoint_000*.pt
```

### Problema 4: Proceso muy lento

**Síntoma**: Velocidad < 3 img/s

**Soluciones**:
1. Verificar uso de GPU:
   ```bash
   nvidia-smi
   ```

2. Verificar otros procesos:
   ```bash
   top
   ```

3. Verificar temperatura:
   ```bash
   nvidia-smi --query-gpu=temperature.gpu --format=csv
   ```

---

## 📊 DESPUÉS DEL PROCESAMIENTO

### 1. Verificar Completitud

```bash
# Cargar y verificar
python3 << EOF
import torch
data = torch.load('resultados_kuramoto_TRAIN_FULL_60k/metricas_completas_TRAIN_60k.pt')
print(f"Total imágenes: {len(data['metricas'])}")
print(f"Dataset: {data['dataset']}")
print(f"Timestamp: {data['timestamp_final']}")

# Verificar distribución por clase
from collections import Counter
labels = [m['label'] for m in data['metricas']]
print("\nDistribución por clase:")
for clase, count in sorted(Counter(labels).items()):
    print(f"  Clase {clase}: {count} imágenes")
EOF
```

**Esperado**: ~6,000 imágenes por clase (puede variar levemente)

### 2. Análisis de Distribuciones

```bash
# CRÍTICO: Verificar normalidad primero
python3 analizar_distribuciones.py --dataset train

# Ver resultados
cat tests_normalidad_TRAIN.csv
cat recomendaciones_agregacion_TRAIN.txt
```

### 3. Análisis Estadístico por Clase

```bash
# Ejecutar análisis con método correcto (media o mediana)
python3 analizar_estadisticas_full_dataset.py --dataset train

# Ver resultados
cat resultados_kuramoto_TRAIN_FULL_60k/resumen_ejecutivo.txt
```

### 4. Comparación Test vs Train

```bash
# Comparar resultados de ambos datasets
python3 comparar_test_vs_train.py
```

---

## 📈 RESULTADOS ESPERADOS

### Mejoras respecto al Test Set (10,000 imágenes)

| Métrica | Test Set (N=1000) | Train Set (N=6000) | Mejora |
|---------|-------------------|---------------------|--------|
| Error estándar | ~0.032 | ~0.013 | 2.5× menor |
| Intervalo confianza 95% | ±0.063 | ±0.025 | 2.5× más estrecho |
| Poder estadístico | ~0.80 | >0.99 | Máximo |
| Detección efecto pequeño | Limitada | Excelente | - |

### Análisis Adicionales Posibles

Con 60,000 imágenes puedes hacer:

1. **Análisis de subpoblaciones**
   - Dividir cada clase en terciles
   - Identificar imágenes "fáciles" vs "difíciles"

2. **Validación cruzada robusta**
   - K-fold con K=10 (6,000 imágenes por fold)
   - Bootstrap con muestras grandes

3. **Análisis de outliers**
   - Identificar casos extremos
   - Estudiar qué hace especiales a ciertas imágenes

4. **Modelos predictivos**
   - Entrenar modelos de ML sobre métricas
   - Predecir criticalidad desde propiedades visuales

---

## 🎯 SIGUIENTE PASO DESPUÉS DE COMPLETAR

### Plan Inmediato (próximas 24 horas)

1. ✅ **Completar procesamiento** (12-30 horas)
2. ✅ **Verificar distribuciones** con `analizar_distribuciones.py`
3. ✅ **Análisis estadístico** con `analizar_estadisticas_full_dataset.py`
4. ✅ **Generar visualizaciones** de alta calidad
5. ✅ **Documentar hallazgos** en `RESULTADOS_TRAIN_60k.md`

### Plan a Mediano Plazo (próxima semana)

1. **Comparar Test vs Train**
   - Verificar consistencia de resultados
   - Documentar diferencias si las hay

2. **Análisis de redes funcionales** (Fase 2.2)
   - Construir redes de conectividad
   - Análisis de propiedades de grafo

3. **Optimización de parámetros** (Fase 2.3)
   - Grid search con submuestra
   - Validar parámetros óptimos en set completo

4. **Preparar paper**
   - Figuras de calidad de publicación
   - Tablas de resultados
   - Análisis estadístico riguroso

---

## 📚 ARCHIVOS GENERADOS

### Durante el Procesamiento
```
resultados_kuramoto_TRAIN_FULL_60k/
├── checkpoints/
│   ├── checkpoint_00099.pt
│   ├── checkpoint_00199.pt
│   ├── ...
│   └── checkpoint_59999.pt (600 archivos)
├── metricas_completas_TRAIN_60k.pt (archivo final)
└── (análisis posteriores)
```

### Logs
```
kuramoto_train.log  (log principal)
kuramoto_train.pid  (PID del proceso)
nohup.out          (output adicional)
```

### Después del Análisis
```
resultados_kuramoto_TRAIN_FULL_60k/
├── tests_normalidad_TRAIN.csv
├── recomendaciones_agregacion_TRAIN.txt
├── distribucion_R_TRAIN.pdf
├── distribucion_DFA_TRAIN.pdf
├── estadisticas_por_clase_TRAIN.csv
├── ranking_criticalidad_TRAIN.csv
├── resumen_ejecutivo_TRAIN.txt
└── ... (más visualizaciones)
```

---

## ⚠️ CONSIDERACIONES IMPORTANTES

### 1. No interrumpir manualmente

El proceso tiene checkpoints automáticos. Si lo interrumpes:
- Se perderá progreso desde el último checkpoint
- Sistema recuperará automáticamente al reiniciar

### 2. Verificar espacio en disco

Antes de empezar:
```bash
df -h
```
Necesitas al menos 10 GB libres.

### 3. No ejecutar otros procesos pesados

Para mantener velocidad óptima:
- No entrenar otros modelos durante el procesamiento
- No ejecutar múltiples instancias de este script

### 4. Mantener sesión SSH activa

Si usas SSH:
- Usa `screen` o `tmux` para evitar desconexiones
- O usa `nohup` como se indica arriba

### 5. Monitorear temperatura GPU

```bash
watch -n 5 nvidia-smi
```

Si temperatura > 80°C, considera pausar.

---

## 🎓 VALOR CIENTÍFICO

### Por qué 60,000 imágenes es importante

1. **Standard del campo**
   - Dataset completo = resultados comparables
   - No cherry-picking de datos

2. **Poder estadístico**
   - Detectar efectos pequeños (d < 0.2)
   - Intervalos de confianza estrechos

3. **Reproducibilidad**
   - Protocolo estándar
   - Código + datos públicos

4. **Análisis avanzados**
   - Subpoblaciones
   - Modelos predictivos
   - Meta-análisis

### Para la publicación

Esto te permite afirmar:
- "Analizamos el dataset MNIST completo (60,000 imágenes)"
- "N=6,000 por clase garantiza poder estadístico >99%"
- "Resultados robustos verificados en 6 millones de observaciones temporales"

---

**Última actualización**: Octubre 20, 2025  
**Estado**: Listo para ejecutar  
**Prerequisito**: Completar test set (10,000 imágenes) o ejecutar en paralelo  
**Contacto**: Cristian Pérez
