# 🤖 PROMPT PARA COPILOT/LLM EN OTRA MÁQUINA

**Copia y pega este prompt completo en tu asistente de IA en la otra máquina:**

---

## 📋 CONTEXTO DEL PROYECTO

Hola, necesito ayuda para ejecutar un análisis de criticalidad en el dataset MNIST completo usando el modelo Kuramoto. El repositorio ya está preparado con todos los scripts necesarios.

### ⚠️ METODOLOGÍA CRÍTICA: Sin Promedios Prematuros

**IMPORTANTE**: Este proyecto implementa una **corrección metodológica fundamental**:

❌ **INCORRECTO** (versión anterior):
```python
# Calcular promedio de R(t) prematuramente
R_mean = np.mean([R(t=0), R(t=1), ..., R(t=100)])
# Guardar solo 1 valor por imagen
```

✅ **CORRECTO** (versión actual):
```python
# Guardar la serie temporal COMPLETA
R_series = [R(t=0), R(t=1), ..., R(t=100)]  # 101 valores
# Guardar array completo por imagen
# Analizar distribuciones ANTES de decidir si usar media o mediana
```

**Razón**: Necesitamos verificar normalidad de las ~6 millones de observaciones temporales antes de calcular cualquier promedio.

---

## 🎯 OBJETIVO

Ejecutar análisis del **Training Set completo de MNIST (60,000 imágenes)** para obtener máxima robustez estadística:

- 60,000 imágenes (~6,000 por clase)
- 101 pasos temporales por imagen
- 6,060,000 observaciones temporales totales
- Sin promedios prematuros (series completas preservadas)

---

## 📁 ARCHIVOS CLAVE A REVISAR

El repositorio contiene estos archivos preparados:

### 1. **QUICK_START.md** 
Léelo PRIMERO - Comandos rápidos para iniciar

### 2. **SETUP_OTRA_MAQUINA.md**
Guía completa de instalación:
- Prerequisitos de hardware/software
- Instalación de dependencias
- Verificación de GPU
- Troubleshooting detallado

### 3. **run_kuramoto_TRAIN_full.py**
Script principal que:
- Procesa 60,000 imágenes del training set
- Guarda series temporales COMPLETAS (sin promedios)
- Implementa checkpoints cada 100 imágenes
- Recuperación automática desde último checkpoint

### 4. **ver_progreso_TRAIN.sh**
Script de monitoreo que muestra:
- Progreso actual (imágenes procesadas)
- Velocidad de procesamiento
- Tiempo restante estimado
- Estado de checkpoints

### 5. **INSTRUCCIONES_TRAIN_60k.md**
Documentación detallada:
- Estimaciones de tiempo/espacio
- Análisis post-procesamiento
- Valor científico del análisis

### 6. **RESUMEN_OPCION_C.md**
Resumen ejecutivo del proyecto completo

---

## 🚀 PASOS PARA EJECUTAR

### Paso 1: Clonar y Preparar
```bash
git clone https://github.com/ACRCris/Proyecto-Inv.-teorica..git
cd Proyecto-Inv.-teorica./codigo/analisis_criticalidad_minimalista

# Leer documentación
cat QUICK_START.md
```

### Paso 2: Setup del Entorno
```bash
python3 -m venv .venv
source .venv/bin/activate

# Instalar PyTorch con CUDA (ajusta versión según tu GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Otras dependencias
pip install numpy scipy matplotlib tqdm pandas seaborn
```

### Paso 3: Verificar GPU (Crítico para velocidad)
```bash
python3 << EOF
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
EOF
```

### Paso 4: Ejecutar Análisis
```bash
# Dar permisos
chmod +x ver_progreso_TRAIN.sh

# Ejecutar en background (durará ~3-4 horas)
nohup python3 run_kuramoto_TRAIN_full.py > kuramoto_train.log 2>&1 &
echo $! > kuramoto_train.pid

# Monitorear progreso
./ver_progreso_TRAIN.sh
```

### Paso 5: Seguimiento
```bash
# Ver progreso periódicamente
watch -n 60 ./ver_progreso_TRAIN.sh

# O ver log en tiempo real
tail -f kuramoto_train.log

# Verificar GPU
watch -n 5 nvidia-smi
```

---

## 📊 QUÉ ESPERAR

### Durante el Procesamiento:
- **Velocidad esperada**: 4-5 img/s con GPU, 0.3-0.5 img/s con CPU
- **Tiempo total**: ~3.7 horas con GPU (mucho más con CPU)
- **Checkpoints**: Se guardan cada 100 imágenes (600 totales)
- **Espacio**: ~3 GB para resultados finales

### Indicadores de Que Va Bien:
✅ Progreso constante (nuevo checkpoint cada ~20-30 segundos)
✅ GPU utilización 60-90%
✅ Sin errores en el log
✅ Checkpoints creciendo en tamaño progresivamente

### Señales de Problemas:
⚠️ Velocidad <1 img/s con GPU (verificar que usa GPU correctamente)
⚠️ Proceso se detuvo (revisar log, reiniciar - tiene checkpoints)
⚠️ Disco lleno (necesitas 10+ GB libres)

---

## 🔍 VERIFICACIÓN POST-PROCESAMIENTO

Cuando termine (archivo `metricas_completas_TRAIN_60k.pt` creado), verifica:

```python
import torch
data = torch.load('resultados_kuramoto_TRAIN_FULL_60k/metricas_completas_TRAIN_60k.pt')

print(f"Total imágenes: {len(data['metricas'])}")
print(f"Dataset: {data['dataset']}")

# VERIFICAR: Las series están COMPLETAS (sin promedios)
primera_imagen = data['metricas'][0]
print(f"\n¿Serie temporal completa? {len(primera_imagen['R_series'])} valores")
print(f"Esperado: 101 valores (sin promedios)")
assert len(primera_imagen['R_series']) == 101, "ERROR: Series fueron promediadas!"

# Verificar distribución por clase
from collections import Counter
labels = [m['label'] for m in data['metricas']]
for clase, count in sorted(Counter(labels).items()):
    print(f"Clase {clase}: {count:,} imágenes")
```

**Resultado esperado**:
```
Total imágenes: 60000
Dataset: MNIST_TRAIN_60k
¿Serie temporal completa? 101 valores
Esperado: 101 valores (sin promedios)

Clase 0: ~5,923 imágenes
Clase 1: ~6,742 imágenes
...
(Cada clase debe tener ~6,000 imágenes)
```

---

## 🎯 SIGUIENTE PASO: ANÁLISIS SIN PROMEDIOS

⚠️ **CRÍTICO**: Después de procesar, NO calcular promedios directamente.

### Flujo Correcto:

#### 1. Primero: Análisis de Distribuciones
```bash
python3 analizar_distribuciones.py --dataset train
```

Este script:
- Construye DataFrame con TODAS las 6,060,000 observaciones temporales
- Ejecuta tests de normalidad (Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling)
- Genera histogramas y Q-Q plots
- **Determina si usar media o mediana según normalidad**

#### 2. Luego: Análisis Estadístico
```bash
python3 analizar_estadisticas_full_dataset.py --dataset train
```

Este script:
- Usa el método de agregación recomendado (media o mediana)
- Calcula estadísticas robustas por clase
- Ejecuta tests paramétricos o no-paramétricos según distribución
- Genera ranking de criticalidad

---

## ❓ PREGUNTAS PARA EL ASISTENTE

Una vez que hayas leído los archivos mencionados, ayúdame con:

1. ✅ **Verificar prerequisitos**: ¿Tengo todo lo necesario (GPU, espacio, etc.)?

2. ✅ **Instalación correcta**: ¿Están todas las dependencias instaladas correctamente?

3. ✅ **Monitoreo durante ejecución**: ¿Cómo interpretar la salida de `ver_progreso_TRAIN.sh`?

4. ✅ **Troubleshooting**: Si algo falla, ¿qué revisar primero?

5. ✅ **Análisis post-procesamiento**: Una vez terminado, ¿cómo ejecutar el análisis de distribuciones correctamente?

6. ✅ **Interpretación de resultados**: ¿Cómo saber si los resultados son válidos?

---

## 📚 DOCUMENTACIÓN ADICIONAL EN EL REPO

Si necesitas más contexto, estos archivos tienen información adicional:

- **`CORRECCION_METODOLOGICA.md`**: Explica en detalle por qué no se deben calcular promedios prematuramente
- **`RESUMEN_CORRECCION.md`**: Resumen ejecutivo de la corrección
- **`README.md`**: Documentación general del módulo
- **`PLAN_DE_ACCION.md`**: Plan completo del proyecto (fases 1-4)

---

## 🎯 OBJETIVO FINAL

Al completar este análisis tendrás:

✅ **60,000 imágenes procesadas** con series temporales completas
✅ **6,060,000 observaciones** sin promedios prematuros
✅ **Tests de normalidad** ejecutados correctamente
✅ **Análisis estadístico robusto** con método apropiado (media o mediana)
✅ **Resultados con máxima robustez** para publicación científica
✅ **Poder estadístico >99%** (vs ~80% con solo 10k imágenes)

---

## 🆘 SI ALGO FALLA

1. **Revisa el log**: `tail -100 kuramoto_train.log | grep -i error`
2. **Lee troubleshooting**: Sección en `SETUP_OTRA_MAQUINA.md`
3. **Verifi