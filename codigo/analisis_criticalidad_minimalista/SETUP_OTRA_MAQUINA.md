# 🖥️ SETUP EN OTRA MÁQUINA - OPCIÓN C (60K IMÁGENES)

**Objetivo**: Ejecutar análisis del Training Set completo (60,000 imágenes) en otra máquina  
**Fecha**: Octubre 20, 2025  
**Commit**: `2f1be83b` - "Agregar Opción C: Training Set Completo (60k imágenes)"

---

## 📋 PREREQUISITOS

### Hardware Requerido
- ✅ GPU con CUDA (recomendado: NVIDIA con ≥8 GB VRAM)
- ✅ CPU: Multi-core (4+ cores recomendado)
- ✅ RAM: Mínimo 16 GB
- ✅ Disco: Mínimo 15 GB libres
  - 3 GB para resultados
  - 5 GB para checkpoints temporales
  - 2 GB para dataset MNIST
  - 5 GB buffer

### Software Requerido
- ✅ Linux (Ubuntu 20.04+ recomendado)
- ✅ Python 3.8+
- ✅ Git
- ✅ CUDA 11.0+ (si tienes GPU NVIDIA)

---

## 🚀 INSTALACIÓN PASO A PASO

### 1. Clonar el Repositorio

```bash
# Clonar desde GitHub
git clone https://github.com/ACRCris/Proyecto-Inv.-teorica..git
cd Proyecto-Inv.-teorica./codigo/analisis_criticalidad_minimalista

# Verificar que estás en el commit correcto
git log --oneline -1
# Debe mostrar: 2f1be83b Agregar Opción C: Training Set Completo (60k imágenes)
```

### 2. Crear Entorno Virtual

```bash
# Crear entorno virtual
python3 -m venv .venv

# Activar entorno
source .venv/bin/activate

# Verificar Python
python --version  # Debe ser 3.8+
```

### 3. Instalar Dependencias

```bash
# Actualizar pip
pip install --upgrade pip

# Instalar PyTorch (con CUDA si tienes GPU)
# Para CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# O para CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# O para CPU solamente (más lento):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Instalar otras dependencias
pip install numpy scipy matplotlib tqdm pandas seaborn

# Opcional: Instalar requirements si existe
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi
```

### 4. Verificar Instalación de GPU (Opcional pero Recomendado)

```bash
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  Running on CPU - Will be slower (~10-20x)")
EOF
```

### 5. Descargar Dataset MNIST

```bash
# El script descargará automáticamente MNIST en la primera ejecución
# Pero puedes pre-descargarlo:
python3 << EOF
from datasets.loader import MNISTLoader
loader = MNISTLoader(root='./data', batch_size=1, img_size=64)
train_loader, test_loader = loader.get_mnist(batch_size=1, train_split=True)
print(f"✅ Training set: {len(train_loader.dataset)} imágenes")
print(f"✅ Test set: {len(test_loader.dataset)} imágenes")
EOF
```

---

## 🏃 EJECUTAR ANÁLISIS

### Opción A: Test Set (10,000 imágenes) - Rápido (~40 min)

```bash
# Dar permisos
chmod +x ver_progreso.sh

# Ejecutar en background
nohup python3 run_kuramoto_full_dataset.py > kuramoto_test.log 2>&1 &
echo $! > kuramoto_test.pid

# Monitorear
./ver_progreso.sh

# O seguir log en tiempo real
tail -f kuramoto_test.log
```

### Opción C: Training Set Completo (60,000 imágenes) - Completo (~3.7 hrs)

```bash
# Dar permisos
chmod +x ver_progreso_TRAIN.sh

# Ejecutar en background
nohup python3 run_kuramoto_TRAIN_full.py > kuramoto_train.log 2>&1 &
echo $! > kuramoto_train.pid

# Monitorear
./ver_progreso_TRAIN.sh

# O seguir log en tiempo real
tail -f kuramoto_train.log
```

---

## 📊 MONITOREO Y VERIFICACIÓN

### Comandos Útiles

```bash
# Ver progreso resumido
./ver_progreso_TRAIN.sh  # Para training set
./ver_progreso.sh        # Para test set

# Verificar que el proceso está corriendo
ps -p $(cat kuramoto_train.pid)

# Ver uso de GPU
watch -n 2 nvidia-smi

# Ver espacio en disco
df -h

# Ver memoria RAM
free -h

# Contar checkpoints guardados
ls resultados_kuramoto_TRAIN_FULL_60k/checkpoints/ | wc -l

# Ver tamaño del directorio de resultados
du -sh resultados_kuramoto_TRAIN_FULL_60k/
```

### Indicadores de Progreso Saludable

✅ **Velocidad esperada**:
- Con GPU: 4-5 img/s
- Con CPU: 0.3-0.5 img/s (mucho más lento)

✅ **Uso de GPU**:
- Utilización: 60-90%
- Memoria: 2-6 GB
- Temperatura: <85°C

✅ **Checkpoints**:
- Frecuencia: Cada 100 imágenes
- Tamaño: ~4-5 MB cada uno
- Total esperado: 600 checkpoints

---

## 🛠️ TROUBLESHOOTING

### Problema 1: Error de CUDA / GPU no detectada

**Síntomas**:
```
RuntimeError: CUDA out of memory
# o
CUDA is not available
```

**Soluciones**:
```bash
# 1. Verificar driver NVIDIA
nvidia-smi

# 2. Reinstalar PyTorch con versión correcta de CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Si persiste, ejecutar en CPU (más lento pero funcional)
# El script detecta automáticamente si CUDA está disponible
```

### Problema 2: Proceso muy lento

**Velocidad < 1 img/s**

**Diagnóstico**:
```bash
# Ver si está usando GPU
nvidia-smi

# Ver procesos que compiten por recursos
top
```

**Soluciones**:
- Cerrar otros procesos que usen GPU
- Verificar que PyTorch está usando GPU:
  ```python
  import torch
  print(torch.cuda.is_available())
  print(torch.cuda.get_device_name(0))
  ```

### Problema 3: Disco lleno

**Síntomas**:
```
OSError: [Errno 28] No space left on device
```

**Soluciones**:
```bash
# Ver espacio disponible
df -h

# Limpiar espacio si es necesario
# (Borrar archivos temporales, comprimir logs, etc.)

# Cambiar directorio de salida (editar script):
# RESULTS_DIR = "/ruta/con/mas/espacio/resultados_TRAIN"
```

### Problema 4: Memoria RAM insuficiente

**Síntomas**:
```
Killed
# o proceso termina sin error claro
```

**Soluciones**:
```bash
# Verificar memoria
free -h

# Cerrar aplicaciones pesadas
# Liberar caché
sudo sync; sudo echo 3 > /proc/sys/vm/drop_caches

# Si persiste, reducir batch interno (editar script)
```

### Problema 5: Proceso se detuvo

**Síntomas**:
```bash
ps -p $(cat kuramoto_train.pid)
# Devuelve: No such process
```

**Solución**:
```bash
# El script tiene checkpoints - simplemente reiniciar
nohup python3 run_kuramoto_TRAIN_full.py > kuramoto_train.log 2>&1 &
echo $! > kuramoto_train.pid

# Automáticamente reanudará desde el último checkpoint
```

---

## 📈 DESPUÉS DEL PROCESAMIENTO

### 1. Verificar Completitud

```bash
python3 << EOF
import torch
data = torch.load('resultados_kuramoto_TRAIN_FULL_60k/metricas_completas_TRAIN_60k.pt')
print(f"Total imágenes procesadas: {len(data['metricas'])}")
print(f"Dataset: {data['dataset']}")
print(f"Timestamp: {data['timestamp_final']}")

# Verificar distribución por clase
from collections import Counter
labels = [m['label'] for m in data['metricas']]
print("\nDistribución por clase:")
for clase, count in sorted(Counter(labels).items()):
    print(f"  Clase {clase}: {count:,} imágenes")
    
# Verificar tasa de éxito
successful = sum(1 for m in data['metricas'] if m.get('success', False))
print(f"\nTasa de éxito: {successful}/{len(data['metricas'])} ({100*successful/len(data['metricas']):.1f}%)")
EOF
```

**Salida esperada**:
```
Total imágenes procesadas: 60000
Dataset: MNIST_TRAIN_60k
Timestamp: 2025-10-20T...

Distribución por clase:
  Clase 0: ~5,923 imágenes
  Clase 1: ~6,742 imágenes
  Clase 2: ~5,958 imágenes
  ...
  (Total debe sumar 60,000)

Tasa de éxito: 60000/60000 (100.0%)
```

### 2. Análisis de Distribuciones

```bash
# CRÍTICO: Verificar normalidad primero
python3 analizar_distribuciones.py --dataset train

# Ver resultados
cat tests_normalidad_TRAIN.csv
cat recomendaciones_agregacion_TRAIN.txt

# Ver gráficas
ls -lh distribucion_*_TRAIN.pdf
```

### 3. Análisis Estadístico por Clase

```bash
# Ejecutar análisis estadístico
python3 analizar_estadisticas_full_dataset.py --dataset train

# Ver resultados
cat resultados_kuramoto_TRAIN_FULL_60k/resumen_ejecutivo.txt
cat resultados_kuramoto_TRAIN_FULL_60k/estadisticas_por_clase_TRAIN.csv
cat resultados_kuramoto_TRAIN_FULL_60k/ranking_criticalidad_TRAIN.csv
```

---

## 📦 TRANSFERIR RESULTADOS

### Opción A: Comprimir y Descargar

```bash
# Comprimir resultados
tar -czf resultados_TRAIN_60k.tar.gz resultados_kuramoto_TRAIN_FULL_60k/

# Ver tamaño
ls -lh resultados_TRAIN_60k.tar.gz

# Descargar con scp (desde tu máquina local)
scp usuario@servidor:/ruta/resultados_TRAIN_60k.tar.gz ./
```

### Opción B: Subir a GitHub (solo resultados clave)

```bash
# NO subir archivos grandes (.pt)
# Solo subir CSVs, PDFs, documentos

cd resultados_kuramoto_TRAIN_FULL_60k/
git add *.csv *.txt *.pdf
git commit -m "Resultados análisis Training Set 60k"
git push
```

### Opción C: Usar Servicio de Almacenamiento

```bash
# Google Drive, Dropbox, etc.
# O servicios académicos como Zenodo, OSF
```

---

## 📚 DOCUMENTACIÓN ADICIONAL

Lee estos archivos para más información:

- **`INSTRUCCIONES_TRAIN_60k.md`** - Guía completa detallada
- **`RESUMEN_OPCION_C.md`** - Resumen ejecutivo del proyecto
- **`README.md`** - Documentación general del módulo
- **`CORRECCION_METODOLOGICA.md`** - Detalles de la metodología corregida

---

## 🎯 CHECKLIST FINAL

Antes de considerar completo:

- [ ] Proceso terminó sin errores
- [ ] 60,000 imágenes procesadas (verificar con script arriba)
- [ ] ~6,000 imágenes por clase
- [ ] Tasa de éxito >99%
- [ ] 600 checkpoints guardados
- [ ] Archivo final ~3 GB
- [ ] Tests de normalidad ejecutados
- [ ] Análisis estadístico completado
- [ ] Visualizaciones generadas
- [ ] Resultados documentados

---

## 🆘 SOPORTE

Si encuentras problemas:

1. **Revisa los logs**:
   ```bash
   tail -100 kuramoto_train.log | grep -i error
   ```

2. **Verifica el último checkpoint**:
   ```bash
   ls -lt resultados_kuramoto_TRAIN_FULL_60k/checkpoints/ | head -5
   ```

3. **Revisa troubleshooting** en `INSTRUCCIONES_TRAIN_60k.md`

4. **Contacta al autor**: Cristian Pérez

---

## 📊 RESUMEN DE TIEMPOS

| Etapa | Tiempo Estimado |
|-------|-----------------|
| Setup inicial | 15-30 min |
| Descarga MNIST | 2-5 min |
| Procesamiento 60k | 3-5 horas |
| Análisis distribuciones | 30-45 min |
| Análisis estadístico | 15-30 min |
| **TOTAL** | **5-7 horas** |

---

**¡Éxito con el análisis en la otra máquina!** 🚀

Si todo sale bien, tendrás resultados con máxima robustez estadística para tu publicación.
