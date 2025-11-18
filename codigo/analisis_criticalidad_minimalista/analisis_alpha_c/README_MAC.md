# Cálculo de C_crítico en Mac con Apple Silicon

Este directorio contiene scripts optimizados para calcular el valor crítico de acoplamiento (C_crítico) para imágenes MNIST en **Mac con Apple Silicon (M1/M2/M3)**.

## 🚀 Características

### ✨ Optimizaciones para Metal (MPS)

- **Aceleración GPU nativa**: Usa Metal Performance Shaders (MPS) de Apple
- **Cálculos optimizados**: Evita operaciones complejas que son lentas en MPS
- **Gestión de memoria**: Liberación inteligente de caché GPU
- **Fallback automático**: Si MPS falla, cambia a CPU automáticamente
- **Reproducibilidad**: Semilla fija y estado inicial consistente

### 💾 Persistencia de Datos

- **Base de datos SQLite**: Una tabla por clase (0-9)
- **Auto-restart**: Detecta imágenes ya procesadas y continúa desde donde se quedó
- **Checkpoints**: Guarda cada imagen procesada inmediatamente

## 📋 Requisitos

```bash
# Python 3.8+
# PyTorch con soporte MPS (>= 1.12)
# Mac con Apple Silicon (M1/M2/M3)

# Instalar dependencias
pip install torch torchvision torchaudio
pip install numpy matplotlib tqdm
```

## 🎯 Uso Rápido

### 1. Verificar rendimiento de tu Mac

Antes de comenzar, ejecuta el test de rendimiento para saber si MPS es más rápido que CPU en tu sistema:

```bash
cd codigo/analisis_criticalidad_minimalista/analisis_alpha_c

python test_mps_performance.py
```

**Ejemplo de salida:**
```
⚡ RESULTADOS:
   CPU:  2.3456s
   MPS:  0.8921s
   Speedup: 2.63x

✅ RECOMENDACIÓN: Usa MPS (--device mps)
   MPS es 2.63x más rápido que CPU
```

### 2. Procesamiento de prueba (recomendado)

Prueba con pocas imágenes primero:

```bash
# Activar entorno virtual
source codigo/.venv/bin/activate

# Procesar solo 10 imágenes de clase 0
python calcular_c_critico_local.py --clases 0 --limite 10 --device mps
```

### 3. Procesamiento completo

Una vez verificado que funciona:

```bash
# Procesar una clase completa (~6000 imágenes)
python calcular_c_critico_local.py --clases 0 --device auto

# Procesar múltiples clases
python calcular_c_critico_local.py --clases 0 1 2 --device auto

# Procesar TODAS las clases (60,000 imágenes)
python calcular_c_critico_local.py --all --device auto
```

## 🔧 Opciones del Script

| Opción | Descripción | Ejemplo |
|--------|-------------|---------|
| `--clases` | Clases a procesar (0-9) | `--clases 0 1 2` |
| `--all` | Procesar todas las clases | `--all` |
| `--limite` | Límite por clase (para pruebas) | `--limite 100` |
| `--device` | Dispositivo (`auto`/`mps`/`cpu`) | `--device mps` |
| `--db` | Ruta de la base de datos | `--db mi_db.db` |
| `--output` | Directorio de salida | `--output resultados` |
| `--no-viz` | No generar gráficos | `--no-viz` |

### Recomendaciones de `--device`

- **`auto`** (recomendado): Detecta automáticamente el mejor dispositivo
- **`mps`**: Fuerza uso de Metal GPU (si está disponible)
- **`cpu`**: Fuerza uso de CPU (útil si MPS da problemas)

## 📊 Monitoreo del Progreso

### Ver cuántas imágenes se han procesado

```bash
# Contar imágenes procesadas de clase 0
sqlite3 resultados_c_critical/mnist_c_critical.db \
  "SELECT COUNT(*) FROM clase_0;"

# Ver las últimas 5 imágenes procesadas
sqlite3 resultados_c_critical/mnist_c_critical.db \
  "SELECT image_idx, c_critical, timestamp FROM clase_0 ORDER BY id DESC LIMIT 5;"
```

### Estadísticas en tiempo real

```bash
# Media de C_crítico de clase 0
sqlite3 resultados_c_critical/mnist_c_critical.db \
  "SELECT 
     COUNT(*) as n_images,
     AVG(c_critical) as mean,
     MIN(c_critical) as min,
     MAX(c_critical) as max
   FROM clase_0;"
```

## 🔄 Reinicio Automático

Si el procesamiento se interrumpe (cierre accidental, fallo de energía, etc.), simplemente vuelve a ejecutar el mismo comando:

```bash
# El script detecta automáticamente las imágenes ya procesadas
python calcular_c_critico_local.py --clases 0 1 2
```

**Salida esperada:**
```
PROCESANDO CLASE 0
✅ Imágenes ya procesadas: 2534
⏳ Imágenes pendientes: 3466
Clase 0: 100%|████████| 3466/3466 [1:23:12<00:00, 1.44s/it]
```

## ⚡ Rendimiento Esperado

### En Apple M1 Max (ejemplo)

| Dispositivo | Tiempo/imagen | Clase completa (6000 img) | MNIST completo (60k) |
|-------------|---------------|---------------------------|----------------------|
| **MPS (GPU)** | ~0.9s | ~1.5 horas | ~15 horas |
| **CPU** | ~2.3s | ~3.8 horas | ~38 horas |

### Factores que afectan el rendimiento

- **Modelo de Mac**: M1 < M1 Pro < M1 Max < M2 < M3
- **RAM**: Más RAM = mejor para procesamiento paralelo
- **Temperatura**: El throttling térmico puede reducir velocidad
- **Procesos en segundo plano**: Cerrar apps innecesarias

## 🐛 Solución de Problemas

### Error: "MPS backend out of memory"

```bash
# Solución 1: Usar CPU en lugar de MPS
python calcular_c_critico_local.py --clases 0 --device cpu

# Solución 2: Procesar menos clases a la vez
python calcular_c_critico_local.py --clases 0  # Solo clase 0
```

### Error: "RuntimeError: MPS does not support complex tensors"

Este error **ya está resuelto** en el script. La versión optimizada usa cálculos polares en lugar de números complejos para MPS.

Si aún lo ves:
```bash
# Actualiza PyTorch
pip install --upgrade torch torchvision torchaudio
```

### El procesamiento es muy lento

```bash
# 1. Verifica que estás usando el dispositivo correcto
python test_mps_performance.py

# 2. Verifica que no hay thermal throttling
# Abre Activity Monitor y revisa la temperatura

# 3. Cierra aplicaciones pesadas (Chrome, Photoshop, etc.)
```

### La base de datos está corrupta

```bash
# Verificar integridad
sqlite3 resultados_c_critical/mnist_c_critical.db "PRAGMA integrity_check;"

# Si está corrupta, eliminar y empezar de nuevo
rm resultados_c_critical/mnist_c_critical.db
```

## 📈 Visualizaciones Generadas

El script genera automáticamente:

1. **`distribucion_c_critical_por_clase.png`**
   - Histogramas de C_crítico para cada clase
   - Media marcada con línea roja

2. **`comparacion_medias_clases.png`**
   - Comparación de medias entre clases
   - Barras de error (desviación estándar)

3. **`resumen_c_critical.json`**
   - Estadísticas completas en formato JSON
   - Útil para análisis posterior

## 🔬 Reproducibilidad

El script garantiza **reproducibilidad total**:

- ✅ Semilla fija: `SEED = 1`
- ✅ Estado inicial consistente: mismo `x_init` para todas las imágenes
- ✅ Parámetros guardados: cada registro en la DB incluye los parámetros usados
- ✅ Timestamps: fecha/hora de cada cálculo

**Verificar reproducibilidad:**
```bash
# Ejecutar dos veces con límite pequeño
python calcular_c_critico_local.py --clases 0 --limite 5
rm resultados_c_critical/mnist_c_critical.db  # Borrar DB
python calcular_c_critico_local.py --clases 0 --limite 5

# Los resultados deben ser IDÉNTICOS
```

## 📚 Archivos en este Directorio

| Archivo | Descripción |
|---------|-------------|
| `calcular_c_critico_local.py` | Script principal (optimizado para MPS) |
| `test_mps_performance.py` | Test de rendimiento CPU vs MPS |
| `calculo_alpha_c_mnist_akorn.ipynb` | Notebook para Google Colab |
| `calculo_alpha_c_mnist_autocontenido.ipynb` | Notebook autocontenido |
| `README_MAC.md` | Este archivo |

## 🎓 Ejemplos de Uso Completos

### Ejemplo 1: Prueba rápida (5 minutos)

```bash
source codigo/.venv/bin/activate
cd codigo/analisis_criticalidad_minimalista/analisis_alpha_c

# Test de rendimiento
python test_mps_performance.py

# Procesar 10 imágenes de cada clase
python calcular_c_critico_local.py --clases 0 1 2 --limite 10
```

### Ejemplo 2: Procesamiento nocturno (toda la noche)

```bash
source codigo/.venv/bin/activate
cd codigo/analisis_criticalidad_minimalista/analisis_alpha_c

# Prevenir suspensión del Mac
caffeinate -dims python calcular_c_critico_local.py --all --device mps

# El Mac no se dormirá y procesará las 60,000 imágenes
```

### Ejemplo 3: Procesamiento por lotes (varios días)

```bash
# Día 1: Clases 0-2
python calcular_c_critico_local.py --clases 0 1 2

# Día 2: Clases 3-5
python calcular_c_critico_local.py --clases 3 4 5

# Día 3: Clases 6-9
python calcular_c_critico_local.py --clases 6 7 8 9
```

## 💡 Tips Adicionales

### Maximizar Rendimiento

1. **Conectar el Mac a corriente**: El rendimiento es mejor con cargador
2. **Cerrar aplicaciones pesadas**: Libera RAM y GPU
3. **Evitar recalentamiento**: Usar soporte con ventilación
4. **Modo bajo consumo desactivado**: Preferencias → Batería

### Análisis Posterior

```python
import sqlite3
import pandas as pd

# Cargar datos en pandas
conn = sqlite3.connect('resultados_c_critical/mnist_c_critical.db')
df = pd.read_sql_query("SELECT * FROM clase_0", conn)

# Análisis estadístico
print(df['c_critical'].describe())

# Visualización personalizada
import matplotlib.pyplot as plt
plt.hist(df['c_critical'], bins=50)
plt.show()
```

## 📞 Soporte

Si encuentras problemas:

1. Revisa la sección **Solución de Problemas** arriba
2. Ejecuta `python test_mps_performance.py` para diagnosticar
3. Verifica la versión de PyTorch: `python -c "import torch; print(torch.__version__)"`
4. Consulta la documentación de PyTorch MPS: https://pytorch.org/docs/stable/notes/mps.html

---

**¡Listo para procesar MNIST con la GPU de tu Mac! 🚀**
