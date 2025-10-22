# Análisis de Distribuciones de α_c en Mac con GPU (MPS)

## 📋 Descripción General

Este documento describe el proceso para ejecutar análisis de distribución de puntos críticos α_c utilizando GPU en Mac (Metal Performance Shaders - MPS) sobre datasets grandes de MNIST.

## 🎯 Objetivo

Calcular curvas R(α) de alta resolución para múltiples imágenes de cada clase del dataset MNIST, detectar puntos críticos α_c mediante `dR/dα`, y analizar la distribución estadística de estos puntos críticos.

## 📊 Metodología

### Parámetros del Modelo Kuramoto

```python
# Parámetros de la red
ch = 4              # Canales
n = 4               # Dimensión del espacio de fases
h, w = 64, 64       # Tamaño de imagen

# Parámetros del modelo
T = 100             # Pasos temporales
gamma = 0.7         # Fuerza de acoplamiento
del_t = 0.9         # Paso de integración
ksize = 7           # Tamaño de ventana convolucional
init_omg = 0.1      # Escala de frecuencias naturales
```

### Cálculo del Parámetro de Orden R

Para cada valor de α:

```python
# Escalar campo de acoplamiento
c_scaled = img_channels.unsqueeze(0) * alpha

# Estado inicial aleatorio
x0 = torch.randn(1, ch, h, w)

# Ejecutar dinámica
x, xs, es = kblock(x0, c_scaled, T=T, gamma=0.7, del_t=0.9, 
                   return_xs=True, return_es=True)

# Calcular R promediando últimos 5 pasos
R_t = []
for x_t in xs[-5:]:
    x_complex = torch.view_as_complex(x_t[0, 0:2].permute(1,2,0).contiguous())
    phases = torch.angle(x_complex)
    R_t.append(np.abs(np.mean(np.exp(1j * phases.detach().numpy()))))

R = np.mean(R_t)
```

### Detección del Punto Crítico α_c

```python
# Calcular derivada numérica
dR = np.gradient(order_params, alphas)

# Punto crítico donde dR/dα es máximo
alpha_c = alphas[np.argmax(dR)]
```

## 🚀 Configuración en Mac con GPU (MPS)

### 1. Requisitos Previos

- **macOS 12.3+** (Monterey o superior)
- **Python 3.8+**
- **PyTorch 2.0+** con soporte MPS
- **Apple Silicon (M1/M2/M3)** o GPU AMD compatible

### 2. Instalación de Dependencias

```bash
# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar PyTorch con soporte MPS
pip install torch torchvision torchaudio

# Verificar que MPS está disponible
python -c "import torch; print(f'MPS disponible: {torch.backends.mps.is_available()}')"

# Otras dependencias
pip install numpy matplotlib scipy tqdm
```

### 3. Verificar Disponibilidad de GPU

```python
import torch

# Verificar MPS
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Usando GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Usando GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("⚠️  Usando CPU")
```

## 📁 Estructura de Archivos

```
analisis_criticalidad_minimalista/
├── kuramoto/
│   ├── __init__.py
│   └── modelo.py              # Modelo KBlock
├── datasets/
│   ├── __init__.py
│   └── loader.py              # MNISTLoader
├── data/
│   └── MNIST/                 # Dataset MNIST
│
├── analizar_distribuciones_alpha_c_gpu.py    # Script principal (GPU)
├── visualizar_resultados_alpha_c.py          # Visualización
├── analisis_distribuciones_alpha_c.md        # Este documento
│
└── resultados/
    ├── distribuciones_alpha_c_[N]imgs.pt     # Datos crudos
    └── plots_distribucion_alpha_c/           # Gráficas
```

## 🔧 Scripts Principales

### Script 1: `analizar_distribuciones_alpha_c_gpu.py`

Script principal para calcular distribuciones en GPU.

**Características:**
- ✅ Soporte automático para MPS/CUDA/CPU
- ✅ Procesamiento en paralelo (batch processing)
- ✅ Guardado incremental para recuperación
- ✅ Barra de progreso con estimación de tiempo
- ✅ Gestión eficiente de memoria GPU

**Uso:**

```bash
# Análisis básico: 10 imágenes por clase
python analizar_distribuciones_alpha_c_gpu.py \
    --images_per_class 10 \
    --alpha_min 0.0 \
    --alpha_max 0.1 \
    --alpha_step 0.0005 \
    --output distribuciones_alpha_c_10imgs.pt

# Análisis amplio: 100 imágenes por clase
python analizar_distribuciones_alpha_c_gpu.py \
    --images_per_class 100 \
    --alpha_min 0.0 \
    --alpha_max 0.1 \
    --alpha_step 0.0005 \
    --output distribuciones_alpha_c_100imgs.pt

# Dataset completo: 1000 imágenes por clase
python analizar_distribuciones_alpha_c_gpu.py \
    --images_per_class 1000 \
    --alpha_min 0.0 \
    --alpha_max 0.1 \
    --alpha_step 0.001 \
    --output distribuciones_alpha_c_1000imgs.pt
```

**Parámetros:**

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--images_per_class` | Imágenes por clase (0-9) | 10 |
| `--alpha_min` | Valor mínimo de α | 0.0 |
| `--alpha_max` | Valor máximo de α | 0.1 |
| `--alpha_step` | Paso de α | 0.0005 |
| `--output` | Archivo de salida | `distribuciones_alpha_c.pt` |
| `--data_root` | Ruta del dataset | `./data` |
| `--device` | Dispositivo (mps/cuda/cpu/auto) | `auto` |
| `--batch_size` | Tamaño de batch (si aplica) | 1 |

### Script 2: `visualizar_resultados_alpha_c.py`

Script para generar visualizaciones completas.

**Uso:**

```bash
python visualizar_resultados_alpha_c.py \
    --input distribuciones_alpha_c_100imgs.pt \
    --output_dir plots_distribucion_alpha_c
```

**Salidas:**
- Histogramas de α_c por clase
- Boxplots comparativos
- Curvas R(α) promedio ± desviación estándar
- Mapas de calor de correlación
- Estadísticas descriptivas en texto

## ⚡ Optimizaciones para GPU

### 1. Gestión de Memoria

```python
# Limpiar caché de GPU periódicamente
if device.type in ['mps', 'cuda']:
    torch.mps.empty_cache() if device.type == 'mps' else torch.cuda.empty_cache()

# Usar torch.no_grad() para ahorrar memoria
with torch.no_grad():
    # Cálculos aquí
    pass
```

### 2. Procesamiento Eficiente

```python
# Mover datos a GPU de forma eficiente
img_channels = img_channels.to(device, non_blocking=True)

# Evitar transferencias CPU-GPU innecesarias
# Hacer todos los cálculos en GPU y transferir solo resultados finales
R = R.cpu().item()  # Solo al final
```

### 3. Precisión Mixta (Opcional)

```python
# Para GPUs con soporte de FP16
with torch.autocast(device_type='cuda', dtype=torch.float16):
    # Cálculos más rápidos
    pass
```

## 📈 Ejecución en Segundo Plano

### En macOS/Linux:

```bash
# Ejecutar en background
nohup python analizar_distribuciones_alpha_c_gpu.py \
    --images_per_class 100 \
    > analisis_gpu.log 2>&1 &

# Ver progreso
tail -f analisis_gpu.log

# Ver procesos
ps aux | grep "analizar_distribuciones"

# Monitorear GPU (Mac)
sudo powermetrics --samplers gpu_power -i 1000
```

### Monitoreo de GPU:

**Mac (MPS):**
```bash
# Actividad GPU general
sudo powermetrics --samplers gpu_power -i 1000 -n 1

# Uso de memoria
vm_stat

# Temperatura
sudo powermetrics --samplers thermal -i 1000 -n 1
```

**NVIDIA (CUDA):**
```bash
# Monitoreo continuo
nvidia-smi -l 1

# Uso específico
watch -n 1 nvidia-smi
```

## 📊 Análisis Esperados

### 1. Por Clase

Para cada clase (0-9):
- **Media de α_c**: Punto crítico promedio
- **Desviación estándar**: Variabilidad dentro de la clase
- **Rango**: [min, max] de α_c
- **Distribución**: Forma (normal, sesgada, bimodal)

### 2. Entre Clases

- **Diferencias significativas**: ¿Qué clases tienen α_c distinto?
- **Correlaciones**: ¿Clases similares tienen α_c similares?
- **Ordenamiento**: Ranking de clases por α_c

### 3. Métricas Globales

- **R(α=0)**: Sincronización sin acoplamiento externo
- **R(α→∞)**: Nivel de saturación
- **dR/dα_max**: Abruptez de la transición
- **Ancho de transición**: Región donde ocurre el cambio

## 🎯 Casos de Uso

### Caso 1: Exploración Rápida (10 imgs/clase)

```bash
# ~2-3 minutos en Mac M1
python analizar_distribuciones_alpha_c_gpu.py \
    --images_per_class 10 \
    --alpha_step 0.001 \
    --output exploracion_rapida.pt
```

**Objetivo**: Validar metodología, verificar rangos de α_c

### Caso 2: Análisis Estadístico (100 imgs/clase)

```bash
# ~20-30 minutos en Mac M1
python analizar_distribuciones_alpha_c_gpu.py \
    --images_per_class 100 \
    --alpha_step 0.0005 \
    --output analisis_estadistico.pt
```

**Objetivo**: Distribuciones confiables, comparaciones entre clases

### Caso 3: Dataset Completo (1000 imgs/clase)

```bash
# ~3-4 horas en Mac M1
python analizar_distribuciones_alpha_c_gpu.py \
    --images_per_class 1000 \
    --alpha_step 0.0005 \
    --output dataset_completo.pt
```

**Objetivo**: Caracterización completa, publicación científica

## 🔍 Troubleshooting

### Problema: MPS no disponible

**Solución:**
```python
# Verificar versión de macOS y PyTorch
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Actualizar PyTorch si es necesario
pip install --upgrade torch torchvision
```

### Problema: Out of Memory (OOM)

**Soluciones:**
1. Reducir `images_per_class`
2. Aumentar `alpha_step` (menos puntos)
3. Procesar en lotes más pequeños
4. Limpiar caché: `torch.mps.empty_cache()`

### Problema: Proceso muy lento

**Diagnóstico:**
```python
import time
import torch

# Benchmark simple
device = torch.device("mps")
x = torch.randn(1000, 1000, device=device)

start = time.time()
for _ in range(100):
    y = x @ x.T
end = time.time()

print(f"Tiempo: {end-start:.3f}s")
print(f"GPU activa: {torch.backends.mps.is_available()}")
```

**Soluciones:**
- Verificar que el código usa `device=mps`
- Revisar transferencias CPU↔GPU innecesarias
- Usar `torch.no_grad()` consistentemente

## 📝 Estructura de Datos de Salida

```python
{
    'alphas': np.array([0.0, 0.0005, 0.001, ...]),  # Valores de α
    'clases': {
        0: {
            'imgs': [
                {
                    'img_idx': 0,
                    'R': np.array([...]),      # Curva R(α)
                    'dR': np.array([...]),     # Derivada
                    'alpha_c': 0.00351,        # Punto crítico
                    'R_at_alpha_c': 0.456,     # R en α_c
                    'dR_max': 357.15,          # dR/dα máximo
                    'R_inicial': 0.0052,       # R(α=0)
                    'R_final': 0.814           # R(α=α_max)
                },
                # ... más imágenes
            ],
            'estadisticas': {
                'alpha_c_mean': 0.00345,
                'alpha_c_std': 0.00023,
                'alpha_c_min': 0.00310,
                'alpha_c_max': 0.00389
            }
        },
        # ... clases 1-9
    },
    'params': {
        'T': 100,
        'gamma': 0.7,
        'del_t': 0.9,
        'ksize': 7,
        'init_omg': 0.1,
        'n_imgs_per_class': 100
    },
    'metadata': {
        'device': 'mps',
        'timestamp': '2025-10-22 09:30:15',
        'duration_seconds': 1834.5
    }
}
```

## 📚 Referencias

### Código Base
- `kuramoto/modelo.py`: Implementación del modelo Kuramoto
- `test_curva_clase1_zoom_0_01.py`: Script de referencia validado
- `kuramoto_pruebas_basico.ipynb`: Notebook con implementación original

### Parámetros Validados
Basados en la implementación del notebook y código de referencia:
- T = 100, gamma = 0.7, del_t = 0.9
- Promedio de últimos 5 pasos temporales para R
- Detección de α_c mediante máximo de dR/dα

## ⏱️ Estimaciones de Tiempo

**Hardware de referencia: Mac M1 (8-core GPU)**

| N imgs/clase | Total imgs | Puntos α | Tiempo estimado |
|--------------|------------|----------|-----------------|
| 10           | 100        | 200      | ~3 min          |
| 50           | 500        | 200      | ~15 min         |
| 100          | 1,000      | 200      | ~30 min         |
| 500          | 5,000      | 200      | ~2.5 hrs        |
| 1000         | 10,000     | 200      | ~5 hrs          |

*Tiempo por simulación: ~3 segundos en GPU MPS*

## 🎓 Siguiente Pasos

1. **Validar en Mac**: Ejecutar caso exploración (10 imgs/clase)
2. **Análisis estadístico**: Ejecutar 100 imgs/clase
3. **Comparar clases**: Analizar diferencias significativas
4. **Publicar resultados**: Generar figuras para paper

## 📧 Contacto y Soporte

Para dudas sobre la implementación, revisar:
- Código validado: `test_curva_clase1_zoom_0_01.py`
- Logs de ejecución: `test_5imgs_clase3.log`
- Notebook original: `kuramoto_pruebas_basico.ipynb`

---

**Última actualización**: 22 de octubre de 2025
**Versión**: 1.0
