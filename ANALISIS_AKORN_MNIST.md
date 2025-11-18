# Análisis AKOrN para Reconocimiento MNIST

## 1. Arquitectura AKOrN: Visión General

**AKOrN** (Artificial Kuramoto Oscillatory Neurons) es un framework de deep learning que utiliza dinámica de osciladores de Kuramoto como mecanismo central para procesamiento de imágenes.

**Publicación**: ICLR 2025 (Oral) - "Artificial Kuramoto Oscillatory Neurons"
**Autores**: Takeru Miyato, Sindy Löwe, Andreas Geiger, Max Welling
**Referencia**: https://arxiv.org/abs/2410.13821

---

## 2. Componentes Clave para Reconocimiento de Dígitos MNIST

### 2.1 Script de Entrenamiento Principal: `train_classification.py`
**Ubicación**: `/codigo/akorn/train_classification.py`

Este es el script diseñado específicamente para tareas de clasificación (10 clases para MNIST).

**Características**:
- ✅ Evaluación de modelos con/sin ataques adversariales (FGSM, PGD, AutoAttack)
- ✅ Entrenamiento con EMA (Exponential Moving Average)
- ✅ Guardado de checkpoints
- ✅ Logging en TensorBoard
- ✅ Soporte multi-dataset

**Flujo de datos**:
```
Data (MNIST) 
  → Augmentación
  → Batch DataLoader
  → AKOrN Model
  → Clasificación (10 salidas)
  → CrossEntropyLoss
  → Backprop → Actualizar pesos
```

---

## 3. Modelo AKOrN: `source/models/classification/knet.py`

### 3.1 Clase Principal: `AKOrN`

```python
class AKOrN(nn.Module):
    def __init__(
        self,
        n=2,              # Dimensión de osciladores
        ch=64,            # Canales base
        out_classes=10,   # Salidas (10 para MNIST)
        L=3,              # Número de capas Kuramoto
        T=3,              # Timesteps por capa
        J="conv",         # Tipo de conectividad
        ksizes=[9, 7, 5], # Tamaños de kernel
        gamma=1.0,        # Parámetro de paso
        # ... más parámetros
    )
```

### 3.2 Arquitectura de Capas

```
Entrada: Imagen MNIST (1×28×28)
    ↓
[Normalización RGB]
    ↓
[Conv inicial: 1→64 canales]
    ↓
[KLayer 1 - Osciladores Kuramoto con T=3 timesteps]
    ├─ Entrada: estado inicial x_0
    ├─ Dinámicas: x_{t+1} = x_t + γ * f_Kuramoto(x_t, C_t)
    └─ Salida: x_final, historiales xs, energías es
    ↓
[Readout Block 1: Extrae características]
    ├─ FF (Feed-Forward layers)
    └─ ResBlock (Conexión residual)
    ↓
[Pooling adaptativo si hay múltiples capas]
    ↓
[KLayer 2, KLayer 3...]
    ↓
[Cabeza de clasificación lineal: 64→10]
    ↓
Salida: Logits (10 clases)
```

### 3.3 Métodos Principales

#### `forward(x, return_xs=False, return_es=False, ensemble=False)`
- **Input**: Imagen MNIST (batch, 1, 28, 28)
- **Output**: Logits de clase (batch, 10)
- **Parámetros opcionales**:
  - `return_xs`: Si es True, retorna historiales de estados
  - `return_es`: Si es True, retorna energías
  - `ensemble`: Si es True, usa ensemble de predicciones

#### `feature(x)`
- Extrae features antes de la cabeza de clasificación
- Útil para visualizar representaciones internas

---

## 4. Dinámica de Osciladores Kuramoto (KLayer)

### 4.1 Qué ocurre en cada KLayer

Para cada timestep `t` (0 a T-1):

1. **Cálculo de Conectividad** (`J`):
   - `J="conv"`: Convoluciones locales (por defecto)
   - `J="attn"`: Atención (no típico para MNIST)
   - Genera matriz de conexiones C_t

2. **Ecuación de Kuramoto**:
   ```
   x_{t+1} = x_t + γ * sin(Θ(x_t) - Θ(C_t @ x_t))
   
   donde:
   - x_t: estado actual
   - γ: tamaño de paso (default 1.0)
   - Θ: función de fase
   - C_t: matriz de conectividad convolucional
   ```

3. **Extracción de Parámetro de Orden**:
   ```
   R_t = |mean(e^{i*x_t})|  ∈ [0, 1]
   
   R ≈ 1: Osciladores sincronizados
   R ≈ 0: Osciladores desincronizados
   ```

### 4.2 Interpretación para MNIST

- **Osciladores sincronizados** (R alto) → Patrón coherente detectado
- **Características** emergen de patrones de sincronización
- **Capas múltiples** → Jerarquía de sincronización
  - Capa 1: Patrones locales (bordes, texturas)
  - Capa 2: Patrones medios (estructuras)
  - Capa 3: Patrones globales (forma de dígito)

---

## 5. Flujo de Datos: MNIST → AKOrN → Predicción

### 5.1 Ejemplo Concreto: Clasificar "5"

```
Entrada: Imagen MNIST 28×28 de dígito "5"
         [valor_píxel ∈ [0, 1]]

↓ [Normalización RGB (no-op para MNIST grayscale)]

↓ [Conv inicial: 1 canal → 64 canales]
  Output: (batch, 64, 28, 28)

↓ [KLayer 1 - L1 Kuramoto (T=3 timesteps)]
  - Timestep 0: 64 osciladores por localidad espacial
  - Timestep 1-2: Evoluciona según dinámicas Kuramoto
  - Output: Features sincronizadas (batch, 64, 28, 28)

↓ [Readout Block 1]
  Output: (batch, 128, 14, 14) [con pooling]

↓ [KLayer 2 - L2 Kuramoto (T=3 timesteps)]
  - Osciladores "ven" features de L1
  - Detectan patrones de segundo orden
  - Output: (batch, 128, 14, 14)

↓ [Readout Block 2]
  Output: (batch, 256, 7, 7)

↓ [KLayer 3 - L3 Kuramoto (T=3 timesteps)]
  - Osciladores "ven" features abstractas
  - Detectan patrones globales (¿es un "5"?)
  - Output: (batch, 256, 7, 7)

↓ [Pooling Adaptativo]
  Output: (batch, 256) [1×1]

↓ [Cabeza Lineal: 256 → 10]
  Logits: [L_0, L_1, ..., L_9]
  donde L_i = score para clase i

↓ [Softmax + Argmax]
  Predicción: clase 5 (si L_5 es máximo)
```

---

## 6. Parámetros Configurables para MNIST

### 6.1 Parámetros de Kuramoto

| Parámetro | Significado | Valor Típico | Efecto |
|-----------|-------------|--------------|--------|
| `n` | Dimensión de osciladores | 2-8 | Mayor n = más capacidad pero más lento |
| `T` | Timesteps por capa | 1-5 | Mayor T = más sincronización pero más lento |
| `gamma` | Tamaño de paso | 0.5-1.5 | Mayor γ = cambios rápidos |
| `L` | Número de capas | 1-4 | Mayor L = más jerarquía |

### 6.2 Parámetros de Arquitectura

| Parámetro | Significado | MNIST Típico |
|-----------|-------------|--------------|
| `ch` | Canales base | 32-64 |
| `ksizes` | Tamaños de kernel | [5, 3, 1] o [9, 7, 5] |
| `J` | Tipo de conectividad | "conv" (recomendado) |
| `out_classes` | Salidas | 10 (fijo para MNIST) |
| `ro_ksize` | Kernel readout | 3 |
| `ro_N` | Dimensión readout | 2 |

### 6.3 Parámetros de Entrenamiento

| Parámetro | Default | Para MNIST |
|-----------|---------|-----------|
| `lr` | 0.0001 | 0.0001-0.0005 |
| `beta` (EMA) | 0.99 | 0.99 |
| `epochs` | 400 | 100-200 |
| `batchsize` | 128 | 128-256 |

---

## 7. Comparación: AKOrN vs Kuramoto Analysis

### 7.1 Similitudes

| Aspecto | AKOrN | Análisis Kuramoto |
|--------|-------|-------------------|
| **Base Teórica** | Dinámica de Kuramoto | Dinámica de Kuramoto |
| **Parámetro Clave** | Osciladores (n) | Parámetro de acoplamiento (α) |
| **Salida Principal** | Predicción de clase | Parámetro de orden R(α) |
| **GPU Acceleration** | Sí (PyTorch) | Sí (MPS) |

### 7.2 Diferencias

| Aspecto | AKOrN | Análisis Kuramoto |
|--------|-------|-------------------|
| **Objetivo** | Clasificación supervisada | Análisis de transiciones críticas |
| **Arquitectura** | Deep network (L capas) | Single dynamic block |
| **Conectividad** | Aprendible (convoluciones) | Fija (α barrido manual) |
| **Timesteps** | T fijo por capa (típico T=3) | T=100 para convergencia |
| **Loss** | CrossEntropy | N/A (análisis descriptivo) |
| **Jerarquía** | Múltiples niveles | Un nivel |

### 7.3 Oportunidad de Integración

```
MNIST Raw Imágenes (60,000)
    ↓
┌─────────────────────────────────────┐
│ RAMA 1: AKOrN Training              │ RAMA 2: Kuramoto Analysis
│ - Entrena clasificador              │ - Analiza α_c por imagen
│ - Genera features                   │ - Genera α_c distribution
│ - Validación: accuracy              │ - Output: 34.8x discrepancia
└─────────────────────────────────────┘
        ↓                       ↓
    Correlación posible:
    ¿Imágenes con α_c alto/bajo
     son más fáciles/difíciles
     para AKOrN clasificar?
```

---

## 8. Setup para Entrenar AKOrN en MNIST

### 8.1 Prerequisitos

```bash
# Ya instalado en tu entorno
cd /Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/akorn
source .venv/bin/activate  # o conda activate akorn
```

### 8.2 Crear Dataset MNIST para AKOrN

**Problema Actual**: `train_classification.py` solo carga CIFAR10
**Solución**: Añadir soporte MNIST

Pasos:

1. **Crear estructura MNIST**:
   ```bash
   mkdir -p data/MNIST/{train,test}
   ```

2. **Descargar MNIST**:
   ```python
   from torchvision import datasets
   datasets.MNIST("./data/MNIST", train=True, download=True)
   datasets.MNIST("./data/MNIST", train=False, download=True)
   ```

3. **Crear/Actualizar dataset loader** en `train_classification.py`

### 8.3 Comando de Entrenamiento (Propuesto)

```bash
# MNIST baseline simple
python train_classification.py \
    mnist_baseline_simple \
    --data mnist \
    --epochs 100 \
    --batchsize 128 \
    --n 2 \
    --L 2 \
    --T 3 \
    --ch 32 \
    --lr 0.0001

# MNIST con más osciladores
python train_classification.py \
    mnist_large_akorn \
    --data mnist \
    --epochs 200 \
    --batchsize 256 \
    --n 4 \
    --L 3 \
    --T 5 \
    --ch 64 \
    --lr 0.0005
```

### 8.4 Salida Esperada

```
Start training...
Epoch: 0 Training loss: 2.302
...
Evaluating original model at epoch 9
Accuracy of the network on the test images: 45.23%
Evaluating EMA model at epoch 9
Accuracy of the network on the test images: 47.81%
...
[Después de 50-100 epochs]
Accuracy: ~98-99% (dependiendo de config)
```

---

## 9. Análisis Detallado del Modelo AKOrN

### 9.1 Clase AKOrN: Métodos Principales

```python
class AKOrN(nn.Module):
    
    def __init__(self, ...):
        # Inicializa capas KLayer, readout blocks, cabeza clasificadora
        
    def _create_layers(self):
        # Construye:
        # - self.layers = nn.ModuleList([KLayer, Readout, KLayer, Readout, ...])
        # - self.head = Linear(last_ch, out_classes)
        
    def feature(self, x):
        # Extrae características internas
        # Output: (x, xs_list, es_list, energy_tensors)
        # Útil para análisis de sincronización
        
    def forward(self, x, return_xs=False, return_es=False, ensemble=False):
        # Pasa imagen a través de todas las capas
        # Output: logits (batch, 10)
        # Si return_xs=True: también retorna historiales de estados
```

### 9.2 KLayer Interno: Cómo Procesa Kuramoto

```python
class KLayer(nn.Module):
    """Capa individual de osciladores Kuramoto"""
    
    def forward(self, x):
        # x: (batch, channels, height, width)
        
        xs = [x]  # Historiales
        es = []   # Energías
        
        for t in range(T):
            # Paso 1: Conectividad (convolución)
            coupling = self.conv(x)  # o atención si J="attn"
            
            # Paso 2: Dinámica Kuramoto
            phase_x = torch.atan2(x.imag, x.real)  # Fase
            phase_c = torch.atan2(coupling.imag, coupling.real)
            delta = phase_c - phase_x
            
            # Paso 3: Actualización
            x_new = x + gamma * torch.sin(delta)
            
            # Paso 4: Registro
            xs.append(x_new)
            R = torch.abs(torch.mean(torch.exp(1j * phase_x)))
            es.append(R)
            
            x = x_new
        
        return x, xs, es
```

### 9.3 Parámetro de Orden Interno (R)

Para cada capa Kuramoto se calcula:

```python
# En cada timestep
phase = torch.atan2(x.imag, x.real)  # fase de cada oscilador
R = torch.abs(torch.mean(torch.exp(1j * phase)))  # parámetro de orden

# R ∈ [0, 1]
# R = 1: Todos los osciladores sincronizados (fase ≈ constante)
# R = 0: Osciladores completamente desincronizados
# R = 0.5: Parcialmente sincronizados
```

---

## 10. Flujo Completo: De MNIST a Clasificación

### 10.1 Pipeline de Datos

```python
# 1. Cargar MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(30)  # Augmentación
])
trainset = MNISTDataset("./data/MNIST", train=True, transform=transform)

# 2. Batch
batch_x, batch_y = next(iter(trainloader))
# batch_x: (128, 1, 28, 28)  [valores en [0, 1]]
# batch_y: (128,)             [etiquetas 0-9]

# 3. AKOrN forward
outputs = model(batch_x)
# outputs: (128, 10)  [logits]

# 4. Loss y backprop
loss = criterion(outputs, batch_y)  # CrossEntropyLoss
loss.backward()
optimizer.step()

# 5. Evaluación
acc = (outputs.argmax(1) == batch_y).float().mean()
```

### 10.2 Visualización Interna (con `return_xs=True`)

```python
x, xs_list, es_list = model.feature(batch_x, return_xs=True, return_es=True)

# xs_list[i] = estado en capa i
#   - xs_list[0]: salida capa 1
#   - xs_list[1]: salida capa 2 (después readout)
#   - xs_list[2]: salida capa 3
#   Cada one es (batch, channels, height, width)

# es_list[i] = parámetro de orden por timestep
#   - es_list[0][t]: R(t) para capa 1, timestep t
#   - es_list[1][t]: R(t) para capa 2, timestep t
#   - es_list[2][t]: R(t) para capa 3, timestep t
```

---

## 11. Relación con tu Análisis Kuramoto Existente

### 11.1 Tabla Comparativa

| Propiedad | Tu Análisis | AKOrN |
|-----------|------------|-------|
| **Entrada** | Imagen MNIST 28×28 | Imagen MNIST 28×28 |
| **Dinámicas** | KBlock, T=100 | KLayer, T=3-5 por capa |
| **Alpha** | Barrido manual [0.0, 0.1], 201 puntos | Aprendible, integrado en convoluciones |
| **Salida** | α_c (punto crítico) | Clasificación 0-9 |
| **GPU** | MPS optimizado, 960 imgs/h | PyTorch estándar |
| **Datos** | 60,000 imágenes analizadas | 60,000 entrenamiento + 10,000 validación |

### 11.2 Posible Conexión

**Hipótesis**:
- Imágenes MNIST con α_c alto pueden ser "patrones complejos" (ej: "2", "5")
- Imágenes con α_c bajo pueden ser "patrones simples" (ej: "1", "0")
- AKOrN puede requerir más timesteps (T más alto) para α_c alto

**Experimento posible**:
```python
# Correlacionar α_c con difficulty en AKOrN
for img_idx in range(60000):
    alpha_c = db.query(img_idx)  # Tu análisis
    confidence = akorn_output[img_idx].max()  # Confianza AKOrN
    
    plt.scatter(alpha_c, confidence)
```

---

## 12. Próximos Pasos Recomendados

### 12.1 Implementación Inmediata

1. ✅ **Entender arquitectura** (este documento)
2. ⏳ **Crear MNIST loader** para `train_classification.py`
3. ⏳ **Entrenar baseline** con configuración simple
4. ⏳ **Validar accuracy** vs CIFAR10 (MNIST debería tener >95%)
5. ⏳ **Extraer features** y correlacionar con α_c

### 12.2 Investigación Avanzada

- **Ablation studies**: Variar n, L, T y medir performance
- **Sincronización analysis**: Usar `return_es=True` para ver R(t) por imagen
- **Integración con análisis**: ¿Correlaciona α_c con confianza?

---

## Referencia Rápida: Archivo Clave

**Para entrenar AKOrN en MNIST**:
- Script: `/codigo/akorn/train_classification.py` 
- Modelo: `/codigo/akorn/source/models/classification/knet.py`
- Dataset: `/codigo/akorn/data/` (crear si no existe)

**Archivos relacionados**:
- Dataset CIFAR10 (referencia): Torchvision builtin
- Augmentación: `source/data/augs.py`
- Ataques adversariales: `source/evals/classification/adv_attacks.py`

