# 🎓 ANÁLISIS AKOrN PARA MNIST: RESUMEN CONSOLIDADO

## 📍 Ubicación Actual

Estamos integrando el framework AKOrN (ICLR 2025) con tu análisis Kuramoto existente.

**Tu investigación actual**:
- ✅ 60,000 imágenes MNIST analizadas
- ✅ Parámetro crítico α_c calculado por imagen
- ✅ GPU optimizado: 960 imgs/h (6.4x speedup)
- ✅ Bases de datos SQLite completadas

**Nueva línea**: Usar AKOrN para clasificar MNIST y correlacionar con α_c

---

## 🏗️ QUÉ ES AKOrN

**AKOrN = Artificial Kuramoto Oscillatory Neurons**

Un modelo de deep learning que utiliza **dinámica de osciladores de Kuramoto acoplados** como mecanismo fundamental para procesar imágenes.

### Analogía Simple
```
Red neuronal tradicional (CNN):
INPUT → Conv layer → ReLU → Conv layer → ... → OUTPUT

AKOrN:
INPUT → Osciladores localmente acoplados → Evolucionan según Kuramoto
     → Sincronización de fase es la "característica"
     → Patrones en sincronización → Clasificación
     → OUTPUT
```

### Por qué Kuramoto
- Física probada: Describe sistemas de osciladores reales
- Interpretable: R(t) muestra sincronización
- Emergente: Patrones emergen sin programarse explícitamente
- Robusto: Mejor a ataques adversariales que CNNs

---

## 🎯 ARQUITECTURA EN 30 SEGUNDOS

```
Imagen MNIST 28×28
    ↓
Conv inicial → 64 canales
    ↓
[KLayer 1] Osciladores Kuramoto (T=3 timesteps)
    ├─ Detectan: Bordes, texturas locales
    └─ Sincronización: R_0=0.3 → R_1=0.5 → R_2=0.7
    ↓
[Readout] Extrae features
    ↓
[KLayer 2] Osciladores en features L1 (T=3)
    ├─ Detectan: Estructuras de 2o orden
    └─ Sincronización: R_0=0.5 → R_1=0.7 → R_2=0.8
    ↓
[Readout] Abstrae features
    ↓
[KLayer 3] Osciladores globales (T=3)
    ├─ Detectan: ¿Es un 5? ¿Es un 2?
    └─ Sincronización: R_0=0.6 → R_1=0.8 → R_2=0.9
    ↓
Clasificador lineal (256 → 10)
    ↓
Predicción: Clase 0-9 ✓
```

---

## 📊 DOCUMENTACIÓN CREADA

**5 documentos principales:**

| Doc | Tema | Cuándo leerlo |
|-----|------|--------------|
| **INDICE_DOCUMENTACION.md** | Índice de todo | PRIMERO |
| **GUIA_MNIST.md** | Cómo usar (paso-a-paso) | Antes de ejecutar |
| **AKORN_ARQUITECTURA_VISUAL.md** | Cómo funciona (diagramas) | Para entender |
| **ANALISIS_AKORN_MNIST.md** | Análisis técnico (12 secciones) | Para detalles |
| **AKORN_RESUMEN_EJECUTIVO.md** | Overview ejecutivo | Para presentar |
| **INTEGRACION_AKORN_ROADMAP.md** | Plan investigación (3 semanas) | Para diseñar experimentos |

**Ubicación**: Todos en raíz del proyecto

---

## 🚀 INICIO RÁPIDO (30 MINUTOS)

### Paso 1: Setup (5 min)
```bash
cd codigo/akorn
python setup_mnist.py
```
✓ Descarga 70,000 imágenes MNIST

### Paso 2: Test Rápido (10 min)
```bash
python train_classification.py mnist_test \
    --data mnist \
    --epochs 2 \
    --n 2 \
    --L 1 \
    --T 2 \
    --ch 32 \
    --batchsize 256
```
✓ Verifica que funciona

### Paso 3: Visualizar (5 min)
```bash
tensorboard --logdir=runs
# http://localhost:6006
```
✓ Ve gráficas en tiempo real

---

## 🔬 QUÉ SUCEDE EN CADA CAPA

### KLayer (Osciladores Kuramoto)

**Dentro de cada KLayer, para T timesteps:**

```
Timestep 0: Inicialización
├─ x_0 = Conv(entrada)
└─ R_0 ≈ 0.2-0.3 (desorganizado)

Timestep 1: Evolución
├─ delta_theta = atan2(Conv(x_0)) - atan2(x_0)
├─ x_1 = x_0 + gamma * sin(delta_theta)
└─ R_1 ≈ 0.4-0.6 (estructura emerge)

Timestep 2: Convergencia
├─ x_2 = x_1 + gamma * sin(atan2(...))
└─ R_2 ≈ 0.6-0.8 (patrón coherente)

...

Timestep T: Final
└─ x_T = estado final (features sincrónicos)
```

**Parámetro de orden R(t)**:
- Mide sincronización: R=0 (caos), R=1 (orden)
- Indica si patrón es coherente
- Base de la "característica" emergente

---

## 📈 RESULTADOS ESPERADOS

### Accuracy por Epoch (Config Recomendada)
```
Epoch 0:    ~10%  (random)
Epoch 10:   ~85%
Epoch 25:   ~93%
Epoch 50:   ~96%
Epoch 100:  ~98%
```

### Robustez Adversarial
```
Clean:           ~98%
Random Noise:    ~95%
FGSM:            ~85-90%
PGD:             ~70-80%
```
AKOrN más robusto que CNNs tradicionales

---

## 🔗 CONEXIÓN CON TU ANÁLISIS

### Similitudes
```
AMBOS usan: Dinámica de Kuramoto
AMBOS usan: 60,000 imágenes MNIST
AMBOS usan: GPU (MPS en tu Mac)
```

### Diferencias
```
Tu análisis:         AKOrN:
─────────────       ───────────
Objetivo: α_c       Objetivo: Clasificar
Alpha: Manual 201    Alpha: Aprendible conv
T: 100 (convergencia)   T: 3-5 (features)
Output: Escalar      Output: 10 clases
Problema: Criticidad Problema: Features
```

### Oportunidad de Investigación
```
HIPÓTESIS:
Imágenes con α_c alto (criticidad) 
  → Requieren más timesteps en AKOrN
  → Difíciles de clasificar
  → Patrón complejo en sincronización

EXPERIMENTO:
1. Para cada imagen: obtener α_c (de tu BD)
2. Inferir con AKOrN: medir confianza
3. Correlacionar: α_c ↔ confianza
4. Analizar: ¿Hay relación?

POSIBLE RESULTADO:
Correlación = criticidad es concepto fundamental
en aprendizaje automático (no solo en física teórica)
```

---

## 📋 PARÁMETROS CLAVE

### De Kuramoto
| Parámetro | Rango | Efecto | Tu análisis |
|-----------|-------|--------|------------|
| **n** | 1-8 | Capacidad | N/A |
| **T** | 1-10 | Timesteps | T=100 en análisis |
| **gamma** | 0.5-2.0 | Acoplamiento | ≈ α en análisis |
| **alpha** | Aprendible | Conectividad | Barrido manual [0, 0.1] |

### De Entrenamiento
| Parámetro | Default | Para MNIST |
|-----------|---------|-----------|
| epochs | 400 | 50-200 |
| batchsize | 128 | 128-256 |
| lr | 0.0001 | 0.0001-0.0005 |
| T (timesteps) | 3 | 3-5 |

---

## 🎓 CONCEPTOS EXPLICADOS

### ¿Qué es "parámetro de orden" R?
```
R = |mean(e^{i*theta})|  ∈ [0, 1]

R cercano a 1 = osciladores sincronizados
               = patrón visual coherente
               = característica "fuerte"

R cercano a 0 = osciladores desincronizados
               = ruido / fondo
               = característica "débil"

En AKOrN:
- Capa 1: R bajo (bordes aislados)
- Capa 2: R medio (estructura local)
- Capa 3: R alto (forma global)
```

### ¿Qué es "timestep"?
```
Cada timestep evoluciona los osciladores
hacia mayor sincronización (si hay patrón coherente)

Más timesteps = más tiempo para converger
             = mejor representación del patrón
             = pero más lento computacionalmente

Tu análisis: T=100 (espera convergencia completa)
AKOrN: T=3-5 (extrae features emergentes)
```

### ¿Qué es "acoplamiento"?
```
En tu análisis: alpha [0, 0.1] controlado manualmente
En AKOrN: Convoluciones que actúan como acoplamiento
         Aprendibles durante entrenamiento

Convoluciones = "¿Qué información local es relevante?"
Alpha = "¿Cuán fuerte es esa información?"

Durante entrenamiento, la red aprende:
- Qué filtros maximizan sincronización → features relevantes
- Qué alpha es óptima para cada capa → arquitectura emergente
```

---

## 🛠️ SCRIPTS LISTOS

### setup_mnist.py
```bash
python setup_mnist.py
```
→ Descarga 70k imágenes MNIST

### train_classification.py (modificado)
```bash
python train_classification.py mnist_baseline_medium \
    --data mnist \
    --epochs 100 \
    --n 3 \
    --L 2 \
    --T 3 \
    --ch 64
```
→ Entrena AKOrN, ~8 horas, ~98% accuracy

### train_mnist.sh (interactivo)
```bash
bash train_mnist.sh
```
→ Menú interactivo para elegir configuración

---

## 🔄 FLUJO COMPLETO

```
1. SETUP
   ↓ python setup_mnist.py
   ↓ Descargar MNIST (70k imágenes)

2. ENTRENAMIENTO
   ↓ python train_classification.py mnist_baseline_medium ...
   ↓ Train loop: 100 epochs
   ↓ Cada epoch: procesar 60k train, medir 10k test
   ↓ Loss decreases, Accuracy increases
   ↓ Checkpoints guardados cada 25 epochs

3. EVALUACIÓN
   ↓ tensorboard --logdir=runs
   ↓ Ver gráficas en tiempo real
   ↓ Training loss, test accuracy, EMA

4. ANÁLISIS CRUZADO (Investigación)
   ↓ Cargar α_c de tu BD (60k valores)
   ↓ Inferir en MNIST test con AKOrN
   ↓ Medir confianza por imagen
   ↓ Correlacionar α_c ↔ confianza
   ↓ Generar figuras
   ↓ Redactar hallazgos

5. RESULTADOS
   ├─ Accuracy: ~98%
   ├─ Robustez: >85% vs FGSM
   ├─ Correlación: (si existe relación con α_c)
   └─ Insights: Criticidad en ML?
```

---

## ⏱️ ESTIMACIONES DE TIEMPO

| Actividad | Duración |
|-----------|----------|
| Setup (descargar MNIST) | 5 min |
| Test rápido (2 epochs) | 15 min |
| Baseline completo (100 epochs) | 8-15 hrs |
| Análisis correlativo | 2-3 hrs |
| Redacción hallazgos | 2-3 hrs |
| **Total investigación** | **2-3 semanas** |

---

## 🎯 PRÓXIMOS PASOS

### HOY
- [ ] Leer AKORN_RESUMEN_EJECUTIVO.md (30 min)
- [ ] Ejecutar `setup_mnist.py` (5 min)
- [ ] Test rápido (15 min)

### ESTA SEMANA
- [ ] Entrenar baseline (8-15 hrs)
- [ ] Alcanzar ~98% accuracy
- [ ] Visualizar en TensorBoard

### PRÓXIMAS SEMANAS
- [ ] Correlacionar α_c ↔ confianza
- [ ] Variar parámetros (n, L, T)
- [ ] Análisis de sincronización
- [ ] Redactar paper/presentación

---

## 📚 DOCUMENTACIÓN RÁPIDA

**¿Necesito saber...?**

| Pregunta | Documento | Sección |
|----------|-----------|---------|
| Cómo empezar | GUIA_MNIST.md | 1 |
| Qué significa cada parámetro | GUIA_MNIST.md | 3 |
| Cómo entrenar | GUIA_MNIST.md | 4 |
| Por qué falla | GUIA_MNIST.md | 6 |
| Cómo funciona internamente | AKORN_ARQUITECTURA_VISUAL.md | 1-2 |
| Comparación con mi análisis | ANALISIS_AKORN_MNIST.md | 7 |
| Plan de investigación | INTEGRACION_AKORN_ROADMAP.md | 1-4 |
| Resumen ejecutivo | AKORN_RESUMEN_EJECUTIVO.md | Todos |

**Índice maestro**: INDICE_DOCUMENTACION.md

---

## ✅ VALIDACIÓN

**Checklist de implementación**:
- ✅ Documentación técnica completada (5 documentos)
- ✅ Scripts creados (setup_mnist.py, train_mnist.sh)
- ✅ Código modificado (train_classification.py + MNIST support)
- ✅ Ejemplos listos para ejecutar
- ✅ Troubleshooting documentado
- ✅ Roadmap de 3 semanas definido

**Estado**: LISTO PARA EJECUTAR

---

## 🚀 COMENZAR AHORA

```bash
cd /Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/akorn

# 1. Descargar MNIST
python setup_mnist.py

# 2. Test rápido
python train_classification.py mnist_test \
    --data mnist --epochs 2 --n 2 --L 1 --T 2 --ch 32

# 3. Visualizar
tensorboard --logdir=runs
# http://localhost:6006
```

**Duración**: 30 minutos total

Si funciona ✓ → Proceder a entrenamiento baseline

---

## 🎓 Última Nota

Tu investigación es innovadora:
- Usas Kuramoto para **análisis** (¿dónde es el punto crítico?)
- Ahora usarás Kuramoto para **aprendizaje** (¿qué aprende?)

**La conexión**: ¿Es la criticidad (α_c) detectada por redes que usan osciladores?

Esta pregunta podría ser contribución significativa a la intersección física-ML.

---

**Fecha**: 7 de noviembre, 2025  
**Estado**: Documentación completada y lista para ejecución  
**Próximo paso**: Ejecutar setup_mnist.py

¡Adelante con AKOrN! 🚀

