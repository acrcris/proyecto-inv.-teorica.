# Integración AKOrN + Análisis Kuramoto: Roadmap Completo

## 📋 Vista de 30,000 Pies

Tu investigación tiene dos dimensiones que ahora pueden integrarse:

```
DIMENSIÓN 1: ANÁLISIS KURAMOTO (COMPLETADO)
├─ Input: 60,000 MNIST imágenes
├─ Procesamiento: KBlock, T=100, α=[0, 0.1] barrido manual
├─ Output: α_c por imagen (punto crítico de sincronización)
├─ BD: resultados_criticalidad.db (100% completo)
└─ Pregunta: ¿Dónde ocurre la transición de fase?

DIMENSIÓN 2: CLASIFICACIÓN AKORN (NUEVO)
├─ Input: 60,000 MNIST imágenes (igual dataset)
├─ Procesamiento: Deep network con osciladores Kuramoto entrenables
├─ Output: Predicción de clase (0-9) + confianza
├─ Entrenamiento: 100-200 epochs (~8-15 horas)
└─ Pregunta: ¿Qué patrones de sincronización reconoce?

INTEGRACIÓN POSIBLE:
├─ Correlación: ¿α_c alto ↔ dígitos complejos (5,2,8)?
├─ Dificultad: ¿α_c predice confianza de clasificación?
├─ Timesteps: ¿T más alto necesario para α_c alto?
└─ Física: ¿Criticidad explica emergencia de features?
```

---

## 🗺️ Timeline de Ejecución

### Semana 1: Setup y Validation
```
Día 1-2:
  ├─ python setup_mnist.py          [5 min]
  ├─ Test rápido (5 epochs)         [30 min]
  └─ Verificación en TensorBoard    [10 min]
  
Día 3-4:
  ├─ Entrenar baseline (100 epochs) [12 horas]
  └─ Análisis: ¿Funciona AKOrN?    [1 hora]
  
Día 5-7:
  └─ Experimentos T=10 vs t=3       [12 horas]
```

### Semana 2-3: Investigación Cruzada
```
Día 8-10:
  ├─ Extraer features de AKOrN      [4 horas]
  ├─ Correlación con α_c            [3 horas]
  └─ Generar figuras                [2 horas]
  
Día 11-14:
  ├─ Ablation studies (variar n,L,T) [16 horas]
  ├─ Análisis de sincronización      [4 horas]
  └─ Redacción de hallazgos         [5 horas]
```

---

## 🎯 Flujo de Datos: Integración Completa

```
60,000 MNIST Imágenes
       │
       ├─────────────────────────────────────────────────────┐
       │                                                       │
       ↓                                                       ↓
[TU ANÁLISIS ACTUAL]                              [AKOrN - NUEVO]
│                                                  │
├─ KBlock(T=100, α∈[0,0.1])                      ├─ Conv inicial
├─ Barrido manual de 201 alphas                  ├─ KLayer 1-3
├─ Cálculo de R(α)                               ├─ Readout blocks
├─ Detección α_c                                 ├─ Clasificador
│                                                 │
├─ Output: α_c ∈ [0.001, 0.100]                 ├─ Output: logits
└─→ BD: resultados_criticalidad.db               └─→ Predicción 0-9
   (60,000 registros)                               (accuracy ~98%)
       │                                            │
       └────────────────┬─────────────────────────┘
                        │
                        ↓
        ┌───────────────────────────────┐
        │  ANÁLISIS CORRELATIVO         │
        ├───────────────────────────────┤
        │ for img_idx in range(60000):  │
        │   alpha_c = db[img_idx]       │
        │   akorn_conf = model(img)     │
        │                               │
        │   plot(alpha_c, akorn_conf)   │ → Hallazgo de física
        │   plot(alpha_c, akorn_logits) │
        │   plot(alpha_c, sync_R_layer) │
        └───────────────────────────────┘
                        │
                        ↓
        ┌───────────────────────────────┐
        │  INSIGHTS POTENCIALES         │
        ├───────────────────────────────┤
        │ ✓ α_c ↔ Complejidad dígito   │
        │ ✓ T óptimo ∝ α_c             │
        │ ✓ R(capa3) ↔ confianza AKOrN│
        │ ✓ Teoría emergente detecta   │
        └───────────────────────────────┘
```

---

## 🔬 Experimentos Específicos Propuestos

### Experimento 1: Baseline Accuracy
**Objetivo**: Verificar que AKOrN funciona en MNIST

```bash
python train_classification.py akorn_baseline_mnist \
    --data mnist \
    --epochs 100 \
    --n 3 \
    --L 2 \
    --T 3 \
    --ch 64
```

**Métricas**: Accuracy ≥ 98%  
**Duración**: ~8 horas  
**Resultado esperado**: Validar framework

---

### Experimento 2: Efecto de Timesteps
**Objetivo**: ¿Mayor T mejora sincronización? ¿Como en tu análisis?

```python
for T in [1, 2, 3, 4, 5, 10]:
    # Entrenar modelo con este T
    model = AKOrN(..., T=T)
    accuracy_T[T] = train_and_evaluate(model)
    
    # Extraer parámetro de orden máximo
    x, xs, es = model.feature(test_batch, return_es=True)
    R_max_per_layer[T] = [max(es[layer]) for layer in range(3)]
```

**Esperado**: T↑ → R↑ → Accuracy↑ (hasta convergencia)

---

### Experimento 3: Correlación α_c ↔ Difficulty

**Pseudocódigo**:
```python
# 1. Cargar α_c de tu BD
alpha_c_values = load_from_db()  # 60,000 valores

# 2. Inferir con AKOrN
akorn_predictions = []
akorn_confidences = []

for img_batch in mnist_test:
    logits = akorn_model(img_batch)
    pred = logits.argmax(1)
    conf = logits.softmax(1).max(1)
    akorn_predictions.append(pred)
    akorn_confidences.append(conf)

# 3. Correlacionar
from scipy.stats import spearmanr
corr, pval = spearmanr(alpha_c_values, akorn_confidences)

# 4. Visualizar
plt.figure(figsize=(12, 4))

# Scatter plot
plt.subplot(1, 3, 1)
plt.scatter(alpha_c_values, akorn_confidences, alpha=0.3)
plt.xlabel('α_c (Tu análisis Kuramoto)')
plt.ylabel('Confianza AKOrN')
plt.title(f'Correlación ρ={corr:.3f} (p={pval:.2e})')

# Por clase
plt.subplot(1, 3, 2)
for digit in range(10):
    mask = (akorn_predictions == digit)
    plt.scatter(alpha_c_values[mask], akorn_confidences[mask], 
                label=f'Digit {digit}', alpha=0.5)
plt.xlabel('α_c')
plt.ylabel('Confianza')
plt.legend()

# Histograma 2D
plt.subplot(1, 3, 3)
plt.hist2d(alpha_c_values, akorn_confidences, bins=50)
plt.xlabel('α_c')
plt.ylabel('Confianza')
plt.colorbar()

plt.tight_layout()
plt.savefig('correlacion_alpha_c_vs_akorn.png')
```

**Posibles hallazgos**:
- Correlación positiva: Imágenes con α_c alto → difíciles para AKOrN
- Correlación negativa: Imágenes simples (baja α_c) requieren más Features
- Sin correlación: α_c mide aspecto diferente de dificultad

---

### Experimento 4: Análisis de Sincronización por Capa

**Objetivo**: Ver evolución de R(t) en cada KLayer

```python
def analyze_synchronization(model, test_image):
    """Estudiar parámetro de orden en cada capa"""
    
    x, xs_list, es_list = model.feature(
        test_image.unsqueeze(0),
        return_xs=True,
        return_es=True
    )
    
    # es_list[i] = [R(t) para timestep t de capa i]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for layer_idx in range(3):
        R_by_timestep = es_list[layer_idx]
        axes[layer_idx].plot(R_by_timestep, marker='o')
        axes[layer_idx].set_xlabel('Timestep t')
        axes[layer_idx].set_ylabel('Parámetro de orden R(t)')
        axes[layer_idx].set_title(f'Capa {layer_idx+1}')
        axes[layer_idx].set_ylim([0, 1])
        axes[layer_idx].grid(True)
    
    plt.tight_layout()
    return fig
```

**Interpretación**:
- R(0) ≈ 0: Inicio desincronizado
- R(T) → 1: Converge a sincronización
- Velocidad convergencia: Medida de "coherencia" del patrón

---

## 📊 Tabla Comparativa: Tu Análisis vs AKOrN

| Aspecto | Tu Análisis Kuramoto | AKOrN Clasificador |
|--------|----------------------|-------------------|
| **Dinámicas** | KBlock estático | KLayer dinámico, entrenable |
| **Alpha (α)** | Manual barrido [0, 0.1] | Integrado en convoluciones |
| **T (timesteps)** | Fijo 100 | Configurable 1-10 |
| **Sincronización** | R(α) → α_c | R(t) → features |
| **Objetivo** | Encontrar punto crítico | Clasificar imágenes |
| **Loss** | N/A (análisis) | CrossEntropyLoss |
| **Salida** | Escalar α_c | Vector 10 clases |
| **Física** | ¿Dónde ocurre transición? | ¿Cómo emerge clasificación? |

---

## 💻 Comandos Listos para Ejecutar

### Setup Inicial
```bash
cd /Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/akorn

# Descargar MNIST
python setup_mnist.py

# Activar entorno
source .venv/bin/activate
```

### Experimento 1: Baseline
```bash
python train_classification.py mnist_baseline \
    --data mnist \
    --epochs 100 \
    --n 3 \
    --L 2 \
    --T 3 \
    --ch 64 \
    --lr 0.0001 \
    --checkpoint_every 25
```

### Experimento 2: T variable
```bash
for T in 1 2 3 4 5 10; do
    python train_classification.py mnist_T${T} \
        --data mnist \
        --epochs 50 \
        --T $T \
        --n 2 \
        --L 2 \
        --ch 64
done
```

### Análisis Post-Entrenamiento
```python
# Script análisis_correlacion.py
import torch
import numpy as np
from pathlib import Path
import sqlite3
import matplotlib.pyplot as plt

# Cargar α_c de tu BD
conn = sqlite3.connect('analisis_criticalidad_minimalista/resultados_kuramoto_TRAIN_MAC_60k/resultados_criticalidad.db')
cursor = conn.cursor()
cursor.execute('SELECT imagen_id, alpha_c FROM metricas ORDER BY imagen_id')
alpha_c_dict = {img_id: alpha_c for img_id, alpha_c in cursor.fetchall()}

# Cargar modelo AKOrN entrenado
checkpoint = torch.load('runs/mnist_baseline/checkpoint_epoch_100.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Inferir en MNIST test
from torchvision.datasets import MNIST
from torchvision import transforms

testset = MNIST('./data', train=False, download=False,
                transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=128)

confidences = []
predictions = []
ground_truth = []

for batch_x, batch_y in testloader:
    batch_x = batch_x.to('cuda')
    with torch.no_grad():
        logits = model(batch_x)
    conf = logits.softmax(1).max(1).values.cpu().numpy()
    pred = logits.argmax(1).cpu().numpy()
    
    confidences.extend(conf)
    predictions.extend(pred)
    ground_truth.extend(batch_y.numpy())

# Correlacionar con α_c (de 60k train, mapear índices correctamente)
# ... [resto de código de análisis]

# Generar figura
plt.scatter(alpha_c_values, confidences, alpha=0.3)
plt.xlabel('α_c (Tu análisis)')
plt.ylabel('Confianza AKOrN')
plt.savefig('akorn_alpha_c_correlation.png', dpi=150)
```

---

## 📈 Métricas a Rastrear

### Durante Entrenamiento
- Training loss: Debe disminuir
- Test accuracy: Debe aumentar (~10%/epoch inicialmente)
- EMA accuracy: Generalmente ligeramente mayor

### Post-Entrenamiento
- Accuracy final: Target ≥ 98%
- Adversarial robustness (FGSM, PGD)
- Tiempo de inferencia por imagen

### Análisis Correlativo
- Pearson/Spearman correlation: α_c ↔ confianza
- Mutual information: α_c ↔ corrección/error
- Entropía condicional: P(error | α_c)

---

## 🎓 Teórico: ¿Por qué esto importa?

### Tu Hallazgo Previo
"En el análisis Kuramoto encontraste que cada imagen tiene un α_c donde ocurre una transición de fase en sincronización"

### Pregunta Nuevas
1. **¿Es esta transición de fase detectada por redes neuronales?**
   - AKOrN usa osciladores, debería ser sensible a α_c

2. **¿Predicen imágenes cercanas a α_c más dificultad?**
   - Hipótesis: complejidad en dinámica ↔ complejidad en clasificación

3. **¿Necesita AKOrN más timesteps para α_c alto?**
   - Convergencia en dinámica Kuramoto podría correlacionar

4. **¿Emerge una jerarquía de criticidad?**
   - ¿Ciertas capas "ven" criticidad antes que otras?

### Implicaciones
- Validación de que criticidad es concepto relevante en ML
- Posible ventaja de AKOrN: Sensibilidad natural a puntos críticos
- Conexión entre física teórica y aprendizaje automático

---

## 📝 Documentos de Referencia

En tu proyecto:
- `ANALISIS_AKORN_MNIST.md` → Análisis técnico detallado (12 secciones)
- `AKORN_RESUMEN_EJECUTIVO.md` → Este resumen (versión corta)
- `codigo/akorn/GUIA_MNIST.md` → Guía práctica paso-a-paso
- `codigo/akorn/setup_mnist.py` → Script setup
- `codigo/akorn/train_mnist.sh` → Script interactivo

---

## ✅ Checklist de Implementación

### Fase 1: Validación
- [ ] Ejecutar `python setup_mnist.py`
- [ ] Verificar descarga: ~60k train + 10k test
- [ ] Test rápido: 5 epochs
- [ ] Visualizar en TensorBoard

### Fase 2: Baseline
- [ ] Entrenar 100 epochs, config recomendada
- [ ] Lograr ~98% accuracy
- [ ] Guardar checkpoint mejor modelo

### Fase 3: Investigación
- [ ] Experimento T=[1,2,3,4,5,10]
- [ ] Extraer parámetro de orden R(t)
- [ ] Cargar α_c de tu BD

### Fase 4: Análisis
- [ ] Correlacionar α_c ↔ confianza
- [ ] Generar figura scatter + correlación
- [ ] Test de significancia estadística

### Fase 5: Redacción
- [ ] Resumen de hallazgos
- [ ] Figuras comparativas
- [ ] Conclusiones e implicaciones

---

## 🚀 Próximo Paso Inmediato

```bash
# En terminal:
cd /Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/akorn

# 1. Setup
python setup_mnist.py

# 2. Test rápido (15 min)
python train_classification.py mnist_test \
    --data mnist --epochs 2 --n 2 --L 1 --T 2 --ch 32

# 3. Visualizar
tensorboard --logdir=runs
```

Si funciona ✓ → Proceder a Experimento 1 (Baseline)

---

**Última actualización**: 7 de noviembre, 2025  
**Estado**: Ready for execution  
**Estimado total**: 2-3 semanas para integración completa

