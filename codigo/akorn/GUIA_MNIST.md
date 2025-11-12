# Guía Ejecutiva: Entrenar AKOrN en MNIST

## Resumen Ejecutivo

AKOrN es un modelo de deep learning basado en osciladores de Kuramoto que puede clasificar dígitos MNIST con alta precisión. Este documento explica cómo entrenar AKOrN en tu dataset MNIST existente (60,000 imágenes).

---

## 1. Inicio Rápido (5 minutos)

### Paso 1: Descargar MNIST
```bash
cd /Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo/akorn
python setup_mnist.py
```

**Salida esperada**:
```
==================================================
CONFIGURANDO MNIST PARA AKORN
==================================================

[1/3] Creando estructura de directorios...
[2/3] Descargando dataset MNIST...
      → Dataset de entrenamiento (60,000 imágenes)
      ✓ Entrenamiento: 60000 imágenes
      → Dataset de validación (10,000 imágenes)
      ✓ Validación: 10000 imágenes

[3/3] Verificando integridad...
      ✓ Batch de prueba cargado correctamente
      
✓ MNIST CONFIGURADO EXITOSAMENTE
==================================================
```

### Paso 2: Entrenar Modelo Simple
```bash
# Entrenamiento rápido (para verificar que funciona)
python train_classification.py mnist_test_quick \
    --data mnist \
    --epochs 5 \
    --n 2 \
    --L 1 \
    --T 2 \
    --ch 32 \
    --lr 0.0001 \
    --batchsize 128
```

**Tiempo estimado**: 3-5 minutos en Apple Silicon

### Paso 3: Visualizar Resultados
```bash
tensorboard --logdir=runs/mnist_test_quick
# Accede a http://localhost:6006 en el navegador
```

---

## 2. Descripción de Componentes

### 2.1 Script Principal: `train_classification.py`
- **Ubicación**: `codigo/akorn/train_classification.py`
- **Función**: Entrena AKOrN en clasificación MNIST
- **Entradas**: Imágenes MNIST 28×28 en escala de grises [0, 1]
- **Salidas**: 
  - Predicciones de clase (0-9)
  - Checkpoints guardados en `runs/mnist_*/`
  - Logs de TensorBoard

### 2.2 Modelo: AKOrN
- **Ubicación**: `source/models/classification/knet.py`
- **Arquitectura**:
  ```
  Imagen 28×28
    ↓ [Conv initial]
    ↓ [KLayer 1: Osciladores Kuramoto]
    ↓ [Readout Block 1]
    ↓ [KLayer 2]
    ↓ [Readout Block 2]
    ↓ [Clasificador lineal]
    ↓ Predicción (10 clases)
  ```

### 2.3 Dataset MNIST
- **Ubicación**: `data/MNIST/`
- **Tamaño**: 70,000 imágenes (60,000 train + 10,000 test)
- **Formato**: Escala de grises 28×28 píxeles
- **Rango**: Valores en [0, 1] (normalizado por ToTensor())

---

## 3. Parámetros Configurables

### 3.1 Parámetros de Modelo (Más Importantes)

| Flag | Descripción | Valor Típico | Efecto |
|------|------------|--------------|--------|
| `--n` | Dimensión de osciladores | 2-4 | Mayor = más capacidad pero más lento |
| `--L` | Número de capas Kuramoto | 1-3 | Mayor = más jerarquía |
| `--T` | Timesteps por capa | 2-5 | Mayor = más sincronización |
| `--ch` | Canales base | 32-128 | Mayor = más features |
| `--J` | Tipo conectividad | "conv" | "conv" recomendado para MNIST |

### 3.2 Parámetros de Entrenamiento

| Flag | Descripción | Default | Ajuste |
|------|------------|---------|--------|
| `--epochs` | Épocas de entrenamiento | 400 | 50-200 para MNIST |
| `--lr` | Learning rate | 0.0001 | 1e-4 típico |
| `--batchsize` | Tamaño de batch | 128 | 128-256 |
| `--beta` | EMA decay | 0.99 | 0.99 standard |

### 3.3 Parámetros de Evaluación

| Flag | Descripción | Default |
|------|------------|---------|
| `--adveval_freq` | Evaluar cada N épocas | 10 |
| `--pgdeval_freq` | PGD attack cada N épocas | 50 |
| `--checkpoint_every` | Guardar checkpoint cada N épocas | 100 |

---

## 4. Ejemplos de Uso

### 4.1 Configuración Mínima (Validación Rápida)
```bash
python train_classification.py mnist_tiny \
    --data mnist \
    --epochs 5 \
    --n 1 \
    --L 1 \
    --T 1 \
    --ch 16 \
    --batchsize 256
```
**Duración**: ~2 minutos  
**Accuracy esperada**: ~50-70% (modelo muy pequeño)

### 4.2 Configuración Recomendada (Balance)
```bash
python train_classification.py mnist_balanced \
    --data mnist \
    --epochs 100 \
    --n 3 \
    --L 2 \
    --T 3 \
    --ch 64 \
    --lr 0.0001 \
    --batchsize 128 \
    --checkpoint_every 25
```
**Duración**: ~1-2 horas  
**Accuracy esperada**: ~97-98%

### 4.3 Configuración Máxima (Mejor Precisión)
```bash
python train_classification.py mnist_large \
    --data mnist \
    --epochs 200 \
    --n 4 \
    --L 3 \
    --T 4 \
    --ch 128 \
    --lr 0.0001 \
    --batchsize 128 \
    --checkpoint_every 50
```
**Duración**: ~4-6 horas  
**Accuracy esperada**: ~98.5-99%

### 4.4 Comparación con Kuramoto (Investigación)
```bash
# Usar T=10 para que los osciladores converjan
# más como en tu análisis Kuramoto (T=100)
python train_classification.py mnist_kuramoto_sync \
    --data mnist \
    --epochs 50 \
    --n 2 \
    --L 2 \
    --T 10 \
    --ch 64 \
    --gamma 1.0 \
    --batchsize 64
```
**Objetivo**: Estudiar sincronización de osciladores  
**Correlación**: Con α_c de tu análisis existente

---

## 5. Interpretación de Salidas

### 5.1 Output en Terminal
```
Epoch: 0 Training loss: 2.302
Epoch: 1 Training loss: 1.856
Epoch: 2 Training loss: 1.234
...
Evaluating original model at epoch 9
Accuracy of the network on the test images: 45.23%
Evaluating EMA model at epoch 9
Accuracy of the network on the test images: 47.81%
```

**Interpretación**:
- **Training loss**: Debe disminuir con épocas
- **Test accuracy**: Debe aumentar gradualmente
- **EMA accuracy**: Generalmente ligeramente mejor (promedio suave)

### 5.2 Logs de TensorBoard

Visualizar con:
```bash
tensorboard --logdir=runs/mnist_balanced
```

**Métricas disponibles**:
- `training loss`: Pérdida durante entrenamiento
- `test accuracy`: Precisión en validación (clean)
- `Random noise test accuracy`: Robustez a ruido
- `FGSM test accuracy`: Robustez adversarial
- `PGD test accuracy`: Robustez a ataque más fuerte

### 5.3 Checkpoints

Se guardan en: `runs/mnist_balanced/checkpoint_epoch_XX.pth`

Cada checkpoint contiene:
```python
{
    'epoch': 25,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'loss': 0.234
}
```

---

## 6. Troubleshooting

### Problema: "Dataset not found"
```bash
# Solución: ejecutar setup_mnist.py primero
python setup_mnist.py
```

### Problema: "CUDA out of memory"
```bash
# Solución: reducir batchsize y canales
python train_classification.py ... --batchsize 64 --ch 32
```

### Problema: Accuracy no mejora
```bash
# Soluciones posibles:
# 1. Aumentar learning rate
python train_classification.py ... --lr 0.0005

# 2. Aumentar número de timesteps
python train_classification.py ... --T 5

# 3. Aumentar dimensión de osciladores
python train_classification.py ... --n 4
```

### Problema: Entrenamiento muy lento
```bash
# Reducir complejidad
python train_classification.py ... \
    --epochs 50 \
    --n 2 \
    --L 1 \
    --T 2 \
    --ch 32
```

---

## 7. Flujo Completo: MNIST → Clasificación

```python
# 1. Cargar imagen MNIST: 28×28 píxel
#    Valor típico: 0.0 a 1.0

# 2. AKOrN procesa:
#    Conv inicial: 1 canal → 64 canales
#    
#    KLayer 1 con T=3 timesteps:
#      t=0: x_0 = conv(input)
#      t=1: x_1 = x_0 + γ·sin(θ(x_0) - θ(C·x_0))
#      t=2: x_2 = x_1 + γ·sin(θ(x_1) - θ(C·x_1))
#      t=3: x_3 = x_2 + γ·sin(θ(x_2) - θ(C·x_2))
#    
#    Parámetro de orden R (sincronización):
#      R = |mean(e^{i·x_t})|
#      R ≈ 1: Osciladores sincronizados
#      R ≈ 0: Osciladores desincronizados
#    
#    Readout: Extrae características
#    KLayer 2, KLayer 3: Similar
#    
# 3. Clasificador final: features → scores [s_0, s_1, ..., s_9]

# 4. Predicción: argmax(scores)
```

---

## 8. Conexión con Tu Análisis Kuramoto Existente

### Similitudes
- **Base**: Ambos usan dinámica de osciladores de Kuramoto
- **Entrada**: 60,000 imágenes MNIST
- **GPU**: Ambos optimizados para Apple Silicon

### Diferencias
| Aspecto | Tu Análisis | AKOrN |
|--------|-----------|-------|
| **Objetivo** | Encontrar α_c (punto crítico) | Clasificar dígito (0-9) |
| **α** | Barrido manual [0.0, 0.1] | Integrado en convoluciones |
| **T** | 100 timesteps | 3-5 por capa |
| **Salida** | Escalar α_c | Vector logits (10 valores) |

### Oportunidad de Investigación
```python
# Correlacionar ambos resultados:
for img_idx in range(60000):
    alpha_c = tu_db[img_idx]  # De tu análisis
    akorn_confidence = akorn_output[img_idx].softmax(0).max()
    
    # ¿Imágenes con α_c alto son más fáciles/difíciles?
    plt.scatter(alpha_c, akorn_confidence)
```

---

## 9. Referencia de Comandos

### Descargar MNIST
```bash
python setup_mnist.py
```

### Entrenar
```bash
python train_classification.py <experiment_name> \
    --data mnist \
    --epochs 100 \
    --n 3 \
    --L 2 \
    --T 3 \
    --ch 64
```

### Visualizar
```bash
tensorboard --logdir=runs/<experiment_name>
```

### Cargar checkpoint (fine-tuning)
```bash
python train_classification.py <new_experiment> \
    --data mnist \
    --finetune runs/<old_experiment>/checkpoint_epoch_50.pth \
    --epochs 50 \
    --lr 0.00005
```

---

## 10. Estimaciones de Tiempo

| Configuración | GPU | CPU Mac | Duración |
|--------------|-----|---------|----------|
| `n=1, L=1, T=1` | ~1 min | ~2 min | por epoch |
| `n=2, L=2, T=2` | ~2 min | ~5 min | por epoch |
| `n=3, L=2, T=3` | ~5 min | ~10 min | por epoch |
| `n=4, L=3, T=4` | ~10 min | ~20 min | por epoch |

**Apple Silicon (MPS)**: Esperado ~5-10 min por epoch para config recomendada

**Total para 100 epochs config recomendada**: ~8-15 horas

---

## Próximos Pasos

1. ✅ Ejecutar `setup_mnist.py` para descargar datos
2. ✅ Entrenar con config mínima (5 epochs) para verificar
3. ✅ Entrenar con config recomendada (100 epochs)
4. ✅ Visualizar resultados en TensorBoard
5. 🔄 Correlacionar con análisis Kuramoto existente
6. 🔄 Experimentar con T=10 para mayor sincronización

---

**Soporte**: Para dudas sobre parámetros específicos, ver `ANALISIS_AKORN_MNIST.md`

