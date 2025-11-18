# 📚 Índice de Documentación: AKOrN para MNIST

## 🎯 Objetivo General

Integrar el framework **AKOrN** (Artificial Kuramoto Oscillatory Neurons) con tu **análisis Kuramoto existente** para:
1. Entrenar un clasificador de dígitos MNIST mediante osciladores acoplados
2. Correlacionar parámetro crítico α_c con dificultad de clasificación
3. Investigar si criticidad en dinámica Kuramoto se refleja en aprendizaje

---

## 📄 Documentación Creada

### 1. **ANALISIS_AKORN_MNIST.md** (Ubicación raíz)
**Tipo**: Análisis Técnico Detallado (12 secciones)  
**Contenido**:
- Arquitectura AKOrN y componentes clave
- Parámetros configurables
- Flujo de datos MNIST → Clasificación
- Comparación con tu análisis Kuramoto existente
- Guía para crear dataset MNIST

**Cuándo leer**: Necesitas entender **cómo funciona AKOrN internamente**

---

### 2. **AKORN_RESUMEN_EJECUTIVO.md** (Ubicación raíz)
**Tipo**: Resumen Ejecutivo para Management/Presentación  
**Contenido**:
- Diagrama visual de arquitectura (ASCII art)
- Tabla comparativa con tu análisis
- Parámetros clave explicados
- Configuraciones recomendadas (rápido/balanceado/máximo)
- Troubleshooting rápido

**Cuándo leer**: Necesitas **entender el big picture** rápidamente

---

### 3. **INTEGRACION_AKORN_ROADMAP.md** (Ubicación raíz)
**Tipo**: Plan de Implementación + Roadmap  
**Contenido**:
- Timeline de ejecución (semanas 1-3)
- Experimentos específicos (baseline, T variable, correlación)
- Pseudocódigo para análisis post-entrenamiento
- Checklist de implementación
- Comandos listos para ejecutar

**Cuándo leer**: Necesitas **saber qué hacer paso-a-paso**

---

### 4. **AKORN_ARQUITECTURA_VISUAL.md** (Ubicación raíz)
**Tipo**: Referencia Visual + Fórmulas  
**Contenido**:
- Diagrama completo de capas (ASCII art detallado)
- Tabla de capas y parámetros
- Flujo de datos paso-a-paso (ejemplo dígito "5")
- Fórmulas matemáticas (Kuramoto, orden, conectividad)
- Interpretación física de cada etapa

**Cuándo leer**: Necesitas **visualizar cómo funciona internamente**

---

### 5. **codigo/akorn/GUIA_MNIST.md**
**Tipo**: Guía Práctica Paso-a-Paso  
**Contenido**:
- Inicio rápido (5 minutos)
- Descripción de componentes
- Tablas de parámetros (QUÉ significan)
- 4 ejemplos de uso listos para copiar-pegar
- Troubleshooting detallado
- Estimaciones de tiempo de entrenamiento

**Cuándo leer**: Necesitas **ejecutar AKOrN** (guía del usuario)

---

## 🛠️ Scripts Creados/Modificados

### **Nuevos Scripts**

#### 1. `codigo/akorn/setup_mnist.py`
- **Función**: Descargar y verificar dataset MNIST
- **Uso**: `python setup_mnist.py`
- **Salida**: 60,000 train + 10,000 test imágenes en `data/MNIST/`

#### 2. `codigo/akorn/train_mnist.sh`
- **Función**: Script interactivo para elegir configuración de entrenamiento
- **Uso**: `bash train_mnist.sh`
- **Opciones**: 
  - mnist_baseline_small (5 min)
  - mnist_baseline_medium (8 hrs)
  - mnist_baseline_large (18 hrs)
  - mnist_kuramoto_debug (T=10, 10 hrs)

### **Modificados**

#### 3. `codigo/akorn/train_classification.py`
- **Cambio**: Añadido soporte para dataset MNIST
- **Antes**: Solo soportaba CIFAR10
- **Después**: Soporta CIFAR10 + MNIST
- **Líneas cambiadas**: 175-205 (dataset loading)

---

## 🚀 Cómo Comenzar en 3 Pasos

### Paso 1: Setup (5 minutos)
```bash
cd codigo/akorn
python setup_mnist.py
```
✓ Descarga 70,000 imágenes MNIST

### Paso 2: Entrenar (8-15 horas según configuración)
```bash
python train_classification.py mnist_baseline \
    --data mnist \
    --epochs 100 \
    --n 3 \
    --L 2 \
    --T 3 \
    --ch 64
```
✓ Entrena modelo que alcanza ~98% accuracy

### Paso 3: Visualizar (Inmediato)
```bash
tensorboard --logdir=runs
# http://localhost:6006
```
✓ Ve gráficas en tiempo real

---

## 📊 Relación Entre Documentos

```
TE QUIERO COMENZAR INMEDIATAMENTE
    ↓
Lee: GUIA_MNIST.md (Sección 1: Inicio Rápido)
Luego: Ejecuta setup_mnist.py + entrenamiento
    ↓
¿Quieres ENTENDER cómo funciona?
    ↓
Lee: AKORN_ARQUITECTURA_VISUAL.md
Lee: ANALISIS_AKORN_MNIST.md (Secciones 3-4)
    ↓
¿Quieres PLANIFICAR experimentos?
    ↓
Lee: INTEGRACION_AKORN_ROADMAP.md
    ↓
¿Necesitas MOSTRAR a otros?
    ↓
Lee: AKORN_RESUMEN_EJECUTIVO.md
    ↓
¿Tienes PROBLEMAS?
    ↓
Lee: GUIA_MNIST.md (Sección 6: Troubleshooting)
```

---

## 📈 Contenido por Nivel de Profundidad

### 🟢 Nivel 1: Usuario (Quiero entrenar AKOrN)
- **Documentos**: GUIA_MNIST.md
- **Scripts**: setup_mnist.py, train_mnist.sh
- **Tiempo**: 30 minutos de lectura + 8-15 horas ejecución

### 🟡 Nivel 2: Técnico (Quiero entender la arquitectura)
- **Documentos**: AKORN_ARQUITECTURA_VISUAL.md, AKORN_RESUMEN_EJECUTIVO.md
- **Scripts**: Los anteriores + analizar train_classification.py
- **Tiempo**: 1-2 horas de estudio

### 🔴 Nivel 3: Investigador (Quiero integrar con mi análisis)
- **Documentos**: TODOS
- **Scripts**: Crear nuevos para análisis correlativo
- **Tiempo**: 1-2 semanas proyecto completo

---

## 🔗 Conexiones Entre Documentos

### ANALISIS_AKORN_MNIST.md
↓ Se basa en
- Sección 3 (Modelo AKOrN): Referencia train_classification.py
- Sección 5 (Dinámica Kuramoto): Explica ecuación matemática
- Sección 9 (Flujo completo): Vincula a tu análisis previo

### AKORN_RESUMEN_EJECUTIVO.md
↓ Resume información de
- ANALISIS_AKORN_MNIST.md (Secciones 1-4)
- AKORN_ARQUITECTURA_VISUAL.md (Diagrama simplificado)

### INTEGRACION_AKORN_ROADMAP.md
↓ Se basa en
- ANALISIS_AKORN_MNIST.md (Componentes)
- AKORN_RESUMEN_EJECUTIVO.md (Tabla comparativa)
- Tus resultados previos: 60k imágenes + α_c

### AKORN_ARQUITECTURA_VISUAL.md
↓ Referencia directa
- train_classification.py (estructura real)
- source/models/classification/knet.py (AKOrN class)
- ANALISIS_AKORN_MNIST.md (Secciones 3-4)

---

## 🎓 Conceptos Clave Explicados Dónde

| Concepto | Documento | Sección | Nivel |
|----------|-----------|---------|-------|
| AKOrN overview | AKORN_RESUMEN_EJECUTIVO | 1 | Intro |
| Arquitectura completa | AKORN_ARQUITECTURA_VISUAL | 1 | Visual |
| Dinámica Kuramoto | AKORN_ARQUITECTURA_VISUAL | 8 | Fórmulas |
| Parámetro de orden R | ANALISIS_AKORN_MNIST | 4.2 | Conceptual |
| Cómo entrenar | GUIA_MNIST | 4-5 | Práctico |
| Parámetros qué significan | GUIA_MNIST | 3 | Tabla |
| Comparación con análisis | ANALISIS_AKORN_MNIST | 7 | Investigación |
| Experimentos propuestos | INTEGRACION_AKORN_ROADMAP | 4 | Diseño |
| Troubleshooting | GUIA_MNIST | 6 | Debug |

---

## 💾 Estructura de Archivos

```
proyecto/
│
├── 📄 ANALISIS_AKORN_MNIST.md          ← Análisis técnico
├── 📄 AKORN_RESUMEN_EJECUTIVO.md       ← Para management
├── 📄 INTEGRACION_AKORN_ROADMAP.md     ← Plan 2-3 semanas
├── 📄 AKORN_ARQUITECTURA_VISUAL.md     ← Diagramas + fórmulas
│
└── codigo/akorn/
    ├── 📜 setup_mnist.py               ← Descargar MNIST
    ├── 📜 train_mnist.sh               ← Entrenar interactivo
    ├── 📄 GUIA_MNIST.md                ← Manual usuario
    │
    ├── train_classification.py         ✏️ MODIFICADO
    │   (Líneas 175-205: Añadido MNIST)
    │
    └── source/models/classification/
        └── knet.py                    ← Modelo AKOrN (SIN CAMBIOS)
```

---

## ✅ Validación de Documentación

Cada documento:
- ✓ Es auto-contenido (puedes leerlo solo)
- ✓ Tiene referencias cruzadas a otros
- ✓ Incluye ejemplos ejecutables
- ✓ Explica conceptos en múltiples niveles
- ✓ Tiene tablas y diagramas visuales

---

## 🎯 Próximos Pasos (Recomendado)

**Ahora (30 min)**:
1. Leer AKORN_RESUMEN_EJECUTIVO.md (sección 1-4)
2. Ver AKORN_ARQUITECTURA_VISUAL.md (flujo visual)

**Hoy (2 horas)**:
1. Ejecutar `setup_mnist.py`
2. Ejecutar entrenamiento test (5 epochs)
3. Visualizar en TensorBoard

**Esta semana (8-15 horas)**:
1. Entrenar baseline (100 epochs)
2. Alcanzar ~98% accuracy
3. Guardar checkpoints

**Próximas semanas (investigación)**:
1. Correlacionar α_c con confianza AKOrN
2. Variar parámetros y medir sincronización
3. Redactar hallazgos

---

## 📞 Referencias Rápidas

**¿Dónde está qué?**

| Pregunta | Respuesta |
|----------|-----------|
| ¿Cómo descargo MNIST? | GUIA_MNIST.md sección 1 o setup_mnist.py |
| ¿Qué significa parámetro X? | GUIA_MNIST.md sección 3 |
| ¿Cómo entreno? | GUIA_MNIST.md sección 4, ejemplos listos |
| ¿Por qué falla mi entrenamiento? | GUIA_MNIST.md sección 6 |
| ¿Cómo funciona AKOrN internamente? | AKORN_ARQUITECTURA_VISUAL.md |
| ¿Cuál es el plan de investigación? | INTEGRACION_AKORN_ROADMAP.md |
| ¿Cómo integro con mi α_c? | ANALISIS_AKORN_MNIST.md sección 11 + ROADMAP sección 4.3 |
| ¿Qué debo saber rápido? | AKORN_RESUMEN_EJECUTIVO.md |

---

## 🚀 Stack Tecnológico

**Software**:
- Python 3.12
- PyTorch + MPS (Apple Silicon)
- TensorBoard
- Torchvision (datasets)

**Hardware**:
- Apple Silicon (MPS backend)
- ~8-15 horas para entrenamiento completo

**Datos**:
- MNIST 60,000 train + 10,000 test
- Tu análisis: 60,000 α_c values en SQLite

---

## 📝 Notas Importantes

1. **No requiere GPU externa**: MPS en Mac funciona bien
2. **Dataset automático**: setup_mnist.py descarga todo
3. **Código modificado mínimo**: Solo train_classification.py, 1 sección pequeña
4. **Compatible con tu análisis**: 60k imágenes = 60k α_c values
5. **Extensible**: Otros datasets usando mismo patrón

---

## 🎓 Créditos y Referencias

**AKOrN Paper**:
- Título: "Artificial Kuramoto Oscillatory Neurons"
- Autores: Miyato, Löwe, Geiger, Welling
- Evento: ICLR 2025 (Oral)
- URL: https://arxiv.org/abs/2410.13821

**Tu Investigación**:
- Análisis Kuramoto: 60,000 imágenes MNIST
- Bases de datos: resultados_criticalidad.db
- Optimización GPU: 6.4x speedup implementado

---

**Documentación completa y lista para usar**  
**Última actualización: 7 de noviembre, 2025**

