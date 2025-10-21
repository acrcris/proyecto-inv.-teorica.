# 🤖 Contexto de Conversación - Restauración para GitHub Copilot

**Fecha**: Octubre 21, 2025  
**Sesión**: Optimización de Parámetros Kuramoto - Fase Exploratoria  
**Estado**: Listo para ejecutar en Mac M3

---

## 📌 INSTRUCCIONES PARA COPILOT EN MAC

**Cuando abras este archivo en tu Mac, di a Copilot:**

> "Lee CONTEXTO_CONVERSACION.md y restablece el contexto. Estoy en el paso de ejecutar la optimización."

O simplemente:

> "Restaurar contexto desde CONTEXTO_CONVERSACION.md"

---

## 🎯 ESTADO ACTUAL DEL PROYECTO

### Problema Principal
El modelo de Kuramoto **NO está en estado crítico**. Las métricas temporales están desalineadas con la teoría.

### Métricas Actuales vs Objetivos

| Métrica | Valor Actual | Objetivo Teórico | Desviación | Fuente Teórica |
|---------|--------------|------------------|------------|----------------|
| **R_final** | 0.54-0.69 | 0.50 | ✅ Cercano | Kuramoto (1984) - Transición de fase |
| **DFA α** | 1.93 | 1.00 | ❌ +93% | Linkenkaer-Hansen (2001) - Correlaciones críticas |
| **PSD slope** | -4.0 | -1.0 | ❌ -300% | Bak et al. (1987) - Ruido rosa 1/f |
| **Score** | 3.264 | < 1.0 | ❌ 3x mayor | Función objetivo personalizada |

### Interpretación
- ✅ **R cercano a 0.5**: El sistema está cerca del punto crítico de sincronización
- ❌ **DFA α muy alto (1.93)**: Dinámicas SUPERDIFUSIVAS → correlaciones demasiado fuertes
- ❌ **PSD muy negativo (-4.0)**: Espectro sobre-amortiguado → falta ruido rosa
- 📊 **Score alto (3.264)**: Lejos del régimen crítico en métricas temporales

---

## 💭 HISTORIAL DE LA CONVERSACIÓN

### Pregunta 1: "¿Qué significan estos resultados?"
**Usuario preguntó**: ¿Qué significa que 0/101 momentos temporales sean normales y 0/10 clases tengan distribución normal?

**Respuesta Copilot**:
- Las distribuciones **NO-normales son ESPERADAS** en sistemas críticos
- Los sistemas críticos exhiben **distribuciones power-law**, no gaussianas
- Esto es consistente con la teoría (Beggs & Plenz, 2003)
- **No es un error**, es una característica

### Pregunta 2: "¿Esto coincide con la bibliografía?"
**Usuario preguntó**: ¿Estos resultados son consistentes con la literatura de criticalidad?

**Respuesta Copilot**:
- ✅ **R ≈ 0.5**: Coincide con Kuramoto (1984) para transición de fase
- ❌ **DFA α ≈ 1.93**: NO coincide con Linkenkaer-Hansen (2001) que predice α ≈ 1.0
- ❌ **PSD ≈ -4.0**: NO coincide con Bak (1987) que predice slope ≈ -1.0
- **Conclusión**: R está bien, pero dinámicas temporales (DFA, PSD) están MAL

### Pregunta 3: "¿DFA y PSD usan información del modelo?"
**Usuario preguntó**: Esas medidas de DFA, PSD ¿no están utilizando información del modelo?

**Respuesta Copilot**:
- Sí, **SÍ usan información del modelo**
- DFA y PSD analizan **R(t)** - la serie temporal del parámetro de orden
- R(t) es la **salida del modelo** Kuramoto (no los parámetros de entrada)
- La serie R(t) captura la **dinámica temporal emergente** del sistema
- Por eso DFA/PSD pueden revelar si las dinámicas son críticas

### Pregunta 4: Decisión de Optimización
**Usuario**: Aceptó propuesta de hacer grid search para optimizar parámetros

**Copilot propuso**:
- Optimización secuencial (greedy): 25 evaluaciones
- Función objetivo: `score = |R-0.5| + 0.8|DFA-1| + 0.8|PSD+1|`
- Evaluar 100 imágenes por configuración
- Tiempo: 1hr GPU / 3-5hrs CPU

**Usuario**: "hagamos lo siguiente haz la fase exploratoria pero usando GPU para que no se demore menos de 3-5hr"

### Pregunta 5: GPU en Servidor vs Mac M3
**Contexto técnico descubierto**:
- Servidor remoto: Intel i7-11700 (8 cores, 16 threads)
- GPU servidor: NVIDIA Quadro P1000 (pero sin drivers instalados)
- PyTorch en servidor: v2.9.0+cpu (sin soporte CUDA)
- Mac del usuario: Apple M3 (10 GPU cores) con Metal

**Decisión final**:
- ❌ No instalar drivers NVIDIA en servidor (es máquina compartida)
- ✅ Ejecutar en Mac M3 con aceleración GPU (MPS)
- Razón: M3 GPU es 4-6x más rápido que CPU, tiempo: 30-60min

---

## 🧪 SOLUCIÓN IMPLEMENTADA

### Script Creado: `optimizar_parametros_secuencial.py`

**Estrategia**: Optimización secuencial (greedy)

```python
# Pseudocódigo del algoritmo
params = baseline_params  # T=100, gamma=0.7, etc.

for param_name in ['T_steps', 'gamma', 'del_t', 'init_omg', 'ksize']:
    mejor_score = infinito
    for valor in valores_a_probar[param_name]:
        params[param_name] = valor
        metricas = evaluar_100_imagenes(params)
        score = calcular_score(metricas)
        if score < mejor_score:
            mejor_valor = valor
            mejor_score = score
    
    params[param_name] = mejor_valor  # Fijar mejor valor
```

**Parámetros a optimizar**:
```python
PARAM_SEARCH_SPACE = {
    'T_steps': [50, 75, 100, 125, 150],     # Pasos de integración temporal
    'gamma': [0.3, 0.5, 0.7, 0.9, 1.1],     # Factor de acoplamiento
    'del_t': [0.5, 0.7, 0.9, 1.1, 1.3],     # Paso temporal dt
    'init_omg': [0.05, 0.1, 0.2, 0.3, 0.5], # Frecuencia natural inicial
    'ksize': [3, 5, 7, 9, 11]                # Tamaño kernel conectividad
}
```

**Función Objetivo**:
```python
def calcular_score(metricas):
    score = 0.0
    score += 1.0 * abs(metricas['R_final'] - 0.5)      # Target: 0.5
    score += 0.8 * abs(metricas['DFA_alpha'] - 1.0)    # Target: 1.0
    score += 0.8 * abs(metricas['PSD_slope'] + 1.0)    # Target: -1.0
    return score
```

**Total**: 25 evaluaciones (5 + 5 + 5 + 5 + 5)

### Modificación en `kuramoto/modelo.py`

**Método añadido** a la clase `KBlock`:

```python
def forward_with_params(self, x, T_steps=100, gamma=0.7, del_t=0.9):
    """
    Wrapper para ejecutar forward con parámetros específicos.
    Útil para grid search de parámetros.
    
    Args:
        x: Imagen de entrada [1, 1, H, W]
        T_steps: Número de pasos temporales
        gamma: Factor de acoplamiento
        del_t: Paso temporal de integración
        
    Returns:
        xs: Lista de estados
        es: Lista de energías
    """
    # Crear campo de acoplamiento desde la imagen
    c = torch.zeros_like(x.expand(-1, self.ch, -1, -1))
    
    # Ejecutar dinámica
    x_final, xs, es = self.forward(
        x.expand(-1, self.ch, -1, -1),
        c,
        T=T_steps,
        gamma=gamma,
        del_t=del_t,
        return_xs=True,
        return_es=True
    )
    
    return xs, es
```

**Propósito**: Permite cambiar parámetros dinámicamente sin recrear el modelo.

---

## 📊 MÉTRICAS CALCULADAS

### Por Imagen (en el loop de optimización)

```python
def calcular_metricas_imagen(xs, es, params):
    """
    xs: Lista de estados [T, B, C, H, W]
    es: Lista de energías [T, B]
    """
    metricas = {}
    
    # 1. Parámetro de orden R(t)
    R_series = KuramotoMetrics.order_parameter(xs, ch_pair=(0, 1))
    metricas['R_final'] = float(R_series[-1])
    
    # 2. DFA alpha (analiza R(t))
    dfa_analyzer = DFA()
    metricas['DFA_alpha'] = dfa_analyzer.compute(R_series)
    
    # 3. PSD slope (analiza R(t))
    psd_analyzer = PSD()
    metricas['PSD_slope'] = psd_analyzer.compute(R_series)
    
    # 4. Magnitud media
    series = KuramotoMetrics.magnitudes_mean_series(xs)
    metricas['Magnitud_mean'] = float(series.mean())
    
    # 5. Mutual Information
    mi_analyzer = MutualInformation()
    mi_matrix = mi_analyzer.compute(xs)
    metricas['MI_mean'] = float(mi_matrix[mask].mean())
    
    # 6. Entropía Shannon
    entropy_analyzer = Entropia()
    entropy_dict = entropy_analyzer.compute(xs)
    metricas['Entropy_mean'] = float(np.mean(list(entropy_dict.values())))
    
    return metricas
```

### Agregación por Configuración

- Se procesan **100 imágenes** (10 por clase)
- Se calcula la **mediana** de cada métrica (más robusta que media)
- Se calcula **score** con las medianas
- Se guarda en CSV

---

## 🔬 FUNDAMENTO TEÓRICO

### ¿Por qué estos valores objetivo?

#### R_final ≈ 0.5 (Kuramoto, 1984)
```
Modelo de Kuramoto original:
dθ_i/dt = ω_i + (K/N) Σ sin(θ_j - θ_i)

Parámetro de orden: R = |⟨e^(iθ)⟩|

- R ≈ 0: Estado desincronizado (K < K_c)
- R ≈ 0.5: TRANSICIÓN DE FASE (K ≈ K_c) ← CRÍTICO
- R ≈ 1: Estado sincronizado (K > K_c)
```

#### DFA α ≈ 1.0 (Linkenkaer-Hansen, 2001)
```
Detrended Fluctuation Analysis mide correlaciones temporales:

F(n) ~ n^α

- α < 0.5: Anti-correlaciones
- α = 0.5: Ruido blanco (sin memoria)
- α = 1.0: RUIDO ROSA (correlaciones críticas) ← CRÍTICO
- α > 1.0: Super-difusión (correlaciones muy fuertes)

En cerebro: α ≈ 1.0 en vigilia, α > 1.0 en epilepsia
```

#### PSD slope ≈ -1.0 (Bak, 1987)
```
Power Spectral Density en sistemas críticos:

S(f) ~ 1/f^β  donde β = -slope

- β = 0 (slope=0): Ruido blanco
- β = 1 (slope=-1): RUIDO ROSA / 1/f ← CRÍTICO
- β = 2 (slope=-2): Ruido browniano

Criticalidad auto-organizada → 1/f
```

### ¿Por qué importa la criticalidad?

**Hipótesis del Cerebro Crítico** (Beggs & Plenz, 2003):
- Máxima capacidad computacional
- Balance entre orden y caos
- Transmisión óptima de información
- Repertorio dinámico máximo

**En nuestro modelo**:
- Si logramos R≈0.5, DFA≈1.0, PSD≈-1.0 simultáneamente
- El modelo estaría en régimen crítico
- Teóricamente: mejor desempeño en clasificación

---

## 📁 ARCHIVOS CREADOS/MODIFICADOS

### Nuevos Archivos

1. **`optimizar_parametros_secuencial.py`** (527 líneas)
   - Script principal de optimización
   - Ubicación: `codigo/analisis_criticalidad_minimalista/`

2. **`INSTRUCCIONES_OPTIMIZACION.md`**
   - Guía de ejecución completa
   - Troubleshooting
   - Interpretación de resultados

3. **`SINCRONIZACION_MAC.md`**
   - Pasos específicos para Mac M3
   - Configuración de MPS (Metal)
   - Checklist pre-ejecución

4. **`CONTEXTO_CONVERSACION.md`** (este archivo)
   - Contexto completo de conversación
   - Historial de preguntas/respuestas
   - Estado del proyecto

### Archivos Modificados

1. **`kuramoto/modelo.py`**
   - Añadido: `forward_with_params()` en clase `KBlock`
   - Líneas añadidas: ~30
   - Propósito: Facilitar grid search

---

## 🎮 CONFIGURACIÓN DE HARDWARE

### Servidor Remoto (donde estamos ahora)
```
CPU: Intel i7-11700 @ 2.50GHz
  - 8 cores físicos
  - 16 threads (hyperthreading)
  - Arquitectura: RocketLake (11th gen, 14nm)

GPU: NVIDIA Quadro P1000
  - Pascal GP107
  - 640 CUDA cores
  - 4GB GDDR5
  - Compute Capability: 6.1
  - Estado: Drivers NO instalados

Python: 3.11.10
PyTorch: 2.9.0+cpu (sin CUDA)
NumPy: 2.1.3
SciPy: 1.14.1
```

### Mac M3 (donde ejecutaremos)
```
Chip: Apple M3
  - Arquitectura: ARM 3nm
  - CPU: 8 cores (4 Performance + 4 Efficiency)
  - GPU: 10 cores
  - Memoria: Unificada (8-24GB según modelo)
  - Metal: Metal 3 API

Ventajas:
  - Memoria unificada → acceso rápido CPU-GPU
  - Eficiencia energética superior
  - PyTorch con MPS (Metal Performance Shaders)
  - 15-25% más rápido que i7-11700 en CPU
  - 4-6x más rápido con GPU vs CPU
```

---

## 📈 EXPECTATIVAS DE RESULTADOS

### Escenario Optimista (Score < 1.0)
```
Ejemplo de configuración exitosa:
  T_steps: 75
  gamma: 0.5
  del_t: 0.7
  init_omg: 0.15
  ksize: 5

Métricas esperadas:
  R_final: 0.498 (error: 0.002)
  DFA α: 1.023 (error: 0.023)
  PSD slope: -1.124 (error: 0.124)
  Score: 0.185

Mejora: 94.3% respecto a baseline

Siguiente paso: Grid search completo (243 combinaciones)
```

### Escenario Moderado (Score 1.0-2.0)
```
Ejemplo:
  T_steps: 100
  gamma: 0.5
  del_t: 0.7
  init_omg: 0.2
  ksize: 5

Métricas:
  R_final: 0.545 (error: 0.045)
  DFA α: 1.234 (error: 0.234)
  PSD slope: -1.456 (error: 0.456)
  Score: 1.523

Mejora: 53% respecto a baseline

Siguiente paso: Grid search selectivo en subespacio prometedor
```

### Escenario Pesimista (Score > 2.5)
```
Métricas:
  R_final: 0.612 (error: 0.112)
  DFA α: 1.687 (error: 0.687)
  PSD slope: -3.234 (error: 2.234)
  Score: 2.862

Mejora: 12% respecto a baseline

Siguiente paso: 
  - Revisar arquitectura (¿4 canales suficientes?)
  - Expandir espacio de búsqueda
  - Considerar otros parámetros (T_model, n, ch)
```

---

## 🚀 PRÓXIMOS PASOS DESPUÉS DE EJECUTAR

### Si Score < 1.0 (Excelente)

1. **Grid Search Completo**:
   ```python
   # 243 combinaciones (3^5)
   T_steps: [mejor-25, mejor, mejor+25]
   gamma: [mejor-0.2, mejor, mejor+0.2]
   del_t: [mejor-0.2, mejor, mejor+0.2]
   init_omg: [mejor*0.5, mejor, mejor*1.5]
   ksize: [mejor-2, mejor, mejor+2]
   ```

2. **Validación Full Dataset**: 10,002 imágenes con mejores parámetros

3. **Análisis de Redes Funcionales** (Fase 2.2):
   - Matriz MI_matrix: conectividad funcional
   - Análisis con NetworkX: modularidad, clustering
   - Visualización por clase

### Si Score 1.0-2.0 (Bueno)

1. **Grid Search Selectivo**: Refinar alrededor de mejores valores
2. **Análisis de sensibilidad**: ¿Qué parámetros importan más?
3. **Validación parcial**: 1,000 imágenes

### Si Score > 2.5 (Marginal)

1. **Diagnóstico**:
   - ¿Qué métricas siguen mal?
   - ¿R, DFA o PSD?
   
2. **Alternativas**:
   - Expandir rangos de búsqueda
   - Probar otros valores de ch (canales): 8, 16
   - Considerar cambios arquitectónicos

---

## 🐛 TROUBLESHOOTING COMÚN

### En Mac: "MPS backend out of memory"

```python
# Solución: Reducir imágenes por evaluación
N_IMAGES_PER_CONFIG = 50  # En lugar de 100
```

### En Mac: "RuntimeError: Placeholder storage has not been allocated"

```python
# Al inicio del script, añadir:
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
```

### Script muy lento (> 2 horas en M3)

```bash
# Verificar que MPS está activo
python3 -c "import torch; print('MPS:', torch.backends.mps.is_available())"

# Si False, reinstalar PyTorch:
pip install --upgrade torch torchvision torchaudio
```

### Error: "ModuleNotFoundError: No module named 'datasets'"

```bash
# El proyecto tiene estructura de paquete local
# Asegurarse de estar en directorio correcto:
cd codigo/analisis_criticalidad_minimalista
```

---

## 🔗 REFERENCIAS TEÓRICAS

### Papeles Clave

1. **Kuramoto, Y. (1984)**. "Chemical Oscillations, Waves, and Turbulence"
   - Definición del parámetro de orden R
   - Transición de fase en K_c

2. **Linkenkaer-Hansen, K. et al. (2001)**. "Long-Range Temporal Correlations and Scaling Behavior in Human Brain Oscillations"
   - DFA α ≈ 1.0 en cerebro sano
   - Pérdida de criticalidad en epilepsia

3. **Bak, P., Tang, C., & Wiesenfeld, K. (1987)**. "Self-organized criticality"
   - Ruido 1/f en sistemas críticos
   - Avalanchas y leyes de potencia

4. **Beggs, J. M., & Plenz, D. (2003)**. "Neuronal avalanches in neocortical circuits"
   - Avalanchas neuronales críticas
   - Distribuciones power-law
   - Máxima capacidad computacional

5. **Acebrón, J. A. et al. (2005)**. "The Kuramoto model: A simple paradigm for synchronization phenomena"
   - Review completo del modelo Kuramoto
   - Aplicaciones en neurociencia

---

## 💾 DATOS Y RESULTADOS PREVIOS

### Dataset
- **Nombre**: MNIST (Modified National Institute of Standards and Technology)
- **Split**: Test set
- **Tamaño**: 10,002 imágenes
- **Clases**: 10 (dígitos 0-9)
- **Distribución**: Natural (desbalanceada)
  - Clase 0: ~980 imágenes
  - Clase 1: ~1,135 imágenes
  - Clase 2: ~1,032 imágenes
  - ...

### Resultados Previos (Full Test Set)

Archivo: `metricas_completas_CORRECTED.pt` (71MB)

Contiene para cada imagen:
```python
{
    'R_series': torch.Tensor,      # Serie temporal de R(t)
    'R_final': float,              # Último valor de R
    'DFA_alpha': float,            # Exponente DFA
    'PSD_slope': float,            # Pendiente PSD
    'Magnitud_mean': float,        # Magnitud media
    'MI_matrix': np.ndarray,       # Matriz 4×4 de MI
    'Entropy_dict': dict,          # Entropía por canal
    'label': int                   # Etiqueta 0-9
}
```

**Estadísticas globales** (medianas por clase):

| Clase | R_final | DFA α | PSD slope | Magnitud | MI | Entropía |
|-------|---------|-------|-----------|----------|-----|----------|
| 0     | 0.691   | 1.928 | -4.012    | 0.433    | 2.13| 3.189    |
| 1     | 0.542   | 1.934 | -3.987    | 0.428    | 2.11| 3.201    |
| 2     | 0.632   | 1.921 | -4.023    | 0.435    | 2.14| 3.195    |
| ...   | ...     | ...   | ...       | ...      | ... | ...      |

**Observación clave**: DFA y PSD son **consistentemente altos** en todas las clases.

---

## 🎯 FUNCIÓN OBJETIVO DETALLADA

### Diseño de Score

```python
def calcular_score(metricas):
    """
    Score compuesto por 3 componentes con pesos ajustados.
    
    Componentes:
    1. Error en R_final (peso 1.0)
       - Más importante: define punto crítico base
    
    2. Error en DFA alpha (peso 0.8)
       - Importante: correlaciones temporales críticas
       - Peso reducido porque afecta dinámica, no equilibrio
    
    3. Error en PSD slope (peso 0.8)
       - Importante: espectro de frecuencias crítico
       - Complementario a DFA
    
    Pesos menores (0.8) permiten flexibilidad en métricas temporales
    mientras se prioriza el punto crítico (R=0.5).
    """
    score = 0.0
    
    # Componente 1: R_final
    if not np.isnan(metricas['R_final']):
        error_R = abs(metricas['R_final'] - 0.5)
        score += 1.0 * error_R
    else:
        score += 10.0  # Penalización severa por falla
    
    # Componente 2: DFA alpha
    if not np.isnan(metricas['DFA_alpha']):
        error_DFA = abs(metricas['DFA_alpha'] - 1.0)
        score += 0.8 * error_DFA
    else:
        score += 10.0
    
    # Componente 3: PSD slope
    if not np.isnan(metricas['PSD_slope']):
        error_PSD = abs(metricas['PSD_slope'] + 1.0)  # Nota: +1.0 porque es negativo
        score += 0.8 * error_PSD
    else:
        score += 10.0
    
    return score
```

### Interpretación de Componentes

**Ejemplo con baseline**:
```
R_final = 0.62  → error = |0.62 - 0.5| = 0.12  → contribución = 1.0 × 0.12 = 0.12
DFA_α = 1.93    → error = |1.93 - 1.0| = 0.93  → contribución = 0.8 × 0.93 = 0.74
PSD = -4.0      → error = |-4.0 + 1.0| = 3.0  → contribución = 0.8 × 3.0 = 2.40

Score total = 0.12 + 0.74 + 2.40 = 3.26 ✓ (coincide con medición)
```

**Objetivo ideal**:
```
R_final = 0.50  → error = 0.00  → contribución = 0.00
DFA_α = 1.00    → error = 0.00  → contribución = 0.00
PSD = -1.0      → error = 0.00  → contribución = 0.00

Score total = 0.00 (perfecto)
```

---

## 🔍 PREGUNTAS FRECUENTES ANTICIPADAS

### ¿Por qué optimización secuencial y no grid search completo?

**Grid search completo**: 5^5 = 3,125 combinaciones × 100 imágenes = 312,500 evaluaciones
- Tiempo: ~173 horas CPU o ~43 horas GPU M3
- Inviable para fase exploratoria

**Optimización secuencial**: 5 parámetros × 5 valores = 25 evaluaciones × 100 imágenes = 2,500 evaluaciones
- Tiempo: ~1 hora GPU M3
- Asume independencia entre parámetros (aproximación razonable)

**Si hay mejora**: Grid search refinado (3^5 = 243 combinaciones) alrededor del óptimo

### ¿Por qué usar mediana en lugar de media?

**Distribuciones NO-normales**:
- Las métricas tienen distribuciones power-law o log-normal
- La **media es sensible a outliers** en distribuciones asimétricas
- La **mediana es robusta** y representa mejor el valor típico

**Ejemplo**:
```
R_values = [0.45, 0.48, 0.51, 0.49, 0.87]  # Un outlier

Media = 0.56  ← Sesgada por 0.87
Mediana = 0.49  ← Valor representativo ✓
```

### ¿100 imágenes son suficientes por configuración?

**Justificación estadística**:
- Con 10 clases → 10 imágenes por clase
- Central Limit Theorem: n=10 suficiente para estimar mediana
- **Fase exploratoria**: balance entre precisión y tiempo
- **Validación final**: 500 imágenes con mejor configuración

**Trade-off**:
- Más imágenes → Mejor estimación, más tiempo
- Menos imágenes → Más rápido, más ruido
- 100 es punto dulce para exploración

### ¿Por qué no optimizar ch (canales) o n (osciladores)?

**Razones**:
1. **Cambios arquitectónicos**: ch y n requieren reentrenar el modelo
2. **Tiempo**: Entrenar modelo es costoso (horas/días)
3. **Fase exploratoria**: Primero optimizar parámetros dinámicos
4. **Siguiente fase**: Si no hay mejora, considerar ch=[8, 16], n=[8]

### ¿Qué pasa si Score no mejora?

**Diagnóstico**:
1. ¿Qué componente de score es mayor? (R, DFA o PSD)
2. ¿Hay tendencias en los parámetros? (ej: mejor con T_steps bajos)

**Opciones**:
- **Expandir rangos**: T_steps=[25, 200], gamma=[0.1, 2.0]
- **Arquitectura**: Probar ch=8 o ch=16
- **Alternativas**: Considerar otros modelos de criticalidad

---

## 📊 ESTRUCTURA DE RESULTADOS ESPERADA

### Archivo CSV (`resultados_optimizacion_secuencial.csv`)

```csv
T_steps,gamma,del_t,init_omg,ksize,R_final_median,DFA_alpha_median,PSD_slope_median,Magnitud_mean,MI_mean,Entropy_mean,score,n_valid,n_total,param_name,param_value
50,0.7,0.9,0.1,7,0.587,1.42,-2.87,0.425,2.09,3.18,2.145,100,100,T_steps,50
75,0.7,0.9,0.1,7,0.524,1.28,-2.13,0.431,2.11,3.19,1.892,100,100,T_steps,75
100,0.7,0.9,0.1,7,0.620,1.93,-4.01,0.433,2.13,3.19,3.264,100,100,T_steps,100
125,0.7,0.9,0.1,7,0.654,2.01,-4.23,0.437,2.15,3.20,3.487,100,100,T_steps,125
150,0.7,0.9,0.1,7,0.678,2.08,-4.45,0.439,2.16,3.21,3.698,100,100,T_steps,150
75,0.3,0.9,0.1,7,0.412,0.89,-0.87,0.398,1.95,3.15,0.892,100,100,gamma,0.3
75,0.5,0.9,0.1,7,0.485,1.05,-1.23,0.418,2.05,3.17,0.647,100,100,gamma,0.5
75,0.7,0.9,0.1,7,0.524,1.28,-2.13,0.431,2.11,3.19,1.892,100,100,gamma,0.7
...
```

### Archivo TXT (`mejor_configuracion.txt`)

```
MEJOR CONFIGURACIÓN ENCONTRADA
==================================================

Parámetros:
  T_steps: 75
  gamma: 0.5
  del_t: 0.7
  init_omg: 0.15
  ksize: 5

Métricas (validación con 500 imágenes):
  R_final: 0.4985
  DFA α: 1.0234
  PSD slope: -1.1245
  Score: 0.1845

Comparación con baseline:
  Baseline score: 3.264
  Mejor score: 0.1845
  Mejora: 94.3%

Componentes de score:
  Error en R: 0.0015 (contribución: 0.0015)
  Error en DFA: 0.0234 (contribución: 0.0187)
  Error en PSD: 0.1245 (contribución: 0.0996)
```

---

## 🎬 COMANDOS DE EJECUCIÓN RÁPIDA

### En Mac (desde cero)

```bash
# 1. Clonar repo
cd ~/proyectos
git clone https://github.com/ACRCris/Proyecto-Inv.-teorica.git
cd Proyecto-Inv.-teorica/codigo/analisis_criticalidad_minimalista

# 2. Crear entorno
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar dependencias
pip install torch torchvision torchaudio
pip install numpy scipy pandas matplotlib tqdm

# 4. Verificar GPU
python3 -c "import torch; print('MPS:', torch.backends.mps.is_available())"

# 5. Modificar detectar_dispositivo() para MPS (ver SINCRONIZACION_MAC.md)

# 6. Ejecutar
python3 optimizar_parametros_secuencial.py
```

### Modificación necesaria en el script

**Archivo**: `optimizar_parametros_secuencial.py`  
**Línea**: ~95  
**Cambio**:

```python
# ANTES (solo detecta CUDA):
def detectar_dispositivo():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠️  GPU no disponible, usando CPU")
    return device

# DESPUÉS (detecta CUDA y MPS):
def detectar_dispositivo():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ GPU CUDA detectada: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✅ GPU Apple Silicon (MPS) detectada")
        print(f"   Usando Metal Performance Shaders")
    else:
        device = torch.device('cpu')
        print("⚠️  GPU no disponible, usando CPU")
    return device
```

---

## ✅ CHECKLIST FINAL ANTES DE EJECUTAR

- [ ] Repo clonado en Mac
- [ ] Entorno virtual creado y activado
- [ ] PyTorch instalado con `torch.backends.mps.is_available() == True`
- [ ] Función `detectar_dispositivo()` modificada para MPS
- [ ] Estás en directorio `codigo/analisis_criticalidad_minimalista`
- [ ] Dataset MNIST descargará automáticamente (necesita ~50MB espacio)
- [ ] Tiempo disponible: ~1 hora para ejecución completa
- [ ] VS Code abierto con GitHub Copilot activo

---

## 🎯 OBJETIVO FINAL DEL EJERCICIO

**Demostrar si el modelo de Kuramoto puede alcanzar criticalidad** mediante optimización de parámetros de integración.

**Pregunta científica**:
¿Es la arquitectura actual (4 canales, 4 osciladores por canal) suficiente para exhibir dinámicas críticas (R≈0.5, DFA≈1.0, PSD≈-1.0) ajustando solo parámetros de integración?

**Resultados posibles**:
1. ✅ **Sí** → El modelo tiene capacidad crítica, solo necesitaba mejores parámetros
2. ❌ **No** → Limitación arquitectónica, requiere modificaciones estructurales

---

## 💬 PARA GITHUB COPILOT EN MAC

**Cuando abras este archivo, di exactamente**:

> "He leído CONTEXTO_CONVERSACION.md. Tengo el proyecto abierto en Mac M3. Voy a ejecutar optimizar_parametros_secuencial.py. ¿Necesito modificar algo antes de ejecutar?"

O más directo:

> "Restaurar contexto desde CONTEXTO_CONVERSACION.md y continuar con optimización"

**Copilot debería entender**:
- Estado actual del proyecto
- Por qué estamos optimizando
- Qué esperamos de resultados
- Cómo interpretar la salida
- Siguientes pasos basados en score

---

**FIN DEL CONTEXTO**

Este archivo contiene TODA la información necesaria para que GitHub Copilot en tu Mac pueda "continuar" la conversación como si no hubiera habido interrupción.

---

**Metadatos**:
- Fecha creación: Octubre 21, 2025
- Sesión original: Servidor remoto (crperezp@sala2)
- Próxima sesión: Mac local con M3
- Archivos relacionados: SINCRONIZACION_MAC.md, INSTRUCCIONES_OPTIMIZACION.md
- Commit: 194cb75
