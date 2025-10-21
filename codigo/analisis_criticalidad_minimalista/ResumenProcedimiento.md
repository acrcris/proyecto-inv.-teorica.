# 🔬 EXPLICACIÓN A GROSSO MODO DEL PROCEDIMIENTO

## FASE 1: CARGA DE DATOS (datasets/loader.py)

### 📥 MNIST Test Dataset (torchvision)

- **10,002 imágenes** de dígitos (0-9)
- **Tamaño original**: 28×28 píxeles
- **Distribución REAL del dataset**:
    - Clase 0: 980 imágenes (dígito "0")
    - Clase 1: 1136 imágenes (dígito "1") ← MÁS imágenes
    - Clase 2: 1032 imágenes
    - Clase 3: 1010 imágenes
    - Clase 4: 982 imágenes
    - Clase 5: 892 imágenes ← MENOS imágenes
    - Clase 6: 958 imágenes
    - Clase 7: 1028 imágenes
    - Clase 8: 974 imágenes
    - Clase 9: 1010 imágenes

> ⚠️ **IMPORTANTE**: La distribución desigual NO es error, es la distribución NATURAL de MNIST test.

### 🔄 Preprocesamiento

1. **Resize**: 28×28 → 64×64 píxeles
2. **Normalización**: media=0.1307, std=0.3081
3. **ToTensor**: Convierte a tensor PyTorch

---

## FASE 2: PROCESAMIENTO POR IMAGEN (run_kuramoto_full_dataset.py)

Para **CADA imagen** (idx = 0, 1, 2, ..., 10001):

1. 📷 **Imagen MNIST** (64×64)
2. 🔁 **Replicar en 4 canales** → c: [1, 4, 64, 64]
     - Cada canal representa un grupo de osciladores
3. 🎲 **Estado inicial aleatorio** → x: [1, 4, 64, 64]
     - Fases iniciales de osciladores

### MODELO KURAMOTO (kuramoto/modelo.py)

#### 🔄 Iteración temporal: t = 0, 1, 2, ..., 100

En cada paso t:

1. **Calcular acoplamiento entre osciladores**
     - `dx/dt = Ω·x + K·(proj(vecinos) - x)`

2. **Integración numérica (Euler)**
     - `x(t+1) = x(t) + γ·Δt·dx/dt`

3. **Normalización**: `x → x/|x|`
     - Mantiene osciladores en círculo unidad

#### Parámetros

- `T_STEPS = 100` (pasos temporales)
- `γ = 0.7` (acoplamiento)
- `Δt = 0.9` (paso temporal)
- `ksize = 7` (kernel de conectividad)
- `init_omg = 0.1` (frecuencias naturales)

### 📊 Salidas del modelo

- **xs**: Lista de 101 estados [x(0), x(1), ..., x(100)]
- **es**: Lista de 101 energías [e(0), e(1), ..., e(100)]

---

## FASE 3: CÁLCULO DE MÉTRICAS (calcular_metricas_imagen_completas)

A partir de xs y es, calcular:

### 1️⃣ PARÁMETRO DE ORDEN R (KuramotoMetrics)

- **Fórmula**: `R(t) = |⟨e^(iθ_k)⟩|`
- **Mide**: Sincronización global
- **Resultado**: `R_series: [R(0), R(1), ..., R(100)]` (101 valores)

### 2️⃣ MAGNITUD GLOBAL (KuramotoMetrics)

- **Fórmula**: `M(t) = mean(|x_canal|)` para cada canal
- **Resultado**: `global_series: [M(0), M(1), ..., M(100)]` (101 valores)

### 3️⃣ ENERGÍA DEL SISTEMA

- **Fórmula**: `E(t) = -∑ x_i · (K·x)_i`
- **Resultado**: `energy_series: [E(0), E(1), ..., E(100)]` (101 valores)

### 4️⃣ DFA ALPHA (Detrended Fluctuation Analysis)

Analiza correlaciones de largo alcance en R(t):

1. Integrar serie: `Y(k) = Σ[R(i) - R_mean]`
2. Dividir en ventanas de tamaño s
3. Ajustar tendencia local en cada ventana
4. Calcular fluctuación: `F(s) = √⟨[Y-fit]²⟩`
5. Ajuste: `F(s) ~ s^α`

**Resultado**: `DFA_alpha`: 1 valor (α ≈ 1.0 = crítico)

### 5️⃣ PSD SLOPE (Power Spectral Density)

Analiza frecuencias en la serie global_series:

1. FFT de la serie temporal
2. Calcular potencia: `P(f) = |FFT|²`
3. Ajuste log-log: `log(P) ~ slope·log(f)`

**Resultado**: `PSD_slope`: 1 valor (≈ -1.0 = ruido 1/f)

### 6️⃣ ENTROPÍA DE SHANNON (por canal)

- **Fórmula**: `H = -∑ p(x)·log₂(p(x))`
- **Mide**: Complejidad de cada canal
- **Resultado**: `entropy_by_channel: {'Canal_0': H₀, 'Canal_1': H₁, ...}` (4 valores)

### 7️⃣ CORRELACIÓN DE PEARSON (entre canales)

- **Fórmula**: `ρ_ij = cov(X_i, X_j) / (σ_i · σ_j)`
- **Mide**: Dependencia lineal entre canales
- **Resultado**: `correlation_matrix`: Matriz 4×4

### 8️⃣ INFORMACIÓN MUTUA (entre canales)

- **Fórmula**: `MI(X,Y) = ∑∑ p(x,y)·log[p(x,y)/(p(x)p(y))]`
- **Mide**: Dependencia no-lineal entre canales
- **Resultado**: `MI_matrix`: Matriz 4×4 (valores altos = alta dependencia)

### 9️⃣ VARIANZA TEMPORAL

- **Fórmula**: `Var(t) = var(canales)` en cada tiempo t
- **Resultado**: `variance_series: [V(0), V(1), ..., V(100)]` (101 valores)

### 📦 RESULTADO GUARDADO

```python
metricas = {
        'idx': número_imagen,
        'label': clase (0-9),
        'R_series': [101 valores],
        'global_series': [101 valores],
        'energy_series': [101 valores],
        'DFA_alpha': 1 valor,
        'PSD_slope': 1 valor,
        'correlation_matrix': 4×4,
        'MI_matrix': 4×4,
        'entropy_by_channel': {'Canal_0': H₀, ...},
        'variance_series': [101 valores],
        'success': True
}
```

---

## FASE 4: GUARDADO (checkpoints cada 100 imágenes)

### Cada 100 imágenes procesadas

💾 **checkpoint_XXXXX.pt**
- Contiene TODAS las métricas de las imágenes 0 a XXXXX
- Tamaño crece: ~700KB → 71MB (checkpoint final)
- Formato: `torch.save()` con `weights_only=False`

### Al finalizar 10,002 imágenes

💾 **metricas_completas_CORRECTED.pt** (71 MB)
- Todas las métricas de TODAS las imágenes
- Sin promediar ni colapsar información
- Listo para análisis estadístico

---

## FASE 5: ANÁLISIS ESTADÍSTICO (posterior)

### Flujo de análisis

1. 📊 Cargar `metricas_completas_CORRECTED.pt`

2. **Por cada imagen, calcular agregados**:
     - `R_median = median(R_series)` ← 101 → 1 valor
     - `R_final = R_series[-1]` ← Último valor
     - `Global_median = median(global_series)`
     - `MI_median = median(triangular_superior(MI_matrix))`
     - `Entropy_median = median([H₀, H₁, H₂, H₃])`
     - `Corr_median = median(triangular_superior(corr_matrix))`

3. **DataFrame** con 10,002 filas × 8 columnas

4. **Agrupar por clase (0-9)**:
     - Calcular mediana, Q1, Q3, IQR por clase
     - Tests de normalidad (Shapiro-Wilk)
     - Skewness, Kurtosis

5. 📈 **Generar gráficas**:
     - Histograma + KDE
     - Q-Q plots
     - Boxplots
     - Violin plots

6. 💾 **80 PDFs** organizados por clase

---

## 📊 VOLUMEN DE DATOS GENERADO

### Por imagen

- `R_series`: 101 valores float32 = 404 bytes
- `global_series`: 101 valores = 404 bytes
- `energy_series`: 101 valores = 404 bytes
- `variance_series`: 101 valores = 404 bytes
- Matrices 4×4: 32 bytes cada una
- Otros: ~100 bytes

**TOTAL por imagen**: ~2 KB

### Para 10,002 imágenes

- 10,002 × 2 KB = ~20 MB (datos puros)
- + Metadata, índices, etc. = **71 MB total**

### Para análisis completo

- R temporal: 10,002 × 101 = **1,010,202 valores**
- Permite distribuciones robustas sin pérdida de información

---

## 🎯 RAZÓN DE LA DISTRIBUCIÓN DESIGUAL POR CLASE

El dataset MNIST test tiene una distribución **NATURAL** desigual:

| Clase | Imágenes | Porcentaje |
|-------|----------|------------|
| Clase 1 | 1136 | 11.4% | ← Más frecuente
| Clase 2 | 1032 | 10.3% |
| Clase 7 | 1028 | 10.3% |
| Clase 3 | 1010 | 10.1% |
| Clase 9 | 1010 | 10.1% |
| Clase 4 | 982 | 9.8% |
| Clase 0 | 980 | 9.8% |
| Clase 8 | 974 | 9.7% |
| Clase 6 | 958 | 9.6% |
| Clase 5 | 892 | 8.9% | ← Menos frecuente

✅ Esto **NO es un error**, es la distribución REAL del dataset.  
✅ Se procesan **TODAS** las imágenes disponibles por clase.  
✅ Las estadísticas se calculan correctamente para cada clase.
