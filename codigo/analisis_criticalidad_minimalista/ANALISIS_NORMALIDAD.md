# Análisis de Normalidad de Distribuciones

## 📊 Descripción General

El análisis de normalidad es fundamental para determinar si las métricas de criticalidad siguen una **distribución gaussiana (normal)**. Esto es importante porque:

1. **Validez de tests estadísticos**: Muchos tests (ANOVA, t-test) asumen normalidad
2. **Interpretación de resultados**: Permite usar media ± desviación estándar con confianza
3. **Detección de anomalías**: Desviaciones de normalidad pueden indicar fenómenos interesantes
4. **Elección de métodos**: Determina si usar tests paramétricos o no paramétricos

---

## 🔬 Tests Estadísticos Aplicados

### 1. Test de Shapiro-Wilk

**¿Qué mide?**
- Compara la distribución observada con una distribución normal
- Es muy sensible a desviaciones de normalidad
- **Mejor para muestras pequeñas a medianas** (N < 5000)

**Hipótesis:**
- **H₀** (hipótesis nula): Los datos provienen de una distribución normal
- **H₁** (hipótesis alternativa): Los datos NO provienen de una distribución normal

**Interpretación del p-valor:**
- `p > 0.05` → **NO se rechaza H₀** → Los datos *posiblemente* son normales ✓
- `p ≤ 0.05` → **Se rechaza H₀** → Los datos NO son normales ✗

**Estadístico W:**
- Varía entre 0 y 1
- Valores cercanos a 1 indican mayor normalidad
- W < 0.95 suele indicar desviación significativa

---

### 2. Test de Kolmogorov-Smirnov (KS)

**¿Qué mide?**
- Compara la función de distribución acumulada (CDF) empírica con la teórica
- Mide la **máxima diferencia** entre ambas distribuciones
- Útil para cualquier tamaño de muestra

**Hipótesis:**
- **H₀**: Los datos provienen de una distribución normal
- **H₁**: Los datos NO provienen de una distribución normal

**Interpretación:**
- `p > 0.05` → Datos posiblemente normales ✓
- `p ≤ 0.05` → Datos NO son normales ✗

**Ventajas:**
- Funciona bien con muestras grandes
- Detecta diferencias en la forma general de la distribución

**Limitaciones:**
- Menos potente que Shapiro-Wilk para muestras pequeñas
- Más sensible en el centro que en las colas

---

### 3. Test de Anderson-Darling

**¿Qué mide?**
- Similar a KS pero **más sensible en las colas** de la distribución
- Importante para detectar outliers o eventos extremos
- Da más peso a las desviaciones en los extremos

**Hipótesis:**
- **H₀**: Los datos provienen de una distribución normal
- **H₁**: Los datos NO provienen de una distribución normal

**Interpretación:**
- Si `estadístico < valor_crítico` → Datos posiblemente normales ✓
- Si `estadístico ≥ valor_crítico` → Datos NO son normales ✗

**Valores críticos (al 5% de significancia):**
- Varían según el tamaño de muestra
- Típicamente entre 0.7 y 0.8 para N grande

---

## 📐 Medidas de Forma de la Distribución

### Asimetría (Skewness)

**¿Qué mide?**
- La **simetría** de la distribución respecto a la media
- Una distribución normal tiene asimetría = 0

**Interpretación:**
- `Skew ≈ 0` → **Distribución simétrica** (ideal para gaussiana) ✓
- `Skew > 0` → **Asimetría positiva** (cola hacia la derecha) →
  - Media > Mediana
  - Más valores bajos, pocos valores muy altos
- `Skew < 0` → **Asimetría negativa** (cola hacia la izquierda) ←
  - Media < Mediana
  - Más valores altos, pocos valores muy bajos

**Valores típicos:**
- `|Skew| < 0.5` → Aproximadamente simétrica
- `0.5 < |Skew| < 1.0` → Moderadamente asimétrica
- `|Skew| > 1.0` → Altamente asimétrica

---

### Curtosis (Kurtosis)

**¿Qué mide?**
- El **peso de las colas** de la distribución
- La **agudeza del pico** central
- Una distribución normal tiene curtosis exceso = 0

**Interpretación (curtosis exceso - Fisher):**
- `Kurt ≈ 0` → **Curtosis similar a gaussiana** (mesocúrtica) ✓
- `Kurt > 0` → **Colas más pesadas** que gaussiana (leptocúrtica)
  - Más valores extremos
  - Pico más agudo
  - Mayor probabilidad de outliers
- `Kurt < 0` → **Colas más ligeras** que gaussiana (platicúrtica)
  - Menos valores extremos
  - Distribución más plana
  - Menor probabilidad de outliers

**Valores típicos:**
- `|Kurt| < 0.5` → Similar a gaussiana
- `0.5 < |Kurt| < 1.0` → Moderadamente diferente
- `|Kurt| > 1.0` → Muy diferente de gaussiana

---

## 📊 Visualizaciones Generadas

### 1. Histogramas con Ajuste Gaussiano

**¿Qué muestran?**
- Barras: Distribución empírica de los datos
- Línea roja: Ajuste de distribución normal N(μ, σ²)
- Si los datos son normales, las barras deben seguir la curva roja

**Cómo interpretar:**
- **Buen ajuste**: Las barras siguen la curva, título verde
- **Mal ajuste**: Las barras se desvían significativamente, título rojo
- **Título verde**: Shapiro-Wilk no rechaza normalidad (p > 0.05)
- **Título rojo**: Shapiro-Wilk rechaza normalidad (p ≤ 0.05)

**Información en título:**
- N: Número de muestras
- p: p-valor del test de Shapiro-Wilk
- Skew: Asimetría de la distribución
- Kurt: Curtosis (exceso) de la distribución

---

### 2. Q-Q Plots (Quantile-Quantile Plots)

**¿Qué muestran?**
- **Eje X**: Cuantiles teóricos de una distribución normal estándar
- **Eje Y**: Cuantiles empíricos de los datos observados
- **Línea roja**: Línea de referencia y = x (normalidad perfecta)
- **Puntos azules**: Datos observados

**Cómo interpretar:**

✅ **Datos normales:**
- Los puntos caen sobre o muy cerca de la línea roja
- Desviaciones mínimas en los extremos son aceptables

⚠️ **Desviaciones de normalidad:**
- **Cola superior por encima de la línea** → Cola derecha pesada (más outliers altos)
- **Cola inferior por debajo de la línea** → Cola izquierda pesada (más outliers bajos)
- **Curva en S** → Asimetría (skewness)
- **Puntos alejados en ambos extremos** → Colas pesadas (kurtosis positiva)
- **Puntos más cerca en extremos** → Colas ligeras (kurtosis negativa)

**Título:**
- Verde: Shapiro-Wilk no rechaza normalidad
- Rojo: Shapiro-Wilk rechaza normalidad

---

## 📋 Criterios de Decisión

### ¿Cuándo considerar que los datos son normales?

**Criterio conservador (estricto):**
- Los 3 tests (Shapiro-Wilk, KS, Anderson) no rechazan H₀
- |Skewness| < 0.5
- |Kurtosis| < 0.5
- Q-Q plot muestra puntos sobre la línea

**Criterio moderado (recomendado):**
- Al menos 2 de 3 tests no rechazan H₀
- |Skewness| < 1.0
- |Kurtosis| < 1.0
- Q-Q plot muestra desviaciones menores

**Criterio flexible:**
- Al menos 1 test no rechaza H₀
- Histograma muestra forma aproximadamente gaussiana
- Desviaciones explican fenómeno de interés

---

## 🎯 Implicaciones para el Análisis de Criticalidad

### Si las métricas son normales:

✅ **Ventajas:**
- Podemos usar **ANOVA** para comparar clases
- Intervalos de confianza basados en μ ± 2σ son válidos
- Tests paramétricos (t-test) son apropiados
- Interpretación estadística estándar

### Si las métricas NO son normales:

⚠️ **Consideraciones:**
- Usar **tests no paramétricos** (Kruskal-Wallis en lugar de ANOVA)
- Reportar **mediana** en lugar de media
- Usar **rangos intercuartílicos** (IQR) en lugar de desviación estándar
- Considerar **transformaciones** (log, Box-Cox) para normalizar

🔬 **Valor científico:**
- Las desviaciones de normalidad pueden ser **biológicamente significativas**
- Colas pesadas → Presencia de estados metaestables
- Asimetría → Transiciones de fase asimétricas
- Multimodalidad → Múltiples estados atractores

---

## 📈 Ejemplo de Interpretación

### Caso: Métrica R_mean en Clase 5

**Resultados hipotéticos:**
```
Shapiro-Wilk: p = 0.032 → Rechaza normalidad ✗
KS: p = 0.048 → Rechaza normalidad ✗
Anderson: stat = 0.85 > 0.75 → Rechaza normalidad ✗
Skewness: +0.82 → Asimetría positiva moderada
Kurtosis: +1.45 → Colas pesadas
```

**Interpretación:**
1. **Conclusión**: Los datos NO son normales
2. **Forma**: Asimetría hacia la derecha, más valores bajos que altos
3. **Colas**: Presencia de valores extremos altos (outliers)
4. **Implicación física**: Posibles eventos de sincronización espontánea
5. **Acción**: 
   - Usar mediana en lugar de media
   - Aplicar tests no paramétricos
   - Investigar outliers (¿imágenes especiales?)

---

## 🔄 Transformaciones para Normalizar

Si las métricas no son normales, se pueden aplicar:

### 1. Transformación Logarítmica
```python
datos_transformados = np.log(datos + constante)
```
- **Útil para**: Asimetría positiva, colas derechas pesadas
- **Efecto**: Comprime valores altos, expande valores bajos

### 2. Transformación de Box-Cox
```python
from scipy.stats import boxcox
datos_transformados, lambda_opt = boxcox(datos)
```
- **Útil para**: Encontrar transformación óptima automáticamente
- **Efecto**: λ óptimo que maximiza normalidad

### 3. Transformación de Raíz Cuadrada
```python
datos_transformados = np.sqrt(datos)
```
- **Útil para**: Datos de conteo, asimetría moderada
- **Efecto**: Menos agresiva que log

---

## ✅ Checklist de Análisis

Cuando ejecutes `python analizar_estadisticas_full_dataset.py`, verifica:

- [ ] **Tests de normalidad** en `tests_normalidad.csv`
- [ ] **Resumen interpretativo** en `resumen_normalidad.txt`
- [ ] **Histogramas** con ajuste gaussiano (6 PDFs)
- [ ] **Q-Q plots** para verificación visual (6 PDFs)
- [ ] **Patrones por clase**: ¿Alguna clase consistentemente no-normal?
- [ ] **Patrones por métrica**: ¿Alguna métrica nunca es normal?
- [ ] **Decisión**: ¿Usar tests paramétricos o no paramétricos?

---

## 📚 Referencias

1. **Shapiro, S. S., & Wilk, M. B. (1965)**. "An analysis of variance test for normality (complete samples)". *Biometrika*, 52(3/4), 591-611.

2. **Anderson, T. W., & Darling, D. A. (1954)**. "A test of goodness of fit". *Journal of the American Statistical Association*, 49(268), 765-769.

3. **Razali, N. M., & Wah, Y. B. (2011)**. "Power comparisons of Shapiro-Wilk, Kolmogorov-Smirnov, Lilliefors and Anderson-Darling tests". *Journal of Statistical Modeling and Analytics*, 2(1), 21-33.

---

**Última actualización**: 2025-10-19  
**Autor**: Análisis de Criticalidad - Fase 2.1
