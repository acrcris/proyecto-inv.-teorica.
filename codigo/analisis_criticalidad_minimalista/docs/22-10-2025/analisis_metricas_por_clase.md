# Análisis de Métricas de Criticalidad por Clase

**Fecha**: Octubre 19, 2025  
**Fase**: 2.1 - Análisis con Dataset Completo MNIST (10,000 imágenes)  
**Estado**: En ejecución (procesamiento en background)

---

## 📊 Interpretación de Métricas

### 1. R_mean - Parámetro de Orden de Kuramoto

**¿Qué mide?** Nivel de sincronización entre osciladores

**Rango de valores:**
- `R = 0`: Desorden total (osciladores completamente desincronizados)
- `R = 0.5`: **PUNTO CRÍTICO** - Balance perfecto entre orden y desorden
- `R = 1`: Orden total (sincronización perfecta)

**Criterio de criticalidad:** ✅ `R ≈ 0.5` (entre 0.4 y 0.6)

**Interpretación física:**
- Estados críticos exhiben máxima capacidad de procesamiento de información
- Balance entre flexibilidad (desorden) y estabilidad (orden)
- Transición de fase de segundo orden en sistemas de osciladores acoplados

---

### 2. DFA_alpha - Detrended Fluctuation Analysis

**¿Qué mide?** Correlaciones de largo alcance en series temporales

**Interpretación de valores:**

| Alpha (α) | Interpretación | Estado |
|-----------|----------------|--------|
| α < 0.5 | Anti-correlaciones (fluctuaciones opuestas) | Sub-crítico |
| α = 0.5 | Ruido blanco (completamente aleatorio) | No correlacionado |
| **α = 1.0** | **Ruido rosa 1/f (CRÍTICO)** | ✅ **Crítico** |
| α = 1.5 | Ruido browniano (movimiento aleatorio integrado) | Super-crítico |
| α > 1.5 | Fuerte memoria de largo plazo | Muy persistente |

**Criterio de criticalidad:** ✅ `α ≈ 1.0` (entre 0.9 y 1.3)

**Significado científico:**
- α = 1.0 indica escala libre (invariancia de escala)
- Sistemas críticos auto-organizados exhiben α ≈ 1.0
- Ejemplos: cerebro en estado de vigilia, terremotos, turbulencia

---

### 3. PSD_slope - Power Spectral Density Slope

**¿Qué mide?** Pendiente de la densidad espectral de potencia (relación frecuencia-amplitud)

**Interpretación de valores:**

| Pendiente | Espectro | Estado |
|-----------|----------|--------|
| slope ≈ 0 | Ruido blanco (plano) | Sin memoria |
| **slope ≈ -1** | **Ruido rosa 1/f (CRÍTICO)** | ✅ **Crítico** |
| slope ≈ -2 | Ruido browniano | Sobre-correlacionado |
| slope < -2 | Dinámica filtrada | Muy amortiguada |

**Criterio de criticalidad:** ✅ `slope ≈ -1.0` (entre -0.5 y -1.5)

**Significado físico:**
- Ruido 1/f es universal en sistemas complejos
- Indica balance entre procesos en múltiples escalas temporales
- Observado en música, fisiología, economía, clima

---

## 🎯 Ejemplo de Análisis: Clase 9 (Resultados Parciales)

### Valores Observados (últimas 100 imágenes)

```
R_mean:    0.5713
DFA_alpha: 1.9334
PSD_slope: -3.9263
```

### Evaluación por Métrica

| Métrica | Valor | Objetivo | Distancia | Evaluación |
|---------|-------|----------|-----------|------------|
| R_mean | 0.571 | 0.5 | 0.071 | ✅ **Excelente** (7% de error) |
| DFA_alpha | 1.933 | 1.0 | 0.933 | ⚠️ Lejos (93% de error) |
| PSD_slope | -3.926 | -1.0 | 2.926 | ⚠️ Muy lejos (293% de error) |

### Interpretación Global de Clase 9

**Fortalezas:**
- ✅ R_mean ≈ 0.5 → Excelente candidato en criticalidad de sincronización
- ✅ Balance orden-desorden muy bueno

**Debilidades:**
- ⚠️ DFA_alpha alto → Demasiada memoria temporal (persistencia excesiva)
- ⚠️ PSD_slope muy negativo → Dinámica sobre-correlacionada en frecuencias

**Veredicto Preliminar:**  
Clase 9 exhibe **criticalidad EN SINCRONIZACIÓN** (R), pero **NO en propiedades temporales/espectrales** (DFA/PSD). Esto sugiere un estado parcialmente crítico.

---

## 📈 Comparación con Resultados de Fase 1

### Clase 5 - Mejor Candidato Global (Fase 1)

| Métrica | Fase 1 (N=1) | Distancia al Crítico |
|---------|--------------|----------------------|
| R_mean | 0.579 | 0.079 |
| DFA_alpha | 1.203 | 0.203 ✅ (mejor) |
| PSD_slope | -2.972 | 1.972 |

**Conclusión Fase 1:** Clase 5 mostró el mejor balance global entre las 3 métricas.

---

### Clase 1 - Criticalidad Espectral (Fase 1)

| Métrica | Fase 1 (N=1) | Distancia al Crítico |
|---------|--------------|----------------------|
| R_mean | 0.746 | 0.246 |
| DFA_alpha | 1.430 | 0.430 |
| PSD_slope | -1.386 | 0.386 ✅ (casi perfecto) |

**Conclusión Fase 1:** Clase 1 exhibió ruido rosa espectral casi perfecto.

---

### Clase 3 - Criticalidad en Sincronización (Fase 1)

| Métrica | Fase 1 (N=1) | Distancia al Crítico |
|---------|--------------|----------------------|
| R_mean | 0.499 | 0.001 ✅ (perfecto) |
| DFA_alpha | 1.723 | 0.723 |
| PSD_slope | -1.861 | 0.861 |

**Conclusión Fase 1:** Clase 3 mostró sincronización exactamente en el punto crítico.

---

## 🔬 Resultados Parciales vs Resultados Finales Completos

### ¿Qué cambiará con los datos completos?

#### NO Esperamos:
- ❌ Que los valores "mejoren" mágicamente
- ❌ Que clases no-críticas se vuelvan críticas
- ❌ Cambios drásticos en promedios (>20%)

#### SÍ Esperamos:
- ✅ **Mayor confianza estadística** (intervalos al 95%)
- ✅ **Mejor precisión** en las estimaciones
- ✅ **Variabilidad intra-clase** documentada
- ✅ **Comparaciones robustas** entre clases (tests ANOVA)
- ✅ **Descubrimiento de subpoblaciones** críticas

---

### Ejemplo de Mejora: Clase 9

**Ahora (100 imágenes):**
```
R_mean:    0.5713 ± ???
DFA_alpha: 1.9334 ± ???
PSD_slope: -3.9263 ± ???
```
*Problema: No sabemos la variabilidad ni confianza*

**Después (1,009 imágenes completas):**
```
R_mean:    0.57 ± 0.08  [IC 95%: 0.49-0.65]
DFA_alpha: 1.85 ± 0.35  [IC 95%: 1.50-2.20]
PSD_slope: -3.75 ± 0.90 [IC 95%: -4.65 a -2.85]
```
*Beneficio: Confianza del 95%, variabilidad conocida, comparaciones válidas*

---

## 📊 Análisis Estadístico Completo (Fase 2.1)

### Objetivos del Análisis Final

Una vez procesadas las 10,000 imágenes, generaremos:

#### 1. Estadísticas Descriptivas por Clase
- Media, mediana, desviación estándar
- Cuartiles (Q1, Q2, Q3)
- Mínimo y máximo
- Intervalos de confianza al 95%

#### 2. Tests de Significancia Estadística
- **ANOVA de una vía**: Comparar las 10 clases
- **p-values**: Determinar si diferencias son significativas (p < 0.05)
- **Post-hoc tests**: Identificar qué pares de clases difieren

#### 3. Ranking de Candidatos Críticos
- **Distancia euclidiana** a valores críticos:
  ```
  d = sqrt((R - 0.5)² + (α - 1.0)² + (slope + 1.0)²)
  ```
- Ordenar clases por proximidad multi-métrica a criticalidad
- Identificar "mejor candidato global"

#### 4. Análisis de Variabilidad Intra-Clase
- **Box plots**: Visualizar distribuciones completas
- **Violin plots**: Mostrar densidad de probabilidad
- Detectar valores atípicos (outliers)
- Identificar subpoblaciones bimodales

#### 5. Correlaciones entre Métricas
- **Matriz de correlación**: R vs DFA vs PSD
- Ejemplo: "¿Alta sincronización implica baja memoria temporal?"
- Identificar trade-offs entre métricas

#### 6. Visualizaciones Científicas
- 6 gráficas en formato PDF vectorial:
  1. Box plots de todas las métricas
  2. Violin plots de métricas clave
  3. Gráficas de medias con error estándar
  4. Matriz de correlaciones
  5. Ranking de distancias a criticalidad
  6. Distribuciones por clase

---

## 💡 Escenarios Esperados con Datos Completos

### Escenario A: Consistencia (más probable)

**Descripción:** Los valores parciales se confirman con todo el dataset

**Ejemplo - Clase 9:**
```
R_mean:    0.57 ± 0.08  → Se mantiene cerca de 0.5 ✓
DFA_alpha: 1.93 ± 0.30  → Se mantiene alto (no crítico)
PSD_slope: -3.92 ± 0.80 → Se mantiene negativo (no crítico)
```

**Conclusión:** Clase 9 es consistentemente crítica EN SINCRONIZACIÓN, pero NO en propiedades temporales/espectrales.

---

### Escenario B: Convergencia (menos probable)

**Descripción:** Los promedios mejoran ligeramente hacia criticalidad

**Ejemplo - Clase 9:**
```
R_mean:    0.55 ± 0.10  → Promedio se acerca más a 0.5
DFA_alpha: 1.75 ± 0.40  → Promedio mejora (era 1.93)
PSD_slope: -3.50 ± 1.20 → Promedio mejora (era -3.92)
```

**Conclusión:** Los valores se estabilizan con mejor precisión, pequeña mejora en DFA y PSD.

---

### Escenario C: Descubrimiento de Subpoblaciones (posible e interesante)

**Descripción:** Alta variabilidad revela grupos con diferentes niveles de criticalidad

**Ejemplo - Clase 9:**
```
R_mean:    0.57 ± 0.15  → Alta variabilidad
DFA_alpha: 1.50 ± 0.60  → Amplio rango [0.9 - 2.1]
PSD_slope: -2.80 ± 1.50 → Amplio rango [-4.3 a -1.3]

Subpoblaciones detectadas:
  • Grupo A (40%): α ∈ [0.9, 1.3] → Críticas temporalmente ✓
  • Grupo B (60%): α ∈ [1.6, 2.3] → No críticas temporalmente
```

**Conclusión:** Clase 9 tiene heterogeneidad interna. Algunas imágenes SÍ exhiben criticalidad en DFA/PSD.

**Implicación científica:** Esto sería un hallazgo importante - la morfología específica dentro de una clase determina el nivel de criticalidad.

---

## 🎓 Valor Científico del Análisis Completo

### De Fase 1 (N=10) a Fase 2.1 (N=10,000)

| Aspecto | Fase 1 | Fase 2.1 |
|---------|--------|----------|
| **Tamaño muestral** | 1 imagen/clase | ~1,000 imágenes/clase |
| **Confianza** | Exploratoria | 95% estadística |
| **Comparaciones** | Descriptivas | Tests significancia |
| **Publicabilidad** | Preliminar | Científicamente válida |
| **Descubrimientos** | Hipótesis | Conclusiones robustas |

---

### Conclusiones que Podremos Hacer

#### Antes (Fase 1):
> "Parece que Clase 5 es más crítica que las otras..."

#### Después (Fase 2.1):
> "Clase 5 es significativamente más crítica que las otras clases (F=45.3, p<0.001), con DFA α=1.20±0.15 (N=892), lo cual es significativamente menor que el promedio de α=1.75±0.40 (p<0.01). Esto indica que las imágenes de Clase 5 exhiben correlaciones de largo alcance consistentes con criticalidad auto-organizada."

**Diferencia:** La segunda afirmación es científicamente robusta y publicable.

---

## 📁 Archivos que se Generarán

### Al Finalizar el Procesamiento

```
resultados_kuramoto_full_dataset/
├── metricas_completas.pt          # Datos completos (10,000 imágenes)
├── checkpoints/
│   ├── checkpoint_00100.pt        # Checkpoints intermedios
│   ├── checkpoint_00200.pt
│   └── ...
└── analisis_estadistico/          # Se crea tras análisis
    ├── estadisticas_por_clase.csv
    ├── tests_significancia.csv
    ├── ranking_criticalidad.csv
    ├── distribuciones_boxplot.pdf
    ├── distribuciones_violin.pdf
    ├── medias_por_clase.pdf
    ├── correlaciones.pdf
    ├── ranking_visual.pdf
    └── resumen_ejecutivo.txt
```

---

## 🚀 Próximos Pasos tras Finalización

1. **Verificar finalización:**
   ```bash
   ps -p $(cat kuramoto_full.pid)  # Si no está activo, terminó
   tail kuramoto_full.log           # Ver estadísticas finales
   ```

2. **Ejecutar análisis estadístico:**
   ```bash
   python analizar_estadisticas_full_dataset.py
   ```

3. **Revisar resultados:**
   - Abrir PDFs de visualizaciones
   - Leer `resumen_ejecutivo.txt`
   - Examinar CSVs de estadísticas

4. **Preparar para Fase 2.2:**
   - Construcción de redes funcionales basadas en correlaciones
   - Análisis de grafos (clustering, hubs, small-world)
   - Detección de comunidades

---

## 📚 Referencias Teóricas

### Criticalidad Auto-Organizada
- Bak, P., Tang, C., & Wiesenfeld, K. (1987). Self-organized criticality. *Physical Review A*, 38(1), 364.

### Ruido 1/f y DFA
- Peng, C. K., et al. (1994). Mosaic organization of DNA nucleotides. *Physical Review E*, 49(2), 1685.

### Sincronización de Kuramoto
- Kuramoto, Y. (1984). *Chemical Oscillations, Waves, and Turbulence*. Springer.

### Criticalidad en Redes Neuronales
- Beggs, J. M., & Plenz, D. (2003). Neuronal avalanches in neocortical circuits. *Journal of Neuroscience*, 23(35), 11167-11177.

---

**Última actualización:** Octubre 19, 2025, 23:30  
**Autor:** Cristian Pérez  
**Proyecto:** Investigación Teórica - Criticalidad en AKOrN sobre MNIST
