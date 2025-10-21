# 📊 ANÁLISIS COMPLETO DE DISTRIBUCIONES - Métricas de Criticalidad

**Dataset:** MNIST Training Set - 60,002 imágenes  
**Fecha:** 21 de Octubre, 2025  
**Análisis:** Distribuciones por clase (dígitos 0-9) de 6 métricas principales de criticalidad

---

## 📋 Índice

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Métricas Analizadas](#métricas-analizadas)
3. [Hallazgos Principales](#hallazgos-principales)
4. [Análisis por Métrica](#análisis-por-métrica)
5. [Comparaciones Entre Clases](#comparaciones-entre-clases)
6. [Tests de Normalidad](#tests-de-normalidad)
7. [Conclusiones](#conclusiones)

---

## 🎯 Resumen Ejecutivo

### Resultados Globales por Métrica

| Métrica | Media Global | Rango de Criticalidad | Estado |
|---------|--------------|----------------------|--------|
| **R (Parámetro de Orden)** | 0.6089 ± 0.0707 | 0.3 - 0.7 | ✅ **EN CRITICALIDAD** |
| **Magnitud Media por Canal** | 0.4185 ± 0.0175 | - | ⚪ N/A |
| **PSD Slope** | -4.0606 ± 0.2661 | -1.2 a -0.8 | ❌ Fuera de rango |
| **DFA Alpha** | 1.7108 ± 0.0830 | 0.5 - 1.0 | ❌ Fuera de rango |
| **Mutual Information** | 1.5011 ± 0.1195 | - | ⚪ N/A |
| **Entropía Shannon** | 4.6111 ± 0.3535 | - | ⚪ N/A |

### 🔑 Hallazgos Clave

1. **Todas las métricas muestran diferencias SIGNIFICATIVAS entre clases** (p < 0.001)
2. **Ninguna métrica sigue distribución normal** (todos los tests rechazaron H₀)
3. **El parámetro R está en rango de criticalidad** para todas las clases
4. **La clase 1 (dígito "1")** muestra valores más extremos en múltiples métricas
5. **La clase 0 (dígito "0")** tiene la menor sincronización (R = 0.5363)

---

## 📐 Métricas Analizadas

### 1. **R (Parámetro de Orden) - Valor Estacionario**
Mide la sincronización global de los osciladores Kuramoto.

- **Interpretación:** 
  - R < 0.3: Fase desordenada
  - 0.3 ≤ R ≤ 0.7: **Criticalidad** (sincronización intermedia)
  - R > 0.7: Fase ordenada

### 2. **Magnitud Media por Canal**
Promedio de las magnitudes en los 4 canales de la representación compleja.

- **Interpretación:** Mide la "energía" o amplitud de las oscilaciones por canal

### 3. **PSD Slope (Pendiente del Espectro de Potencia)**
Pendiente del espectro de potencia en escala log-log.

- **Interpretación:**
  - Pendiente ≈ -1: **Ruido 1/f (criticalidad)**
  - Pendiente ≈ -2: Movimiento browniano
  - Pendiente ≈ 0: Ruido blanco

### 4. **DFA Alpha (Exponente de Fluctuaciones)**
Exponente del análisis de fluctuaciones sin tendencias (DFA).

- **Interpretación:**
  - α = 0.5: No correlacionado (ruido blanco)
  - 0.5 < α < 1.0: **Correlaciones de largo alcance (criticalidad)**
  - α = 1.0: Ruido 1/f
  - α > 1.0: Correlaciones no estacionarias

### 5. **Mutual Information (Información Mutua)**
Información compartida entre canales.

- **Interpretación:** Mide dependencias no lineales entre canales

### 6. **Entropía de Shannon (Media por Canal)**
Entropía promedio de las distribuciones de magnitud por canal.

- **Interpretación:** Mide la incertidumbre/diversidad de estados

---

## 🔍 Hallazgos Principales

### Por Clase (ordenadas por R medio)

| Clase | R medio | DFA α | PSD Slope | Shannon H | MI | Interpretación |
|-------|---------|-------|-----------|-----------|-----|----------------|
| **1** | 0.6736 | 1.8349 | -3.8902 | 5.1920 | 1.4307 | Mayor sincronización |
| **7** | 0.6460 | 1.7598 | -4.0028 | 4.9260 | 1.4894 | Alta sincronización |
| **4** | 0.6309 | 1.7089 | -4.0432 | 4.7212 | 1.4947 | Alta sincronización |
| **9** | 0.6291 | 1.6987 | -4.0413 | 4.6655 | 1.5077 | Sincronización intermedia |
| **6** | 0.6080 | 1.6931 | -4.0789 | 4.6029 | 1.5183 | Sincronización intermedia |
| **5** | 0.6072 | 1.6923 | -4.1283 | 4.5880 | 1.5258 | Sincronización intermedia |
| **3** | 0.5892 | 1.6841 | -4.0965 | 4.5100 | 1.5218 | Sincronización intermedia |
| **2** | 0.5813 | 1.6829 | -4.0981 | 4.5145 | 1.5208 | Sincronización intermedia |
| **8** | 0.5768 | 1.6826 | -4.0927 | 4.4136 | 1.5163 | Menor sincronización |
| **0** | 0.5363 | 1.6718 | -4.0779 | 4.1781 | 1.5097 | **Menor sincronización** |

### 📊 Observaciones Clave

#### **Parámetro de Orden (R)**
- ✅ **Todas las clases en rango de criticalidad** (0.3-0.7)
- **Clase 1**: Mayor sincronización (R = 0.6736) - posiblemente por su simplicidad visual (líneas verticales)
- **Clase 0**: Menor sincronización (R = 0.5363) - mayor complejidad topológica
- **Dispersión moderada** (CV: 0.078-0.122) indica estabilidad dentro de cada clase

#### **DFA Alpha**
- ❌ **Todos los valores > 1.0** (fuera de rango típico de criticalidad 0.5-1.0)
- Rango: 1.6718 (clase 0) a 1.8349 (clase 1)
- Indica **correlaciones de largo alcance muy fuertes** o **no estacionariedad**
- Posible explicación: dinámica de 101 pasos con γ=0.7 genera memoria extendida

#### **PSD Slope**
- ❌ **Todos los valores ≈ -4** (esperado: -1 para criticalidad)
- Rango: -3.8902 (clase 1) a -4.1283 (clase 5)
- Pendiente muy negativa indica **decaimiento rápido de frecuencias altas**
- Posible explicación: 
  - Acoplamiento fuerte (γ=0.7) suaviza oscilaciones rápidas
  - Tamaño kernel (7×7) actúa como filtro paso-bajo

#### **Entropía de Shannon**
- Correlación positiva con R (clases más sincronizadas → mayor entropía)
- **Clase 1**: Mayor entropía (H = 5.1920)
- **Clase 0**: Menor entropía (H = 4.1781)
- Diferencia de ~25% entre extremos

#### **Mutual Information**
- Valores relativamente estables (1.43-1.52)
- **Clase 8**: Mayor MI (1.5163) - alta dependencia entre canales
- **Clase 1**: Menor MI (1.4307) - canales más independientes

---

## 📈 Análisis por Métrica

### 1. R (Parámetro de Orden)

#### Estadísticas Descriptivas por Clase

| Clase | N | Media | Mediana | Std | Min | Max | Skewness | Kurtosis |
|-------|---|-------|---------|-----|-----|-----|----------|----------|
| 0 | 5,913 | 0.5363 | 0.5387 | 0.0655 | 0.1649 | 0.6989 | -0.403 | 0.301 |
| 1 | 6,693 | **0.6736** | 0.6979 | 0.0638 | 0.4310 | 0.7845 | -1.566 | 1.856 |
| 2 | 5,968 | 0.5813 | 0.5843 | 0.0621 | 0.3253 | 0.7485 | -0.369 | -0.075 |
| 3 | 6,148 | 0.5892 | 0.5941 | 0.0628 | 0.3269 | 0.7727 | -0.485 | -0.029 |
| 4 | 5,810 | 0.6309 | 0.6407 | 0.0496 | 0.3982 | 0.7532 | -0.919 | 0.758 |
| 5 | 5,467 | 0.6072 | 0.6153 | 0.0625 | 0.3174 | 0.7621 | -0.687 | 0.110 |
| 6 | 5,904 | 0.6080 | 0.6156 | 0.0614 | 0.3513 | 0.7531 | -0.633 | 0.090 |
| 7 | 6,288 | 0.6460 | 0.6589 | 0.0509 | 0.4119 | 0.7614 | -1.301 | 1.881 |
| 8 | 5,827 | 0.5768 | 0.5836 | 0.0636 | 0.2401 | 0.7104 | -0.619 | 0.333 |
| 9 | 5,984 | 0.6291 | 0.6384 | 0.0528 | 0.3780 | 0.7588 | -1.004 | 1.108 |

#### Interpretación
- **Asimetría negativa** en todas las clases: más imágenes con R alto que bajo
- **Clase 1 y 7**: Kurtosis positiva (distribución leptocúrtica = pico pronunciado)
- **Variabilidad moderada** (CV: 7.8%-12.2%)

---

### 2. Magnitud Media por Canal

| Clase | Media | Std | Min | Max |
|-------|-------|-----|-----|-----|
| 0 | 0.4103 | 0.0158 | 0.3538 | 0.4530 |
| 1 | **0.4307** | 0.0138 | 0.3905 | 0.4674 |
| 2 | 0.4144 | 0.0148 | 0.3653 | 0.4590 |
| 3 | 0.4157 | 0.0154 | 0.3698 | 0.4645 |
| 4 | 0.4233 | 0.0130 | 0.3841 | 0.4587 |
| 5 | 0.4194 | 0.0158 | 0.3656 | 0.4609 |
| 6 | 0.4195 | 0.0153 | 0.3724 | 0.4601 |
| 7 | 0.4260 | 0.0126 | 0.3875 | 0.4595 |
| 8 | 0.4143 | 0.0165 | 0.3575 | 0.4549 |
| 9 | 0.4228 | 0.0142 | 0.3802 | 0.4632 |

#### Interpretación
- **Muy baja dispersión** (CV: 3%-4%) - valores muy consistentes
- Correlación positiva con R (más sincronización → mayor magnitud)

---

### 3. PSD Slope

| Clase | Media | Std | Min | Max | Interpretación |
|-------|-------|-----|-----|-----|----------------|
| 0 | -4.0779 | 0.2638 | -4.9955 | -3.1378 | Decaimiento rápido |
| 1 | **-3.8902** | 0.2389 | -4.8218 | -3.1019 | Menor decaimiento |
| 2 | -4.0981 | 0.2560 | -5.0195 | -3.2043 | Decaimiento rápido |
| 3 | -4.0965 | 0.2598 | -5.0340 | -3.2185 | Decaimiento rápido |
| 4 | -4.0432 | 0.2385 | -4.8911 | -3.2534 | Decaimiento rápido |
| 5 | **-4.1283** | 0.2670 | -5.0844 | -3.2632 | Mayor decaimiento |
| 6 | -4.0789 | 0.2644 | -5.0495 | -3.2243 | Decaimiento rápido |
| 7 | -4.0028 | 0.2332 | -4.8530 | -3.1880 | Decaimiento rápido |
| 8 | -4.0927 | 0.2821 | -5.0924 | -3.1546 | Decaimiento rápido |
| 9 | -4.0413 | 0.2451 | -4.9268 | -3.2227 | Decaimiento rápido |

#### Interpretación
- ❌ **Lejos del valor crítico** (-1)
- Pendiente ~-4 sugiere filtrado paso-bajo fuerte
- **Clase 1**: Pendiente menos negativa (espectro más rico en altas frecuencias)

---

### 4. DFA Alpha

| Clase | Media | Std | Min | Max | Interpretación |
|-------|-------|-----|-----|-----|----------------|
| 0 | 1.6718 | 0.0782 | 1.3757 | 1.9253 | Correlaciones fuertes |
| 1 | **1.8349** | 0.0688 | 1.5778 | 2.0377 | Correlaciones muy fuertes |
| 2 | 1.6829 | 0.0751 | 1.4209 | 1.9563 | Correlaciones fuertes |
| 3 | 1.6841 | 0.0762 | 1.4192 | 1.9694 | Correlaciones fuertes |
| 4 | 1.7089 | 0.0701 | 1.4779 | 1.9633 | Correlaciones fuertes |
| 5 | 1.6923 | 0.0778 | 1.4195 | 1.9787 | Correlaciones fuertes |
| 6 | 1.6931 | 0.0760 | 1.4384 | 1.9757 | Correlaciones fuertes |
| 7 | 1.7598 | 0.0670 | 1.5170 | 1.9863 | Correlaciones muy fuertes |
| 8 | 1.6826 | 0.0826 | 1.3900 | 1.9505 | Correlaciones fuertes |
| 9 | 1.6987 | 0.0724 | 1.4613 | 1.9698 | Correlaciones fuertes |

#### Interpretación
- ❌ **Todos > 1.0** (fuera de rango típico de criticalidad)
- **Clase 1**: α más alto (1.8349) - mayor memoria/persistencia
- Indica series no estacionarias con correlaciones de largo alcance

---

### 5. Mutual Information

| Clase | Media | Std | Min | Max |
|-------|-------|-----|-----|-----|
| 0 | 1.5097 | 0.1128 | 1.1373 | 1.8399 |
| 1 | **1.4307** | 0.1055 | 1.0982 | 1.7385 |
| 2 | 1.5208 | 0.1168 | 1.1386 | 1.8684 |
| 3 | 1.5218 | 0.1174 | 1.1404 | 1.8723 |
| 4 | 1.4947 | 0.1050 | 1.1696 | 1.8002 |
| 5 | 1.5258 | 0.1214 | 1.1383 | 1.8819 |
| 6 | 1.5183 | 0.1199 | 1.1416 | 1.8746 |
| 7 | 1.4894 | 0.1026 | 1.1544 | 1.7819 |
| 8 | **1.5163** | 0.1303 | 1.1129 | 1.8661 |
| 9 | 1.5077 | 0.1119 | 1.1516 | 1.8358 |

#### Interpretación
- **Valores relativamente estables** entre clases (CV: 7.3%-8.6%)
- **Clase 1**: Menor MI - canales más independientes
- **Clase 8**: Mayor MI - mayor dependencia entre canales

---

### 6. Entropía de Shannon (Media por Canal)

| Clase | Media | Std | Min | Max |
|-------|-------|-----|-----|-----|
| 0 | 4.1781 | 0.3088 | 2.9949 | 5.0419 |
| 1 | **5.1920** | 0.2656 | 4.3557 | 5.7953 |
| 2 | 4.5145 | 0.3021 | 3.4656 | 5.4292 |
| 3 | 4.5100 | 0.3083 | 3.4822 | 5.4667 |
| 4 | 4.7212 | 0.2582 | 3.8345 | 5.4461 |
| 5 | 4.5880 | 0.3186 | 3.4780 | 5.4957 |
| 6 | 4.6029 | 0.3094 | 3.6238 | 5.4633 |
| 7 | 4.9260 | 0.2547 | 4.0695 | 5.6166 |
| 8 | **4.4136** | 0.3390 | 3.2578 | 5.2755 |
| 9 | 4.6655 | 0.2868 | 3.7424 | 5.4995 |

#### Interpretación
- **Gran variabilidad entre clases** (diferencia de ~25% entre clase 0 y 1)
- **Clase 1**: Entropía más alta (H = 5.19) - mayor diversidad de estados
- **Clase 0**: Entropía más baja (H = 4.18) - estados más concentrados
- Correlación positiva con R y magnitud

---

## 🔬 Comparaciones Entre Clases

### Tests Estadísticos (ANOVA y Kruskal-Wallis)

| Métrica | ANOVA F | ANOVA p | K-W H | K-W p | Conclusión |
|---------|---------|---------|-------|-------|------------|
| **R** | 2674.89 | < 0.001 | 18316.35 | < 0.001 | ✅ **MUY SIGNIFICATIVO** |
| **Magnitud** | 795.51 | < 0.001 | 14037.47 | < 0.001 | ✅ **MUY SIGNIFICATIVO** |
| **PSD Slope** | 485.68 | < 0.001 | 4474.37 | < 0.001 | ✅ **MUY SIGNIFICATIVO** |
| **DFA Alpha** | 2980.44 | < 0.001 | 12888.85 | < 0.001 | ✅ **MUY SIGNIFICATIVO** |
| **MI** | 327.72 | < 0.001 | 4753.85 | < 0.001 | ✅ **MUY SIGNIFICATIVO** |
| **Entropía** | 4165.96 | < 0.001 | 15841.90 | < 0.001 | ✅ **MUY SIGNIFICATIVO** |

### Interpretación
- **Todas las métricas muestran diferencias altamente significativas** entre clases (p < 0.001)
- Tanto tests paramétricos (ANOVA) como no paramétricos (Kruskal-Wallis) coinciden
- **Las clases de dígitos son distinguibles** a través de métricas de criticalidad

---

## 📊 Tests de Normalidad

### Resumen General
**RESULTADO: Ninguna métrica en ninguna clase sigue distribución normal**

Todos los tests (Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov, D'Agostino-Pearson) rechazan la hipótesis nula de normalidad (p < 0.05).

### Características de las Distribuciones

#### **Asimetría (Skewness)**
- **R**: Asimetría negativa en todas las clases (-1.57 a -0.37)
  - Más imágenes con valores altos de sincronización
  - Cola izquierda (valores bajos) más extendida

- **DFA Alpha**: Asimetría cercana a cero o ligeramente negativa
  - Distribuciones más simétricas

- **PSD Slope**: Asimetría variable entre clases
  - Mayoría con asimetría positiva (cola derecha)

#### **Curtosis (Kurtosis)**
- **R en clases 1, 7, 9**: Kurtosis alta (1.1 a 1.9)
  - Distribuciones leptocúrticas (pico pronunciado + colas pesadas)
  
- **R en clases 2, 3**: Kurtosis cercana a 0
  - Distribuciones mesocúrticas (similar a normal)

### Implicaciones
1. **No usar tests paramétricos** que asuman normalidad (t-test, ANOVA requieren cautela)
2. **Preferir tests no paramétricos**: Kruskal-Wallis, Mann-Whitney U
3. **Medianas y percentiles** son más robustos que medias
4. **Transformaciones** (log, Box-Cox) no mejoran normalidad significativamente

---

## 🎯 Conclusiones

### 1. **Criticalidad del Sistema**
- ✅ **Parámetro R confirma estado crítico** en todas las clases (0.3 < R < 0.7)
- ❌ **DFA Alpha y PSD Slope están fuera de rangos típicos** de criticalidad
  - DFA α > 1.0 sugiere correlaciones de largo alcance muy fuertes
  - PSD slope ≈ -4 indica filtrado paso-bajo excesivo

### 2. **Diferenciación por Clase**
- **Clase 1 (dígito "1")**: 
  - Mayor sincronización (R = 0.67)
  - Mayor DFA α (1.83) - máxima memoria
  - Mayor entropía (H = 5.19)
  - Menor MI (1.43)
  
- **Clase 0 (dígito "0")**:
  - Menor sincronización (R = 0.54)
  - Menor DFA α (1.67)
  - Menor entropía (H = 4.18)
  
- **Interpretación topológica**:
  - Dígitos más "simples" (1, 7) → mayor sincronización
  - Dígitos más "complejos" (0, 8) → menor sincronización

### 3. **Distribuciones No Normales**
- **Todas las distribuciones rechazan normalidad**
- Características:
  - Asimetría negativa (R, Entropía)
  - Kurtosis variable (leptocúrtica en clases 1, 7, 9)
  - Colas pesadas

### 4. **Diferencias Significativas Entre Clases**
- **Todas las métricas discriminan entre clases** (p < 0.001)
- Posibilidad de **clasificación basada en métricas de criticalidad**

### 5. **Efecto de Parámetros del Modelo**
- **γ = 0.7** (acoplamiento fuerte):
  - Genera sincronización moderada-alta
  - Produce correlaciones de largo alcance (DFA α > 1)
  - Suaviza espectro (PSD slope muy negativo)
  
- **T = 100 pasos**:
  - Suficiente para alcanzar estado estacionario
  - Permite correlaciones temporales extendidas

---

## 📂 Archivos Generados

```
resultados_kuramoto_TRAIN_MAC_60k/
├── analisis_distribuciones/
│   ├── informe_completo_distribuciones.txt        (Informe detallado)
│   ├── estadisticas_descriptivas_por_clase.csv    (Todas las estadísticas)
│   ├── tests_normalidad_por_clase.csv             (Resultados de tests)
│   └── comparaciones_entre_clases.csv             (ANOVA y Kruskal-Wallis)
└── graficas_por_clase/
    ├── clase0/  (6 PDFs)
    ├── clase1/  (6 PDFs)
    ├── ...
    └── clase9/  (6 PDFs)
```

---

## 🔗 Referencias

### Archivos de Código
- `analizar_distribuciones_completo.py` - Script de análisis
- `generar_graficas_por_clase.py` - Generación de visualizaciones

### Datos Originales
- `metricas_completas_TRAIN_MAC_60k.pt` - 60,002 imágenes procesadas

### Configuración del Modelo
- **Modelo**: KBlock (Kuramoto oscillator network)
- **Canales**: 4
- **Osciladores**: 4 por píxel
- **Pasos temporales**: 100 (T_STEPS)
- **Acoplamiento**: γ = 0.7
- **Paso temporal**: Δt = 0.9
- **Kernel size**: 7×7

---

**Documento generado el:** 21 de Octubre, 2025  
**Análisis realizado por:** Sistema automatizado de análisis de criticalidad  
**Dataset:** MNIST Training Set (60,002 imágenes)
