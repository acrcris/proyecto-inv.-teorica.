# ANÁLISIS COMPLETO - TODAS LAS MÉTRICAS
## Test Set MNIST (10,002 imágenes)

**Fecha**: Octubre 20, 2025  
**Método**: Mediana (distribuciones no-normales confirmadas)

---

## 📊 TABLA RESUMEN - TODAS LAS MÉTRICAS POR CLASE

| Clase | N | 1. R<br>(Mediana) | 2. R final<br>(Estacionario) | 3. Magnitud<br>Global | 4. PSD<br>slope | 5. DFA<br>alpha | 6. MI<br>(media) | 7. Entropía<br>Shannon | 8. Correlación<br>(media) |
|:-----:|---:|:-------:|:----------:|:--------:|:--------:|:-------:|:--------:|:----------:|:------------:|
| **0** | 980 | **0.5419** | 0.5883 | 0.4342 | -4.0787 | 1.9564 | 2.1390 | 3.1913 | -0.2666 |
| **1** | 1136 | **0.6891** | 0.7449 | 0.4265 | -3.7177 | 1.9211 | 2.0990 | 3.1700 | -0.3504 |
| **2** | 1032 | **0.5758** | 0.6160 | 0.4322 | -4.1853 | 1.9404 | 2.1364 | 3.1910 | -0.2689 |
| **3** | 1010 | **0.5824** | 0.6191 | 0.4313 | -4.2233 | 1.9377 | 2.1361 | 3.1941 | -0.2709 |
| **4** | 982 | **0.6297** | 0.6574 | 0.4291 | -4.3104 | 1.9362 | 2.1200 | 3.1896 | -0.2624 |
| **5** | 892 | **0.6042** | 0.6275 | 0.4304 | -4.2302 | 1.9351 | 2.1310 | 3.1920 | -0.2734 |
| **6** | 958 | **0.5957** | 0.6317 | 0.4317 | -4.1103 | 1.9326 | 2.1297 | 3.1948 | -0.2629 |
| **7** | 1028 | **0.6488** | 0.6732 | 0.4273 | -4.3337 | 1.9281 | 2.1122 | 3.1870 | -0.2643 |
| **8** | 974 | **0.5725** | 0.6131 | 0.4326 | -4.0583 | 1.9432 | 2.1451 | 3.1944 | -0.2753 |
| **9** | 1010 | **0.6275** | 0.6582 | 0.4298 | -4.2415 | 1.9288 | 2.1293 | 3.1965 | -0.2622 |

---

## 🎯 ANÁLISIS POR MÉTRICA

### 1. **R (Parámetro de Orden) - Mediana de Series Temporales**

**Valores de Referencia Teóricos**:
- R ≈ 0.0: Desorden completo
- R ≈ 0.5: **Estado crítico** (transición orden-desorden)
- R ≈ 1.0: Sincronización completa

**Ranking por Cercanía a Criticalidad (R ≈ 0.5)**:

| Pos | Clase | R (Mediana) | Distancia de 0.5 | Interpretación |
|:---:|:-----:|:-----------:|:----------------:|----------------|
| 🥇 **1** | **0** | **0.5419** | **0.0419** | ⭐ **MÁS CRÍTICO** |
| 🥈 2 | 8 | 0.5725 | 0.0725 | Muy cercano |
| 🥉 3 | 2 | 0.5758 | 0.0758 | Muy cercano |
| 4 | 3 | 0.5824 | 0.0824 | Cercano |
| 5 | 6 | 0.5957 | 0.0957 | Cercano |
| 6 | 5 | 0.6042 | 0.1042 | Moderadamente alejado |
| 7 | 9 | 0.6275 | 0.1275 | Alejado |
| 8 | 4 | 0.6297 | 0.1297 | Alejado |
| 9 | 7 | 0.6488 | 0.1488 | Muy alejado (ordenado) |
| 10 | 1 | 0.6891 | 0.1891 | Extremo (muy ordenado) |

**Conclusión**: **Clase 0** muestra el comportamiento más crítico según R.

---

### 2. **R Final (Valor Estacionario)**

Valor de R en el último timestep (t=100), indicando estado estacionario:

| Ranking | Clase | R Final | Interpretación |
|:-------:|:-----:|:-------:|----------------|
| 1 | 0 | 0.5883 | Más cercano a equilibrio crítico |
| 2 | 8 | 0.6131 | Cerca de equilibrio |
| 3 | 2 | 0.6160 | Cerca de equilibrio |
| 4 | 3 | 0.6191 | Moderado |
| ... | ... | ... | ... |
| 10 | 1 | 0.7449 | Muy ordenado (alejado de criticalidad) |

**Observación**: R final es consistentemente mayor que R mediana, sugiriendo tendencia hacia mayor sincronización con el tiempo.

---

### 3. **Magnitud Media Global (Promedio de Osciladores)**

Magnitud promedio de todos los osciladores a través del tiempo.

**Rango observado**: 0.4265 - 0.4342 (muy homogéneo entre clases)

| Clase | Magnitud Global | Variación respecto a media |
|:-----:|:---------------:|:--------------------------:|
| 8 | 0.4326 | +0.6% |
| 0 | 0.4342 | +0.9% (máximo) |
| 2 | 0.4322 | +0.5% |
| 1 | 0.4265 | -0.8% (mínimo) |

**Conclusión**: Poca diferenciación entre clases en esta métrica. Todas rondan ~0.43, indicando comportamiento dinámico similar en magnitud.

---

### 4. **PSD Slope (Pendiente Espectral)**

**Valor teórico crítico**: -1.0 (ruido rosa / 1/f)

**Ranking por Cercanía a -1.0**:

| Pos | Clase | PSD Slope | Distancia de -1.0 | Interpretación |
|:---:|:-----:|:---------:|:-----------------:|----------------|
| 1 | 1 | -3.7177 | 2.7177 | Menos alejado |
| 2 | 0 | -4.0787 | 3.0787 | Alejado |
| 3 | 8 | -4.0583 | 3.0583 | Alejado |
| ... | ... | ... | ... | ... |
| 10 | 7 | -4.3337 | 3.3337 | Más alejado |

⚠️ **PROBLEMA IDENTIFICADO**: Todos los valores están entre -3.7 y -4.3, **MUY LEJOS del valor crítico -1.0**.

**Interpretación**:
- Pendientes muy negativas (< -3) indican **rápida caída en altas frecuencias**
- Sugiere **sobre-amortiguamiento** o **dinámica muy persistente**
- NO corresponde a ruido rosa crítico
- **Acción requerida**: Ajustar parámetros del modelo (Fase 2.3)

---

### 5. **DFA Alpha (Exponente de Fluctuación sin Tendencia)**

**Valor teórico crítico**: α ≈ 1.0 (correlaciones de largo alcance críticas)

**Ranking por Cercanía a 1.0**:

| Pos | Clase | DFA α | Distancia de 1.0 | Interpretación |
|:---:|:-----:|:-----:|:----------------:|----------------|
| 1 | 7 | 1.9281 | 0.9281 | Menos alejado |
| 2 | 9 | 1.9288 | 0.9288 | Menos alejado |
| 3 | 1 | 1.9211 | 0.9211 | Menos alejado |
| ... | ... | ... | ... | ... |
| 10 | 0 | 1.9564 | 0.9564 | Más alejado |

⚠️ **PROBLEMA CRÍTICO**: Todos los valores α ≈ 1.92-1.96, **MUY LEJOS del valor crítico 1.0**.

**Interpretación**:
- α ≈ 1.93 indica **comportamiento superdifusivo** extremo
- Correlaciones de largo alcance **demasiado fuertes**
- **NO es comportamiento crítico** (esperaríamos α ≈ 1.0 o α ≈ 1.5)
- α ≈ 2.0 → Proceso tipo Browniano integrado (acumulativo)
- **Causa probable**: Parámetros del modelo generan dinámicas muy persistentes
- **Acción CRÍTICA**: Reajustar parámetros en Fase 2.3

---

### 6. **Mutual Information (MI) - Promedio entre Canales**

Mide **dependencias no lineales** entre canales de osciladores.

**Rango observado**: 2.099 - 2.145 (moderadamente alto)

| Ranking | Clase | MI (mediana) | Interpretación |
|:-------:|:-----:|:------------:|----------------|
| 1 | 8 | 2.1451 | Mayor interdependencia |
| 2 | 0 | 2.1390 | Alta interdependencia |
| 3 | 2 | 2.1364 | Alta interdependencia |
| ... | ... | ... | ... |
| 10 | 1 | 2.0990 | Menor interdependencia |

**Interpretación**:
- Valores MI > 2.0 indican **fuerte acoplamiento no lineal** entre canales
- Clase 8 muestra mayor información mutua entre osciladores
- Clase 1 (más sincronizada) tiene menor MI → sincronización puede reducir variabilidad

---

### 7. **Entropía de Shannon (Promedio por Canal)**

Mide **complejidad/desorden** en las dinámicas de cada canal.

**Rango observado**: 3.187 - 3.197 (muy estrecho)

| Clase | Entropía Shannon | Variación |
|:-----:|:----------------:|:---------:|
| 9 | 3.1965 | +0.2% (máximo) |
| 6 | 3.1948 | +0.1% |
| 8 | 3.1944 | +0.1% |
| ... | ... | ... |
| 1 | 3.1700 | -0.6% (mínimo) |

**Interpretación**:
- Entropía muy homogénea entre clases (~3.19 bits)
- Poca diferenciación por esta métrica
- Clase 1 (más sincronizada) tiene ligeramente menor entropía
- Valores altos sugieren comportamiento razonablemente complejo

---

### 8. **Correlación (Promedio entre Canales)**

Correlación **lineal** promedio entre pares de canales.

**Observación CRÍTICA**: ⚠️ **Todas las correlaciones son NEGATIVAS**

| Ranking | Clase | Correlación | Interpretación |
|:-------:|:-----:|:-----------:|----------------|
| 1 | 1 | **-0.3504** | Más anticorrelada |
| 2 | 8 | -0.2753 | Anticorrelada |
| 3 | 5 | -0.2734 | Anticorrelada |
| ... | ... | ... | ... |
| 10 | 9 | -0.2622 | Menos anticorrelada |

**Interpretación IMPORTANTE**:
- Correlaciones negativas sugieren **oscilaciones en contrafase** entre canales
- Puede ser efecto de la arquitectura del modelo Kuramoto con 4 canales
- **NO es necesariamente problemático**, pero es un patrón interesante
- Clase 1 (más sincronizada en R) tiene correlaciones más negativas → sincronización global NO implica correlación lineal local

---

## 🎯 RANKING FINAL DE CRITICALIDAD

Basado en las 3 métricas más relevantes para criticalidad:

### Criticalidad según **R (Parámetro de Orden)**:
1. 🥇 **Clase 0** (R=0.5419) - ⭐ **MÁS CRÍTICO**
2. 🥈 Clase 8 (R=0.5725)
3. 🥉 Clase 2 (R=0.5758)

### Criticalidad según **DFA Alpha**:
⚠️ **NINGUNA CLASE ES CRÍTICA** (todas α ≈ 1.93, muy lejos de 1.0)

### Criticalidad según **PSD Slope**:
⚠️ **NINGUNA CLASE ES CRÍTICA** (todas slope ≈ -4.0, muy lejos de -1.0)

---

## 🔬 CONCLUSIONES GENERALES

### ✅ Métricas que Discriminan Bien:
1. **R (Parámetro de Orden)**: Rango 0.54-0.69, clara diferenciación
2. **DFA Alpha**: Pequeñas diferencias (1.92-1.96) pero consistentes
3. **PSD Slope**: Rango -3.7 a -4.3, diferencias moderadas

### ⚠️ Métricas Poco Discriminativas:
1. **Magnitud Global**: Muy homogénea (~0.43)
2. **Entropía Shannon**: Extremadamente homogénea (~3.19)

### ❌ Problemas Metodológicos Detectados:

1. **DFA α ≈ 1.93 (NO crítico)**:
   - Esperado: α ≈ 1.0 (ruido rosa)
   - Observado: α ≈ 1.93 (superdifusivo)
   - **Causa**: Parámetros del modelo generan dinámicas muy persistentes
   - **Solución**: Fase 2.3 - Ajustar T_steps, init_omg, ksize

2. **PSD Slope ≈ -4.0 (NO crítico)**:
   - Esperado: slope ≈ -1.0 (1/f)
   - Observado: slope ≈ -4.0 (caída muy rápida)
   - **Causa**: Posible sobre-amortiguamiento
   - **Solución**: Ajustar parámetros temporales

3. **Correlaciones Negativas Generalizadas**:
   - Todas las clases muestran correlación negativa entre canales
   - Puede ser artefacto de arquitectura de 4 canales
   - Requiere análisis más profundo

---

## 🚀 PRÓXIMOS PASOS

### Inmediatos:
1. ✅ Documentar estos hallazgos (este archivo)
2. ⏳ Commitear cambios al repositorio
3. ⏳ Generar visualizaciones comparativas

### Fase 2.2 - Redes Funcionales:
- Analizar estructura de matrices MI y Correlación
- Identificar patrones de conectividad por clase
- Buscar comunidades y hubs

### Fase 2.3 - Optimización CRÍTICA:
⚠️ **PRIORITARIO**: Ajustar parámetros para acercar:
- DFA α hacia 1.0
- PSD slope hacia -1.0
- Grid search sobre T_steps, init_omg, ksize

### Fase 2.4 - Validación:
- Training set (60,000 imágenes) en otra máquina
- Comparar resultados con test set
- Confirmar robustez de hallazgos

---

## 📊 ARCHIVOS GENERADOS

- `metricas_completas_por_imagen.csv` (10,002 filas × 19 columnas)
- `estadisticas_por_clase_COMPLETO.csv` (10 clases × 24 estadísticas)
- Este documento: `ANALISIS_COMPLETO_TODAS_METRICAS.md`

---

**Fecha de análisis**: Octubre 20, 2025 - 14:45  
**Analista**: Análisis automatizado con Python  
**Próxima revisión**: Después de Fase 2.3 (optimización de parámetros)
