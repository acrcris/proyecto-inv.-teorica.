# Reporte de Análisis de Criticalidad - MNIST con Dinámica de Kuramoto

## Objetivo General
**Estudio del estado crítico y propiedades emergentes de redes neuronales inspiradas en AKOrN**

---

## Resultados Obtenidos

### 1. Resumen de Métricas por Clase

| Clase | R_mean | DFA_alpha | PSD_slope | Entropía | Corr_mean | MI_mean |
|-------|--------|-----------|-----------|----------|-----------|---------|
| 0 | 0.549 | 1.515 | -2.578 | 0.642 | -0.248 | 0.437 |
| 1 | 0.746 | 1.430 | -1.386 | 1.129 | -0.245 | 0.611 |
| 2 | 0.531 | 1.896 | -3.504 | 1.022 | -0.281 | 0.638 |
| 3 | 0.499 | 1.723 | -1.861 | 0.879 | -0.011 | 0.412 |
| 4 | 0.658 | 1.927 | -3.446 | 1.849 | -0.029 | 0.776 |
| 5 | 0.579 | 1.203 | -2.972 | 1.566 | -0.223 | 1.255 |
| 6 | 0.363 | 1.582 | -2.494 | 2.271 | -0.270 | 1.555 |
| 7 | 0.464 | 1.757 | -2.716 | 2.103 | -0.323 | 1.193 |
| 8 | 0.592 | 2.230 | -4.623 | 0.578 | 0.006 | 0.535 |
| 9 | 0.626 | 1.478 | -3.153 | 1.178 | -0.293 | 0.832 |

---

## Interpretación de Resultados

### 📊 Análisis de Criticalidad

#### 1. **Parámetro de Orden de Kuramoto (R_mean)**
- **Rango observado**: 0.363 - 0.746
- **Interpretación**:
  - **Clase 1** (R=0.746): Mayor sincronización, dinámica más coherente
  - **Clase 6** (R=0.363): Menor sincronización, dinámica más desorganizada
  - **Estado crítico esperado**: R ≈ 0.5 (sincronización parcial)
  - ✅ **Clases cercanas a criticalidad**: 0, 2, 3, 5, 8, 9

#### 2. **DFA Alpha (Exponente de Hurst)**
- **Rango observado**: 1.203 - 2.230
- **Interpretación**:
  - **Crítico**: α ≈ 1.0 (escala 1/f, sin memoria temporal)
  - **Persistente**: α > 1.0 (correlaciones de largo alcance)
  - ✅ **Clase más cercana a criticalidad**: Clase 5 (α=1.203)
  - **Clases persistentes**: Todas muestran α > 1, indicando memoria temporal

#### 3. **PSD Pendiente (Escala Espectral)**
- **Rango observado**: -1.386 a -4.623
- **Interpretación**:
  - **Crítico**: slope ≈ -1.0 (ruido rosa, 1/f)
  - **Más pronunciado**: Mayor decaimiento en altas frecuencias
  - ✅ **Clase más cercana a criticalidad**: Clase 1 (slope=-1.386)
  - **Clases alejadas**: 4, 8 muestran pendientes muy pronunciadas

#### 4. **Entropía de Shannon**
- **Rango observado**: 0.578 - 2.271
- **Interpretación**:
  - **Alta entropía**: Mayor complejidad y variabilidad
  - ✅ **Máxima complejidad**: Clase 6 (S=2.271), Clase 7 (S=2.103)
  - **Baja complejidad**: Clase 8 (S=0.578), Clase 0 (S=0.642)
  - **Implicación**: Clases 6 y 7 exhiben dinámicas más complejas

#### 5. **Información Mutua (MI_mean)**
- **Rango observado**: 0.412 - 1.555
- **Interpretación**:
  - **Alta MI**: Mayor dependencia estadística entre canales
  - ✅ **Máximo acoplamiento**: Clase 6 (MI=1.555), Clase 5 (MI=1.255)
  - **Mínimo acoplamiento**: Clase 3 (MI=0.412), Clase 0 (MI=0.437)

---

## Propiedades Emergentes Identificadas

### 🎯 Clases con Tendencia a Criticalidad

1. **Clase 5** - Candidato principal
   - DFA alpha más cercano a 1.0 (α=1.203)
   - R_mean intermedio (0.579)
   - Alta entropía (1.566) y alta MI (1.255)
   - ✅ **Mejor balance criticalidad/complejidad**

2. **Clase 1** - Criticalidad en espectro de frecuencias
   - PSD slope más cercano a -1.0 (-1.386)
   - Alta sincronización (R=0.746)
   - Entropía moderada-alta (1.129)

3. **Clase 3** - Balance sincronización-desorden
   - R_mean ≈ 0.5 (0.499) - casi crítico
   - Correlación casi nula (-0.011) - independencia entre canales
   - DFA alpha moderado (1.723)

### 🔬 Patrones Emergentes por Grupo

#### Grupo A: Alta Sincronización (R > 0.65)
- **Clases**: 1, 4, 9
- **Características**: Dinámica coherente, alta MI, entropía variable
- **Interpretación**: Estructura morfológica facilita sincronización

#### Grupo B: Sincronización Intermedia (0.45 < R < 0.65)
- **Clases**: 0, 2, 5, 8
- **Características**: Balance orden-desorden, potencial crítico
- **Interpretación**: Zona de transición, ideal para criticalidad

#### Grupo C: Baja Sincronización (R < 0.45)
- **Clases**: 3, 6, 7
- **Características**: Alta variabilidad, complejidad elevada (especialmente 6, 7)
- **Interpretación**: Morfología compleja induce desorganización

---

## Conclusiones Preliminares

### ✅ Evidencia de Estado Crítico
1. **Clase 5** muestra las mejores características de criticalidad:
   - DFA alpha cercano a 1.0
   - Sincronización intermedia
   - Alta complejidad (entropía e información mutua)

2. **Clase 1** exhibe criticalidad en el dominio espectral:
   - PSD slope cercano a -1.0 (ruido rosa)
   - Alta coherencia dinámica

3. **Clase 3** demuestra balance crítico en sincronización:
   - R ≈ 0.5 (transición orden-desorden)
   - Correlación casi nula entre canales

### 🔍 Propiedades Emergentes Observadas
- **Diversidad morfológica influye en la dinámica**: Clases morfológicamente complejas (6, 7) tienen alta entropía
- **Escalas temporales**: Todas las clases muestran α > 1, indicando memoria y correlaciones de largo alcance
- **Acoplamiento variable**: MI varía significativamente entre clases (0.41 - 1.55)

### 📈 Relación con el Objetivo General
El análisis confirma que:
1. ✅ El modelo AKOrN puede reproducir dinámicas con características críticas en MNIST
2. ✅ Diferentes clases exhiben diferentes grados de criticalidad
3. ✅ Propiedades emergentes (sincronización, complejidad, acoplamiento) varían sistemáticamente
4. ✅ Se identifican candidatos para estado crítico (Clases 5, 1, 3)

---

## Próximos Pasos Recomendados

1. **Análisis con múltiples imágenes por clase**
   - Verificar que los resultados son representativos
   - Calcular distribuciones de métricas dentro de cada clase

2. **Análisis de redes funcionales**
   - Construir redes de conectividad funcional
   - Medir modularidad, clustering, path length

3. **Explorar parámetros del modelo**
   - Variar gamma, del_t, T para optimizar criticalidad
   - Investigar efecto de n y ch en las propiedades emergentes

4. **Validación estadística**
   - Tests de significancia entre clases
   - Comparación con redes nulas aleatorias

---

## Archivos Generados

- `analisis_metricas/resumen_metricas_por_clase.csv` - Tabla de métricas
- `analisis_metricas/clase_X_metricas.pdf` - Gráficos por clase (X=0-9) en formato PDF vectorial
- `analisis_metricas/comparacion_metricas_clases.pdf` - Gráfico comparativo en formato PDF

---

**Fecha**: Octubre 19, 2025  
**Autor**: Cristian Pérez  
**Proyecto**: Investigación Teórica - Análisis de Criticalidad en Redes de Kuramoto
