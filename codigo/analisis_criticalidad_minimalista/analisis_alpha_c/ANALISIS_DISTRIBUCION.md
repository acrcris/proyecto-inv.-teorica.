# Análisis de Distribución de α_c: Original vs Refactorizado

## 📊 Resumen Ejecutivo

Se han completado exitosamente los análisis de criticalidad para **60,000 imágenes MNIST** en dos versiones:

- **Original**: `analizar_con_sqlite.py` → 60,000/60,000 (100%)
- **Refactorizado**: `analizar_con_sqlite_REFACTORIZADO.py` → 59,910/60,000 (99.85%)

### ⚠️ Hallazgo Crítico

**Las distribuciones de α_c son significativamente diferentes entre ambas versiones:**

```
Versión Original:      α_c_promedio = 0.001806  (σ = 0.001414)
Versión Refactorizada: α_c_promedio = 0.061177  (σ = 0.026741)

Diferencia: Δ = +0.059371 (33x más alto en refactorizado)
```

## 📈 Estadísticas por Clase

### Versión Original (60,000 imágenes)

| Clase | μ (promedio) | σ (desv. std) | n (muestras) |
|-------|-------------|---------------|------------|
| 0     | 0.0013      | 0.0005        | 5,887      |
| 1     | 0.0030      | 0.0029        | 6,779      |
| 2     | 0.0015      | 0.0006        | 6,040      |
| 3     | 0.0016      | 0.0006        | 6,121      |
| 4     | 0.0018      | 0.0014        | 5,849      |
| 5     | 0.0018      | 0.0007        | 5,442      |
| 6     | 0.0017      | 0.0014        | 5,918      |
| 7     | 0.0019      | 0.0007        | 6,232      |
| 8     | 0.0015      | 0.0014        | 5,833      |
| 9     | 0.0018      | 0.0008        | 5,899      |

**Observaciones:**
- Distribución muy concentrada (α_c ≈ 0.001-0.003)
- La clase 1 tiene el valor promedio más alto (0.0030)
- Variabilidad baja (σ < 0.003 en todas las clases)

### Versión Refactorizada (59,910 imágenes)

| Clase | μ (promedio) | σ (desv. std) | n (muestras) |
|-------|-------------|---------------|------------|
| 0     | 0.0453      | 0.0323        | 5,911      |
| 1     | 0.0673      | 0.0167        | 6,733      |
| 2     | 0.0569      | 0.0302        | 5,945      |
| 3     | 0.0592      | 0.0288        | 6,123      |
| 4     | 0.0667      | 0.0224        | 5,832      |
| 5     | 0.0645      | 0.0257        | 5,413      |
| 6     | 0.0608      | 0.0271        | 5,910      |
| 7     | 0.0677      | 0.0203        | 6,257      |
| 8     | 0.0566      | 0.0300        | 5,844      |
| 9     | 0.0659      | 0.0225        | 5,942      |

**Observaciones:**
- Distribución mucho más dispersa (α_c ≈ 0.045-0.068)
- La clase 7 tiene el valor más alto (0.0677)
- La clase 0 tiene el valor más bajo (0.0453)
- Mayor variabilidad (σ = 0.016-0.032)
- Las clases 1, 4 y 7 muestran mayor consistencia (σ < 0.023)

## 🔍 Comparación por Clase

| Clase | Δ promedio | Ratio | Interpretación |
|-------|-----------|-------|----------------|
| 0     | +0.0440   | 34.8x | Significantemente mayor |
| 1     | +0.0643   | 22.5x | Muy diferente |
| 2     | +0.0554   | 37.0x | Significantemente mayor |
| 3     | +0.0576   | 36.7x | Significantemente mayor |
| 4     | +0.0649   | 37.1x | Significantemente mayor |
| 5     | +0.0627   | 36.0x | Significantemente mayor |
| 6     | +0.0591   | 35.5x | Significantemente mayor |
| 7     | +0.0658   | 35.1x | Significantemente mayor |
| 8     | +0.0551   | 37.4x | Significantemente mayor |
| 9     | +0.0641   | 36.8x | Significantemente mayor |

**Promedio global: 34.8x más alto en refactorizado**

## 🚨 Posibles Causas de la Discrepancia

### 1. **Diferencia en Configuración de Timesteps**
- **Original**: T=100 timesteps (configuración por defecto)
- **Refactorizado**: T=50 timesteps (según README del módulo)
- **Impacto**: Menos iteraciones → dinámicas menos estables → α_c más alto

### 2. **Diferencia en Escala de Valores de Entrada**
- Posible normalización diferente de imágenes
- Escalado diferente del parámetro de control α

### 3. **Condiciones Iniciales**
- Random seed diferente → dinámicas distintas
- Aunque se implementó reproducibilidad, puede haber diferencias sutiles

### 4. **Implementación de KBlock**
- Posibles diferencias en cómo se calculan las métricas de criticalidad
- Diferentes métodos de cálculo del parámetro de orden R

## 📋 Recomendaciones

### Prioridad Alta
1. **Auditar configuración de timesteps** en ambas versiones
   - Verificar si T=100 vs T=50 es la causa principal
   
2. **Comparar escalado de imágenes MNIST**
   - Verificar normalización [0,1] vs [-1,1] vs raw values
   
3. **Revisar cálculo de α_c**
   - Asegurar que ambas versiones usan el mismo algoritmo

### Prioridad Media
4. **Validar reproducibilidad**
   - Ejecutar ambas versiones con mismos seeds de RNG
   - Comparar resultados elemento por elemento
   
5. **Documentar diferencias claras**
   - Enumerar todas las diferencias en código
   - Explicar rationales

### Prioridad Baja
6. **Análisis de sensibilidad**
   - Variar T en el rango [50, 100, 150]
   - Medir cambios en α_c promedio

## 📁 Archivos Generados

- `distribucion_clases.png` - Gráfico comparativo (violinplot + boxplot)
- `ANALISIS_DISTRIBUCION.md` - Este documento

## ✅ Conclusión

Aunque ambos procesos completaron exitosamente el análisis de 60,000 imágenes, **los resultados son significativamente diferentes**. Se requiere investigación urgente para identificar y resolver la causa de esta discrepancia antes de validar los resultados como científicamente confiables.

**Próximo paso**: Revisar la configuración exacta de ambas versiones y ejecutar pruebas controladas con subconjuntos de datos idénticos.
