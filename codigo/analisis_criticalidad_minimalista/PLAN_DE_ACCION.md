# PLAN DE ACCIÓN - ANÁLISIS DE CRITICALIDAD EN AKORN

**Proyecto**: Estudio del estado crítico y propiedades emergentes de redes neuronales inspiradas en AKOrN  
**Dataset**: MNIST (Modified National Institute of Standards and Technology)  
**Fecha**: Octubre 19, 2025  
**Estado**: Fase 2.1 en ejecución

---

## 📊 ESTADO ACTUAL DEL PROYECTO

### ✅ COMPLETADO

#### Fase 1: Análisis Exploratorio (100% completado)
- [x] Migración completa de 23 funciones desde notebook a módulo
- [x] Implementación de 6 métricas de criticalidad
- [x] Ejecución sobre 10 imágenes (1 por clase)
- [x] Identificación de 3 candidatos críticos iniciales
- [x] Documentación completa (7 archivos .md)
- [x] Suite de pruebas (10/10 tests passing)
- [x] Visualizaciones en PDF (11 gráficas)

**Resultados Fase 1**:
- **Clase 5**: Mejor candidato global (DFA α=1.203)
- **Clase 1**: Criticalidad espectral (PSD slope=-1.386)
- **Clase 3**: Criticalidad en sincronización (R≈0.5)

### ✅ RECIÉN COMPLETADO

#### Fase 2.1: Análisis Estadístico Robusto (COMPLETADO - Oct 20, 2025)
- [x] Scripts creados y validados
- [x] Proceso ejecutado exitosamente (PID 7028)
- [x] **10,000 imágenes procesadas** (test set completo)
- [x] **Métricas completas generadas** (1,010,202 valores de R)
- [x] **Metodología corregida aplicada** (sin promediado prematuro)
- [ ] Análisis de distribuciones pendiente
- [ ] Análisis estadístico por clase pendiente

**Ejecución completada**:
- Inicio: 09:25 (Oct 20, 2025)
- Finalización: 14:12 (Oct 20, 2025)
- Tiempo total: ~4.8 horas
- Proceso retomado desde checkpoint_09599 a las 14:10
- Últimas 400 imágenes completadas en 2 minutos
- Tasa de éxito: 100% (10,002/10,002 imágenes)
- Sistema de checkpoints: 100 archivos generados
- Tamaño total: 2.9 GB de datos completos
- Checkpoint final: checkpoint_09999.pt (71 MB)

### 🔄 EN EJECUCIÓN

#### Fase 2.1b: Análisis de Resultados (En progreso)
- [x] Datos completos generados (2.9 GB)
- [ ] **Análisis de distribuciones temporales** (SIGUIENTE PASO)
- [ ] Validación de normalidad estadística
- [ ] Análisis estadístico por clase
- [ ] Generación de reporte final

**Estado actual**:
- Dataset completado: 10,000 imágenes del test set
- Total valores preservados: 1,010,202 observaciones temporales
- Checkpoints respaldados en GitHub: 45 checkpoints (00099-04499)
- Checkpoints locales: 55 checkpoints (04599-09999)
- Espacio en disco: 1.1 GB disponibles
- Próxima acción: Ejecutar `python analizar_distribuciones.py`

### PLAN A: Fase 2.1 - Análisis Completo MNIST ⏳ EN EJECUCIÓN

**Objetivo**: Validar estadísticamente los hallazgos de Fase 1 con 10,000 imágenes

#### Acciones Inmediatas (Próximas horas)

1. **Monitoreo del Proceso** 🔍
   ```bash
   # Verificar estado
   ps -p $(cat kuramoto_full.pid)
   
   # Ver progreso
   python monitor_progreso.py
   
   # Revisar log
   tail -f kuramoto_full.log
   ```
   
   **Responsable**: Sistema automático  
   **Tiempo**: Continuo hasta finalización  
   **Output esperado**: 10,000 métricas calculadas

2. **Verificación de Checkpoints** ✅
   ```bash
   ls -lh resultados_kuramoto_full_dataset/checkpoints/
   ```
   
   **Frecuencia**: Cada hora  
   **Criterio de éxito**: 100 checkpoints generados (checkpoint_00100.pt ... checkpoint_10000.pt)

#### Acciones Post-Procesamiento (Después de 3-6 horas)

3. **Análisis de Distribuciones Temporales** 📊 **[CRÍTICO - NUEVA METODOLOGÍA]**
   ```bash
   python analizar_distribuciones.py
   ```
   
   **⚠️ IMPORTANTE - METODOLOGÍA CORREGIDA**:
   Este script implementa la **corrección metodológica fundamental** identificada en `CORRECCION_METODOLOGICA.md`:
   
   - **NO se calcularán promedios hasta verificar normalidad**
   - Se analizan las 1,010,000 observaciones temporales (10,000 imágenes × 101 timesteps)
   - Se construye un DataFrame completo con todas las observaciones temporales
   - Se ejecutan tests de normalidad (Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling)
   - Se generan histogramas y Q-Q plots para inspección visual
   - **SOLO después de confirmar normalidad** se decidirá usar media o mediana
   
   **Tiempo estimado**: 30-45 minutos (análisis exhaustivo)  
   **Outputs esperados**:
   - `distribucion_R.pdf` - Histograma + Q-Q plot de parámetro de orden
   - `distribucion_DFA_alpha.pdf` - Histograma + Q-Q plot de DFA
   - `distribucion_PSD_slope.pdf` - Histograma + Q-Q plot de pendiente espectral
   - `distribucion_Entropy.pdf` - Histograma + Q-Q plot de entropía
   - `distribucion_Correlation.pdf` - Histograma + Q-Q plot de correlación
   - `distribucion_Mutual_Info.pdf` - Histograma + Q-Q plot de información mutua
   - `distribucion_Variance.pdf` - Histograma + Q-Q plot de varianza
   - `tests_normalidad.csv` - Resultados de todos los tests estadísticos
   - `recomendaciones_agregacion.txt` - Indicaciones sobre uso de media vs mediana
   
   **Criterios de decisión**:
   - **Si p-value > 0.05 en tests**: Distribución normal → usar **media aritmética**
   - **Si p-value < 0.05 en tests**: Distribución no-normal → usar **mediana**
   - Se considerarán también curtosis y asimetría para decisión final

4. **Análisis Estadístico por Clase** 📊
   ```bash
   python analizar_estadisticas_full_dataset.py
   ```
   
   **⚠️ NOTA**: Este script se ejecutará **DESPUÉS** del análisis de distribuciones, utilizando el método de agregación recomendado (media o mediana según normalidad).
   
   **Tiempo estimado**: 15-30 minutos  
   **Outputs esperados**:
   - `estadisticas_por_clase.csv` - Tabla completa de estadísticas (con método correcto)
   - `tests_significancia.csv` - Resultados de ANOVA o Kruskal-Wallis (según distribución)
   - `ranking_criticalidad.csv` - Ranking definitivo
   - `distribuciones_boxplot.pdf` - Box plots por métrica
   - `distribuciones_violin.pdf` - Violin plots de métricas clave
   - `medias_por_clase.pdf` - Gráficas de centralidad con barras de error
   - `resumen_ejecutivo.txt` - Interpretación científica

5. **Validación de Resultados** ✓
   - Verificar que todas las clases tienen N ≈ 1,000 imágenes
   - Confirmar tests estadísticos con p-values < 0.05
   - Verificar que se usó el estadístico correcto (media o mediana)
   - Identificar candidatos críticos robustos
   - Comparar con resultados de Fase 1

6. **Documentación de Hallazgos** 📝
   - Crear `RESULTADOS_FASE2_1.md` con hallazgos principales
   - Incluir sección sobre validación de supuestos estadísticos
   - Actualizar `OBJETIVO_GENERAL_Y_RESULTADOS.md`
   - Generar tabla comparativa Fase 1 vs Fase 2.1
   - Documentar decisiones metodológicas (media vs mediana)

**Criterios de éxito Fase 2.1**:
- ✅ 10,000 imágenes procesadas exitosamente (>95%)
- ✅ **Series temporales completas preservadas** (1,010,000 observaciones)
- ✅ **Tests de normalidad ejecutados** para todas las métricas
- ✅ **Metodología de agregación validada** (media vs mediana según distribución)
- ✅ Distribuciones completas por clase
- ✅ Tests estadísticos significativos (paramétricos o no-paramétricos según corresponda)
- ✅ Ranking definitivo de candidatos críticos
- ✅ Intervalo de confianza del 95% para todas las métricas

---

### PLAN B: Fase 2.2 - Redes Funcionales ⏭️ SIGUIENTE

**Objetivo**: Construir y analizar redes funcionales basadas en conectividad

**Prerequisito**: Fase 2.1 completada

#### Acciones Planificadas

1. **Construcción de Matrices de Conectividad**
   - Calcular matrices de correlación completas
   - Calcular matrices de información mutua
   - Generar grafos ponderados por conectividad
   
   **Métricas**: Pearson correlation, Mutual Information
   **Output**: `matrices_conectividad_clase_X.npy` (X=0-9)

2. **Análisis de Propiedades de Grafo**
   - Clustering coefficient (C)
   - Average path length (L)
   - Small-world index (σ = (C/C_rand) / (L/L_rand))
   - Modularidad (Q)
   - Distribución de grados
   
   **Librería**: NetworkX
   **Output**: `propiedades_grafo_por_clase.csv`

3. **Detección de Comunidades**
   - Algoritmo de Louvain
   - Análisis de estructura modular
   - Identificación de hubs
   
   **Output**: `comunidades_clase_X.json`

4. **Visualizaciones de Redes**
   - Grafos 2D con layout force-directed
   - Matrices de adyacencia organizadas por comunidades
   - Distribuciones de métricas de red
   
   **Output**: PDFs de visualizaciones de redes

**Tiempo estimado**: 2-3 días  
**Scripts a crear**:
- `construir_redes_funcionales.py`
- `analizar_propiedades_grafo.py`
- `visualizar_redes.py`

**Criterios de éxito**:
- ✅ Redes funcionales para las 10 clases
- ✅ Identificación de propiedades small-world
- ✅ Comparación de modularidad entre clases
- ✅ Correlación entre criticalidad y propiedades de red

---

### PLAN C: Fase 2.3 - Optimización de Parámetros ⏭️ FUTURO

**Objetivo**: Encontrar parámetros óptimos que maximicen criticalidad

**Prerequisito**: Fase 2.2 completada

#### Acciones Planificadas

1. **Grid Search de Parámetros**
   
   Parámetros a optimizar:
   - `T_steps`: [50, 100, 150, 200]
   - `ksize`: [3, 5, 7, 9]
   - `init_omg`: [0.05, 0.1, 0.15, 0.2]
   - `ch`: [2, 4, 8]
   
   **Total combinaciones**: 4×4×4×4 = 256 configuraciones
   **Muestra**: 100 imágenes por clase (para rapidez)

2. **Función Objetivo Multi-Métrica**
   ```python
   def criticality_score(metrics):
       dist_R = abs(metrics['R_mean'] - 0.5)
       dist_DFA = abs(metrics['DFA_alpha'] - 1.0)
       dist_PSD = abs(metrics['PSD_slope'] - (-1.0))
       return 1 / (dist_R + dist_DFA + dist_PSD + 1e-6)
   ```

3. **Análisis de Sensibilidad**
   - Gráficas de superficie de respuesta
   - Heatmaps de interacciones de parámetros
   - Identificación de configuraciones robustas

**Tiempo estimado**: 3-5 días (con paralelización)  
**Scripts a crear**:
- `optimizar_parametros_grid.py`
- `analisis_sensibilidad.py`

**Criterios de éxito**:
- ✅ Configuración óptima identificada
- ✅ Mejora >10% en criticality score
- ✅ Análisis de robustez completado

---

### PLAN D: Fase 2.4 - Validación Estadística Avanzada ⏭️ FUTURO

**Objetivo**: Validación rigurosa para publicación científica

**Prerequisito**: Fases 2.1, 2.2, 2.3 completadas

#### Acciones Planificadas

1. **Tests de Hipótesis Múltiples**
   - Corrección de Bonferroni
   - False Discovery Rate (FDR)
   - Tests post-hoc (Tukey HSD)

2. **Modelos Nulos**
   - Redes aleatorias (Erdős–Rényi)
   - Redes con misma distribución de grados
   - Datos shuffled temporalmente
   
   **Comparación**: Métricas reales vs modelos nulos

3. **Cross-Validation**
   - K-fold validation de hallazgos
   - Bootstrap para intervalos de confianza
   - Pruebas de robustez

4. **Análisis de Potencia**
   - Cálculo de tamaños de efecto (Cohen's d)
   - Verificación de poder estadístico >80%

**Tiempo estimado**: 2-3 días  
**Output**: Documento de validación estadística para publicación

---

## 📅 CRONOGRAMA GENERAL

```
Octubre 2025
├── ✅ Fase 1: Completada (Oct 1-18)
├── ✅ Fase 2.1: Procesamiento completado (Oct 19-20)
│   └── Finalización: Oct 20, 14:12
├── 🔄 Fase 2.1b: Análisis de resultados (Oct 20)
│   ├── Análisis distribuciones (pendiente)
│   └── Análisis estadístico (pendiente)
│
Octubre-Noviembre 2025
├── ⏭️ Fase 2.2: Redes funcionales (Oct 21-24)
├── ⏭️ Fase 2.3: Optimización (Oct 25-30)
└── ⏭️ Fase 2.4: Validación (Oct 31 - Nov 3)

Noviembre 2025
├── Fase 3: Análisis teórico (Nov 4-15)
└── Fase 4: Redacción paper (Nov 16-30)
```

---

## 🚨 CONTINGENCIAS Y RIESGOS

### Riesgo 1: Proceso de Fase 2.1 se interrumpe
**Probabilidad**: Baja (sistema de checkpoints)  
**Impacto**: Medio  
**Mitigación**:
- Sistema de checkpoints cada 100 imágenes
- Recuperación automática desde último checkpoint
- Comando: `python run_kuramoto_full_dataset.py` (detecta y reanuda)

### Riesgo 2: Resultados no muestran criticalidad clara
**Probabilidad**: Media  
**Impacto**: Alto (afecta interpretación)  
**Mitigación**:
- Análisis de subpoblaciones críticas
- Optimización de parámetros (Fase 2.3)
- Exploración de métricas alternativas

### Riesgo 3: Recursos computacionales insuficientes
**Probabilidad**: Baja  
**Impacto**: Medio  
**Mitigación**:
- Reducción de muestra si necesario (5,000 imágenes aún robusto)
- Procesamiento por lotes
- Optimización de código

---

## 📊 MÉTRICAS DE PROGRESO

### Fase 2.1 (Completada - Oct 20, 2025)
| Métrica | Objetivo | Actual | Estado |
|---------|----------|--------|--------|
| Imágenes procesadas | 10,000 | 10,002 | ✅ Completado |
| Checkpoints guardados | 100 | 100 | ✅ Completado |
| Tiempo transcurrido | 3-6 hrs | 4.8 hrs | ✅ Dentro rango |
| Errores | <5% | 0% | ✅ Excelente |
| Datos generados | ~500 MB | 2.9 GB | ✅ Completado |
| Series temporales | Preservadas | 1,010,202 | ✅ Método correcto |

### Fase 2.1b (En progreso - Análisis de resultados)
| Métrica | Objetivo | Actual | Estado |
|---------|----------|--------|--------|
| Análisis distribuciones | Completado | Pendiente | ⏳ Siguiente |
| Tests normalidad | Ejecutados | Pendiente | ⏳ Siguiente |
| Análisis por clase | Completado | Pendiente | ⏳ Siguiente |
| Reporte final | Generado | Pendiente | ⏳ Siguiente |

### Proyecto General
| Fase | Progreso | Estado |
|------|----------|--------|
| Fase 1 | 100% | ✅ Completada |
| Fase 2.1 (Procesamiento) | 100% | ✅ Completada |
| Fase 2.1b (Análisis) | 0% | 🔄 En progreso |
| Fase 2.2 | 0% | ⏭️ Pendiente |
| Fase 2.3 | 0% | ⏭️ Pendiente |
| Fase 2.4 | 0% | ⏭️ Pendiente |
| **TOTAL** | **35%** | **🔄 En progreso** |

---

## 🎯 OBJETIVOS A CORTO PLAZO (Próximas 24 horas)

1. **Análisis de Distribuciones** ⏰ Urgente - SIGUIENTE PASO
   - Ejecutar `python analizar_distribuciones.py`
   - Verificar normalidad de las 7 métricas
   - Determinar método de agregación (media vs mediana)
   - Generar visualizaciones de distribuciones

2. **Análisis Estadístico por Clase** 📊 Importante
   - Ejecutar `python analizar_estadisticas_full_dataset.py`
   - Comparar con Fase 1
   - Verificar significancia estadística
   - Identificar candidatos críticos definitivos

3. **Documentar Resultados** 📝 Importante
   - Crear documento de resultados Fase 2.1
   - Actualizar README con hallazgos
   - Preparar visualizaciones para presentación

4. **Gestión de Almacenamiento** 💾 Importante
   - Continuar respaldo de checkpoints en GitHub
   - Liberar espacio eliminando checkpoints locales respaldados
   - Mantener 1+ GB libre para análisis

---

## 📞 COMANDOS DE REFERENCIA RÁPIDA

### Monitoreo
```bash
# Ver estado del proceso
ps -p $(cat kuramoto_full.pid)

# Monitorear progreso
python monitor_progreso.py

# Ver log en tiempo real
tail -f kuramoto_full.log

# Ver checkpoints
ls -lh resultados_kuramoto_full_dataset/checkpoints/
```

### Análisis (cuando termine)
```bash
# 1. PRIMERO: Verificar normalidad de distribuciones (CRÍTICO)
python analizar_distribuciones.py

# Ver recomendaciones de agregación
cat recomendaciones_agregacion.txt

# Ver resultados de tests de normalidad
cat tests_normalidad.csv

# 2. SEGUNDO: Ejecutar análisis estadístico por clase
python analizar_estadisticas_full_dataset.py

# Ver resultados
cat resultados_kuramoto_full_dataset/analisis_estadistico/resumen_ejecutivo.txt
```

### Gestión
```bash
# Detener proceso (si necesario)
kill $(cat kuramoto_full.pid)

# Reanudar desde checkpoint
python run_kuramoto_full_dataset.py
```

---

## 📚 DOCUMENTACIÓN RELACIONADA

- `OBJETIVO_GENERAL_Y_RESULTADOS.md` - Objetivos y estado del proyecto
- `FASE2_INSTRUCCIONES.md` - Guía de ejecución Fase 2.1
- `CORRECCION_METODOLOGICA.md` - **[CRÍTICO]** Corrección de error en promediado prematuro
- `RESUMEN_CORRECCION.md` - Resumen ejecutivo de la corrección metodológica
- `REPORTE_RESULTADOS.md` - Interpretación Fase 1
- `PROGRESO_FASE1.md` - Resumen ejecutivo Fase 1
- `README.md` - Documentación del módulo
- `RESUMEN_IMPLEMENTACION.md` - Detalles técnicos

---

**Última actualización**: Octubre 20, 2025 - 14:15  
**Próxima revisión**: Octubre 20, 2025 - 18:00 (post análisis distribuciones)  
**Responsable**: Cristian Pérez  
**Estado general**: 🔄 En progreso - Fase 2.1 procesamiento completado, análisis pendiente

**Logros recientes**:
- ✅ Procesamiento completo de 10,000 imágenes (test set MNIST)
- ✅ Preservación de 1,010,202 observaciones temporales (metodología correcta)
- ✅ Sistema de checkpoints funcionando perfectamente
- ✅ Recuperación exitosa desde checkpoint_09599
- ✅ 45 checkpoints respaldados en GitHub
- ✅ Liberación de 1.1 GB de espacio en disco

**Próximos pasos inmediatos**:
1. Ejecutar análisis de distribuciones temporales
2. Validar supuestos de normalidad estadística
3. Realizar análisis estadístico por clase con método correcto
4. Generar reporte de resultados Fase 2.1

