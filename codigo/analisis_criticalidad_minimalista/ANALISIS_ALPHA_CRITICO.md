# Análisis de Alpha Crítico por Clase

## Descripción

Este análisis calcula el **valor crítico de alpha (α_c)** para cada imagen del dataset MNIST.

### ¿Qué es alpha_c?

- **alpha**: Factor que escala el campo de acoplamiento externo `c` en el modelo de Kuramoto
  - `c = imagen × alpha`
  - Controla la fuerza de la influencia externa sobre la dinámica de sincronización

- **alpha_c**: Valor de alpha donde ocurre el **máximo cambio** en el parámetro de orden R
  - Se calcula como: `argmax|dR/dα|`
  - Representa el punto de transición más abrupta en la sincronización
  - Similar al punto crítico en transiciones de fase

### Metodología

1. **Por cada imagen**:
   - Barre valores de alpha en rango [0.0, 2.0] con paso 0.05
   - Para cada alpha:
     - Construye campo `c = imagen × alpha`
     - Ejecuta dinámica de Kuramoto por T=80 pasos
     - Calcula `R(t)` (parámetro de orden)
     - Promedia últimos 5 valores: `R_mean = mean(R[-5:])`
   - Construye curva `R(alpha)`
   - Suaviza con media móvil (ventana=3)
   - Calcula derivada: `dR/dα`
   - Extrae: `alpha_c = argmax|dR/dα|`

2. **Por clase (0-9)**:
   - Agrupa alpha_c por clase
   - Calcula estadísticos: mediana, media, desviación estándar
   - **Test de normalidad**:
     - Si n ≥ 8: D'Agostino-Pearson (normaltest)
     - Si 3 ≤ n < 8: Shapiro-Wilk
     - Si n < 3: NaN
   - Genera visualizaciones:
     - Histograma (si n ≥ 5)
     - QQ-plot (si n ≥ 8)

## Archivos Generados

```
resultados_alpha/
├── resultados_alpha_por_imagen.csv          # Alpha_c por cada imagen
│   Columnas: idx, label, alpha_c, R_mean
│
├── resumen_por_clase.csv                    # Estadísticos por clase
│   Columnas: label, n, median, mean, std, normaltest_pvalue, test_used
│
└── plots/
    ├── hist_alpha_c_class_0.png            # Histogramas por clase
    ├── hist_alpha_c_class_1.png
    ├── ...
    ├── qq_alpha_c_class_0.png              # QQ-plots por clase
    └── qq_alpha_c_class_1.png
```

## Uso

### Ejecución Básica

```bash
cd codigo/analisis_criticalidad_minimalista
source ../.venv/bin/activate
python3 analizar_alpha_por_clase.py
```

### Parámetros Configurables

```bash
python3 analizar_alpha_por_clase.py \
  --alpha_min 0.0 \
  --alpha_max 2.0 \
  --alpha_step 0.05 \
  --T_steps 80 \
  --R_last 5 \
  --max_images 10002 \
  --gamma 0.7 \
  --del_t 0.9
```

**Parámetros principales**:
- `--alpha_min/max/step`: Rango de barrido de alpha
- `--T_steps`: Pasos de integración temporal por alpha
- `--R_last`: Número de pasos finales para promediar R
- `--max_images`: Límite de imágenes (None = todas)
- `--gamma`: Factor de acoplamiento del modelo
- `--del_t`: Paso temporal de integración

### Monitoreo de Progreso

Si ejecutaste en background:

```bash
# Ver log en tiempo real
tail -f analisis_alpha_full.log

# Verificar proceso activo
ps aux | grep analizar_alpha

# Ver archivo parcial (se guarda cada 100 imágenes)
wc -l resultados_alpha/resultados_alpha_por_imagen_partial.csv
```

### Tiempo Estimado

**Con CPU** (Intel i7-11700):
- Por imagen: ~0.6-0.8 segundos (41 alphas × 80 pasos)
- 10,002 imágenes: **~2-2.5 horas**

**Con GPU M3** (Metal):
- Por imagen: ~0.15-0.2 segundos
- 10,002 imágenes: **~30-40 minutos**

## Interpretación de Resultados

### Alpha_c y Criticalidad

- **Alpha_c bajo (< 0.5)**: 
  - La imagen requiere poca influencia externa para cambiar sincronización
  - Podría indicar estructuras con patrones intrínsecos fuertes

- **Alpha_c medio (0.5 - 1.5)**:
  - Balance entre estructura interna y externa
  - Zona crítica típica

- **Alpha_c alto (> 1.5)**:
  - Requiere fuerte campo externo para modificar dinámica
  - Posible resistencia a sincronización

### Tests de Normalidad

**Hipótesis nula (H₀)**: Los datos provienen de distribución normal

- **p > 0.05**: No se rechaza H₀ → Distribución **compatible con normal**
- **p ≤ 0.05**: Se rechaza H₀ → Distribución **NO normal**

**En sistemas críticos**: Se espera que alpha_c **NO** siga distribución normal si:
- Hay heterogeneidad estructural por clase
- Existen múltiples regímenes de sincronización
- Hay transiciones de fase de primer orden

**Si fuera normal**: Sugiere que el campo externo afecta homogéneamente a todas las imágenes de una clase.

### QQ-Plots

- **Puntos alineados con diagonal**: Distribución normal
- **Colas pesadas arriba**: Valores extremos más frecuentes que normal
- **Colas pesadas abajo**: Menos valores extremos que normal
- **Forma de S**: Asimetría o sesgo

## Conexión con Otras Métricas

Este análisis complementa los análisis previos:

1. **R_final** (run_kuramoto_full_dataset.py):
   - R_final es el valor estacionario con parámetros fijos
   - Alpha_c indica sensibilidad de R a campo externo

2. **DFA y PSD** (analisis distribuciones):
   - DFA/PSD analizan correlaciones temporales
   - Alpha_c caracteriza respuesta a perturbaciones externas

3. **Optimización de parámetros** (optimizar_parametros_secuencial.py):
   - Optimiza T_steps, gamma, del_t para alcanzar criticalidad
   - Alpha_c es propiedad emergente que podría correlacionar con score de criticalidad

## Próximos Análisis Sugeridos

1. **Correlación alpha_c con otras métricas**:
   ```python
   # ¿Alpha_c correlaciona con R_final, DFA_alpha, PSD_slope?
   df_alpha = pd.read_csv('resultados_alpha/resultados_alpha_por_imagen.csv')
   df_metrics = torch.load('metricas_completas_CORRECTED.pt')
   # Merge y calcular correlaciones de Spearman
   ```

2. **Alpha_c vs complejidad visual**:
   - ¿Imágenes más complejas tienen alpha_c diferente?
   - Calcular entropía de pixel, varianza, etc.

3. **Curvas R(alpha) por clase**:
   - Visualizar R(alpha) promedio por clase
   - Identificar si forma de curva difiere entre clases

4. **Clasificación usando alpha_c**:
   - Entrenar clasificador simple (SVM, RandomForest) solo con alpha_c
   - Comparar accuracy con modelo Kuramoto completo

## Referencias Teóricas

- **Kuramoto, Y. (1984)**: Transición de fase en osciladores acoplados
- **Strogatz, S. (2000)**: "From Kuramoto to Crawford" - teoría de sincronización
- **Pikovsky et al. (2001)**: "Synchronization: A Universal Concept in Nonlinear Sciences"

**Nota**: Este análisis es original del proyecto y no tiene precedente directo en literatura de Kuramoto estándar, ya que estamos aplicando el modelo a imágenes con campo externo variable.

---

**Última actualización**: Octubre 22, 2025  
**Script**: `analizar_alpha_por_clase.py`  
**Función auxiliar**: `KuramotoMetrics.compute_alpha_c()` en `analisis/criticalidad.py`
