# Gráficas de Distribución por Clase

Análisis de distribuciones de métricas de criticidad para cada clase del dataset MNIST test, procesado con modelo de osciladores de Kuramoto.

## 📁 Estructura de Directorios

```
graficas_por_clase/
├── clase_0/     (8 PDFs + README.md)
├── clase_1/     (8 PDFs + README.md)
├── clase_2/     (8 PDFs + README.md)
├── clase_3/     (8 PDFs + README.md)
├── clase_4/     (8 PDFs + README.md)
├── clase_5/     (8 PDFs + README.md)
├── clase_6/     (8 PDFs + README.md)
├── clase_7/     (8 PDFs + README.md)
├── clase_8/     (8 PDFs + README.md)
└── clase_9/     (8 PDFs + README.md)
```

**Total**: 80 archivos PDF + 10 archivos README.md

## 📊 Métricas Analizadas

Cada carpeta de clase contiene 8 gráficas PDF, una por métrica:

### 1. **R_median** - Parámetro de Orden (mediana temporal)
- **Objetivo crítico**: R ≈ 0.5
- **Interpretación**: Mide la sincronización global de osciladores
- **Criticidad**: Valores cercanos a 0.5 indican régimen crítico

### 2. **R_final** - Parámetro de Orden (valor estacionario)
- **Interpretación**: Estado final de sincronización
- **Relación**: Indica estabilidad del sistema

### 3. **Global_median** - Magnitud Global
- **Interpretación**: Amplitud media de oscilaciones
- **Relación**: Relacionada con la coherencia del sistema

### 4. **PSD_slope** - Pendiente del Espectro de Potencias
- **Objetivo crítico**: slope ≈ -1.0 (ruido 1/f)
- **Interpretación**: Caracteriza las fluctuaciones temporales
- **Criticidad**: Slope = -1 indica criticalidad auto-organizada

### 5. **DFA_alpha** - Exponente DFA (Detrended Fluctuation Analysis)
- **Objetivo crítico**: α ≈ 1.0
- **Interpretación**: Mide correlaciones de largo alcance
- **Criticidad**: α = 1.0 indica ruido 1/f (crítico)

### 6. **MI_median** - Información Mutua (mediana entre canales)
- **Interpretación**: Dependencias no-lineales entre canales
- **Rango**: [0, ∞), valores altos indican alta dependencia

### 7. **Entropy_median** - Entropía de Shannon (mediana por canal)
- **Interpretación**: Complejidad de las señales
- **Rango**: [0, log₂(N)], valores altos indican mayor complejidad

### 8. **Corr_median** - Correlación (mediana entre canales)
- **Interpretación**: Dependencias lineales entre canales
- **Rango**: [-1, +1]

## 📈 Contenido de cada PDF

Cada archivo PDF contiene una figura con 4 subplots:

1. **Top-Left: Histograma + KDE**
   - Distribución empírica de la métrica
   - Kernel Density Estimation (línea negra)
   - Mediana (línea roja punteada)
   - Q1 y Q3 (líneas naranjas punteadas)
   - Objetivo teórico (línea verde, si aplica)

2. **Top-Right: Q-Q Plot**
   - Test de normalidad vs distribución gaussiana
   - Test de Shapiro-Wilk (p-value)
   - Skewness (asimetría)
   - Kurtosis (curtosis)

3. **Bottom-Left: Boxplot Comparativo**
   - Distribución de la clase actual (color)
   - Distribución del resto de clases (gris)
   - Permite comparar visualmente esta clase vs resto

4. **Bottom-Right: Estadísticas Descriptivas**
   - N (número de imágenes)
   - Mediana, Media, Desv. Estándar
   - Q1, Q3, IQR (rango intercuartílico)
   - Mínimo, Máximo, Rango
   - Skewness, Kurtosis
   - Distancia al objetivo crítico (si aplica)

## 🔍 Hallazgos Principales

### Clase más Crítica (según R):
- **Clase 0**: R = 0.5419 (distancia = 0.0419)

### Rankings de Criticidad:

#### Por Parámetro R (objetivo: 0.5):
1. Clase 0: R = 0.5419
2. Clase 8: R = 0.5725
3. Clase 2: R = 0.5758
4. Clase 3: R = 0.5824
5. Clase 6: R = 0.5957

#### Por DFA α (objetivo: 1.0):
⚠️ **PROBLEMA**: Todas las clases tienen α ≈ 1.92-1.96 (muy lejos de 1.0)

#### Por PSD slope (objetivo: -1.0):
⚠️ **PROBLEMA**: Todas las clases tienen slope ≈ -3.7 a -4.3 (muy lejos de -1.0)

## ⚠️ Advertencias Importantes

1. **NO Normalidad**: Todas las distribuciones son NO normales (p < 0.05)
   - Se deben usar estadísticos no paramétricos (mediana, no media)
   - IQR en lugar de desviación estándar

2. **DFA y PSD NO críticos**: 
   - El modelo actual NO está en régimen crítico según DFA y PSD
   - Se requiere optimización de parámetros (Fase 2.3)

3. **Correlaciones Negativas**:
   - Todas las clases muestran correlaciones negativas (-0.26 a -0.35)
   - Indica relación inversa entre canales

## 📊 Datos del Análisis

- **Dataset**: MNIST test set (10,002 imágenes)
- **Distribución por clase**:
  - Clase 0: 980 imágenes
  - Clase 1: 1136 imágenes
  - Clase 2: 1032 imágenes
  - Clase 3: 1010 imágenes
  - Clase 4: 982 imágenes
  - Clase 5: 892 imágenes
  - Clase 6: 958 imágenes
  - Clase 7: 1028 imágenes
  - Clase 8: 974 imágenes
  - Clase 9: 1010 imágenes

## 🔧 Modelo Utilizado

- **Tipo**: Osciladores de Kuramoto acoplados
- **Configuración**: 4 canales (grupos de osciladores)
- **Parámetros**: T_steps, init_omg, ksize (ver documentación principal)

## 📚 Referencias

Para detalles completos del análisis, ver:
- `RESULTADOS_FASE2_1.md` - Resultados completos Fase 2.1
- `ANALISIS_COMPLETO_TODAS_METRICAS.md` - Análisis detallado de métricas
- `tests_normalidad_todas_metricas.csv` - Tests de normalidad

---

**Fecha de generación**: 2025-10-20  
**Fase del proyecto**: 2.1b (Análisis de distribuciones)  
**Estado**: ✅ Completado
