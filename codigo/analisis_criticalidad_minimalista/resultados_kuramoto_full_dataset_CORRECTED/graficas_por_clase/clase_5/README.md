# Gráficas de Distribución - Clase 5

Distribuciones de todas las métricas de criticidad para imágenes de la clase 5 del dataset MNIST test.

## 📊 Archivos disponibles:

### Parámetro de Orden R:
- **clase_05_R_median.pdf** - Distribución de R (mediana temporal)
  - Objetivo crítico: R ≈ 0.5
- **clase_05_R_final.pdf** - Distribución de R (valor estacionario final)

### Magnitud Global:
- **clase_05_Global_median.pdf** - Distribución de magnitud global (mediana)

### Criticidad Temporal:
- **clase_05_PSD_slope.pdf** - Pendiente del espectro de potencias
  - Objetivo crítico: slope ≈ -1.0
- **clase_05_DFA_alpha.pdf** - Exponente de análisis de fluctuaciones
  - Objetivo crítico: α ≈ 1.0

### Conectividad Funcional:
- **clase_05_MI_median.pdf** - Información mutua entre canales (mediana)
- **clase_05_Corr_median.pdf** - Correlación entre canales (mediana)

### Complejidad:
- **clase_05_Entropy_median.pdf** - Entropía de Shannon por canal (mediana)

## 📈 Contenido de cada PDF:

Cada gráfica contiene 4 subplots:

1. **Histograma + KDE**: Distribución con kernel density estimation
   - Mediana (línea roja punteada)
   - Q1 y Q3 (líneas naranjas)
   - Objetivo teórico (línea verde, si aplica)

2. **Q-Q Plot**: Test de normalidad
   - Test de Shapiro-Wilk (p-value)
   - Skewness (asimetría)
   - Kurtosis (curtosis)

3. **Boxplot Comparativo**: 
   - Distribución de esta clase vs resto de clases
   - Mediana, cuartiles, outliers

4. **Estadísticas Descriptivas**:
   - N (número de imágenes)
   - Mediana, Media, Desviación estándar
   - Q1, Q3, IQR
   - Mínimo, Máximo, Rango
   - Distancia al objetivo (si aplica)

## 🔬 Dataset:
- **Fuente**: MNIST test set
- **Clase**: 5
- **Modelo**: Osciladores de Kuramoto acoplados

---
Generado: 2025-10-20 15:41:41
