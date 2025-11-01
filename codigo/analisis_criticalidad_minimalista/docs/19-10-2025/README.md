# Análisis de Criticalidad Minimalista

Módulo de Python para análisis de criticalidad en redes de osciladores de Kuramoto aplicado a datos MNIST.

## 📋 Descripción

Este módulo implementa todas las funcionalidades del notebook `kuramoto_pruebas_basico.ipynb` en una estructura modular, limpia y escalable. Incluye:

- **Modelo de Kuramoto**: Implementación completa de osciladores acoplados con dinámica temporal
- **Métricas de criticalidad**: Parámetro de orden, DFA, PSD, entropía, información mutua
- **Análisis de series temporales**: Estadísticas, correlaciones, análisis de fase
- **Visualizaciones**: Gráficos estáticos y animaciones de la evolución temporal
- **Integración con MNIST**: Carga y procesamiento de datos

## 🏗️ Estructura del Módulo

```
analisis_criticalidad_minimalista/
├── __init__.py                      # Importaciones principales
├── kuramoto/
│   ├── __init__.py
│   └── modelo.py                    # KBlock, KConv2d, ModReLU, funciones auxiliares
├── datasets/
│   ├── __init__.py
│   └── loader.py                    # MNISTLoader para carga de datos
├── segmentacion/
│   ├── __init__.py
│   └── clustering.py                # SegmentadorKMeans
├── analisis/
│   ├── __init__.py
│   ├── criticalidad.py             # Métricas: Kuramoto, Entropía, DFA, PSD, MI, Correlación
│   └── series_temporales.py        # SeriesAnalysis: estadísticas avanzadas
├── utils/
│   ├── __init__.py
│   └── visualizacion.py            # Visualizador, Animaciones
├── main.py                          # Pipeline completo para análisis de MNIST
└── ejemplo_completo.py             # Script de demostración
```

## 🚀 Instalación

### Requisitos

```bash
torch
torchvision
numpy
scipy
matplotlib
einops
scikit-learn
```

### Instalación desde el entorno existente

El módulo está diseñado para funcionar con el entorno virtual ya configurado:

```bash
source /home/crperezp/proyectos/ProyectoInvTeorica/Proyecto-Inv.-teorica./codigo/.venv/bin/activate
```

## 📖 Uso

### Ejemplo Rápido

```python
from analisis_criticalidad_minimalista import (
    KBlock, 
    KuramotoMetrics, 
    Visualizador
)
import torch

# 1. Crear modelo
kblock = KBlock(n=4, ch=4, connectivity='conv', T=100, ksize=3)

# 2. Preparar datos
x = torch.randn(1, 4, 64, 64)  # Estados iniciales
c = torch.randn(1, 4, 64, 64)  # Campo de acoplamiento

# 3. Ejecutar dinámica
x_final, xs, es = kblock(x, c, T=100, gamma=0.7, del_t=0.9, 
                         return_xs=True, return_es=True)

# 4. Analizar
R = KuramotoMetrics.order_parameter(xs)
print(f"Parámetro de orden final: {R[-1]:.4f}")

# 5. Visualizar
Visualizador.plot_energy(es)
```

### Ejemplo Completo

Ejecuta el script de demostración que replica el notebook:

```bash
cd /home/crperezp/proyectos/ProyectoInvTeorica/Proyecto-Inv.-teorica./codigo/analisis_criticalidad_minimalista
python ejemplo_completo.py
```

Este script:
1. Carga datos MNIST
2. Inicializa el modelo Kuramoto
3. Ejecuta la dinámica temporal
4. Calcula todas las métricas de criticalidad
5. Genera visualizaciones y animaciones

### Pipeline Completo para MNIST

Analiza todo el dataset MNIST agrupado por clase:

```bash
python main.py
```

Esto ejecuta el pipeline completo:
- Carga todas las imágenes de MNIST
- Agrupa por clase (0-9)
- Ejecuta dinámica de Kuramoto en cada clase
- Calcula distribuciones de métricas de criticalidad
- Genera reportes y gráficos

## 📊 Métricas Implementadas

### 1. Parámetro de Orden de Kuramoto

```python
from analisis_criticalidad_minimalista import KuramotoMetrics

R = KuramotoMetrics.order_parameter(xs, ch_pair=(0, 1))
# R(t) ∈ [0, 1]: sincronización de fase
```

### 2. DFA (Detrended Fluctuation Analysis)

```python
from analisis_criticalidad_minimalista import DFA

scales, F, alpha = DFA.dfa(series)
# alpha ≈ 1: criticalidad (escala 1/f)
# alpha > 1: persistencia
# alpha < 1: anti-persistencia
```

### 3. PSD (Power Spectral Density)

```python
from analisis_criticalidad_minimalista import PSD

f, Pxx, slope = PSD.psd_slope(series)
# slope ≈ -1: ruido 1/f (criticalidad)
```

### 4. Entropía de Shannon

```python
from analisis_criticalidad_minimalista import Entropia

S = Entropia.shannon(series, bins=30)
# Alta entropía: complejidad, variabilidad
```

### 5. Información Mutua

```python
from analisis_criticalidad_minimalista import MutualInformation

MI = MutualInformation.mutual_info(series_x, series_y)
# MI > 0: dependencia estadística entre canales
```

### 6. Correlación de Pearson

```python
from analisis_criticalidad_minimalista import Correlacion

corr_matrix = Correlacion.pearson_matrix(series)
# Matriz de correlación entre todos los canales
```

### 7. Análisis de Series Temporales

```python
from analisis_criticalidad_minimalista import SeriesAnalysis

stats = SeriesAnalysis.compute_channel_statistics(xs)
# Devuelve: medias, desviaciones, cambios relativos, correlaciones
```

## 🎨 Visualizaciones

### Gráficos Estáticos

```python
from analisis_criticalidad_minimalista import Visualizador

# Series temporales
Visualizador.plot_series(series, title='Evolución de magnitudes')

# Matriz de correlación
Visualizador.plot_matrix(corr_matrix, title='Correlación entre canales')

# Energía del sistema
Visualizador.plot_energy(energies, normalize=True)
```

### Animaciones

```python
from analisis_criticalidad_minimalista import Animaciones

# Evolución de un canal
Animaciones.animate_dynamics(xs, channel=0, filename='canal.gif')

# Magnitud total
Animaciones.animate_magnitude(xs, filename='magnitud.gif')

# Campo vectorial
Animaciones.animate_vector_field(xs, filename='campo.gif')

# Fases de todos los canales
Animaciones.animate_phase_evolution(xs, filename='fases.gif')

# Magnitudes por canal
Animaciones.animate_magnitude_evolution(xs, filename='mags.gif')
```

## 🧪 Modelo de Kuramoto

### Componentes Principales

#### KBlock
Bloque principal que integra la dinámica de Kuramoto:

```python
kblock = KBlock(
    n=4,              # Dimensión de osciladores
    ch=4,             # Número de canales
    connectivity='conv',  # Tipo de conectividad
    T=100,            # Pasos de integración
    ksize=3,          # Tamaño de kernel convolucional
    init_omg=0.1,     # Inicialización de frecuencias
    c_norm='gn',      # Normalización del campo c
    use_omega=True,   # Usar frecuencias naturales
    use_omega_c=True  # Aplicar omega al campo c
)
```

#### KConv2d
Capa convolucional que define la conectividad:

```python
kconv = KConv2d(n=4, ch=4, connectivity='conv', ksize=3)
```

#### ModReLU
Activación que preserva la dirección de fase:

```python
modrelu = ModReLU(n=4, ch=4, norm='gn')
```

### Funciones Auxiliares

```python
from analisis_criticalidad_minimalista import (
    reshape,           # Reorganiza canales en grupos de osciladores
    reshape_back,      # Restaura forma original
    nrm,              # Norma euclidiana
    gaussian_kernel_2d # Kernel gaussiano 2D
)
```

## 📈 Interpretación de Resultados

### Criticalidad
Un sistema está en estado crítico cuando:
- **R(t) → intermedio** (ni totalmente sincronizado ni desincronizado)
- **DFA alpha ≈ 1** (escala 1/f)
- **PSD slope ≈ -1** (ruido rosa)
- **Entropía alta** (complejidad máxima)

### Sincronización
- **R(t) → 1**: Osciladores sincronizados
- **R(t) → 0**: Osciladores desordenados

### Correlaciones
- **MI alta**: Dependencia fuerte entre canales
- **Correlación alta**: Covariación lineal entre magnitudes

## 🔧 Desarrollo

### Testing

Prueba componentes individuales:

```python
# Test del modelo
from analisis_criticalidad_minimalista import KBlock
import torch

kblock = KBlock(n=2, ch=4, T=10)
x = torch.randn(1, 4, 16, 16)
c = torch.randn(1, 4, 16, 16)
x_final, xs, es = kblock(x, c, T=10, gamma=0.5, del_t=1.0, 
                         return_xs=True, return_es=True)
assert len(xs) == 11  # t=0 + 10 pasos
print("✅ Test del modelo pasado")
```

### Extensiones

Para agregar nuevas métricas:

1. Agregar clase en `analisis/criticalidad.py` o `analisis/series_temporales.py`
2. Actualizar `analisis/__init__.py`
3. Actualizar `__init__.py` principal
4. Agregar tests en `ejemplo_completo.py`

## 📚 Referencias

Este módulo implementa conceptos de:

- **Kuramoto, Y.** (1984). Chemical Oscillations, Waves, and Turbulence
- **Bullmore, E. & Sporns, O.** (2009). Complex brain networks: graph theoretical analysis of structural and functional systems
- **Rubinov, M. & Sporns, O.** (2010). Complex network measures of brain connectivity

## 👥 Autor

Cristian Pérez  
Proyecto de Investigación Teórica  
2025

## 📄 Licencia

Este módulo es parte del proyecto de investigación en el repositorio `Proyecto-Inv.-teorica.`
