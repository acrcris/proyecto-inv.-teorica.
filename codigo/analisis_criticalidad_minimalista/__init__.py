"""
Módulo de análisis de criticalidad minimalista.

Este módulo proporciona herramientas para:
- Simulación de dinámica de Kuramoto (kuramoto/)
- Carga de datos MNIST (datasets/)
- Segmentación por clustering (segmentacion/)
- Métricas de criticalidad (analisis/)
- Visualización y animaciones (utils/)
"""

__version__ = "0.1.0"

# Importaciones principales
from .kuramoto.modelo import KBlock, KConv2d, ModReLU, Reshape
from .kuramoto.modelo import reshape, reshape_back, nrm, gaussian_kernel_2d

from .datasets.loader import MNISTLoader

from .segmentacion.clustering import SegmentadorKMeans

from .analisis.criticalidad import (
    KuramotoMetrics, 
    Entropia, 
    DFA, 
    PSD, 
    MutualInformation, 
    Correlacion
)
from .analisis.series_temporales import SeriesAnalysis

from .utils.visualizacion import Visualizador, Animaciones

__all__ = [
    # Modelo Kuramoto
    'KBlock', 'KConv2d', 'ModReLU', 'Reshape',
    'reshape', 'reshape_back', 'nrm', 'gaussian_kernel_2d',
    
    # Datasets
    'MNISTLoader',
    
    # Segmentación
    'SegmentadorKMeans',
    
    # Análisis
    'KuramotoMetrics', 'Entropia', 'DFA', 'PSD', 
    'MutualInformation', 'Correlacion', 'SeriesAnalysis',
    
    # Visualización
    'Visualizador', 'Animaciones'
]
