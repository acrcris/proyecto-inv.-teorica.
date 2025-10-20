"""Submódulo de análisis de criticalidad y series temporales."""
from .criticalidad import (
    KuramotoMetrics,
    Entropia,
    DFA,
    PSD,
    MutualInformation,
    Correlacion
)
from .series_temporales import SeriesAnalysis

__all__ = [
    'KuramotoMetrics',
    'Entropia',
    'DFA',
    'PSD',
    'MutualInformation',
    'Correlacion',
    'SeriesAnalysis'
]
