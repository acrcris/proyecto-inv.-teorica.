"""
Utilidades compartidas para análisis de alpha crítico.

Este módulo contiene funciones y configuraciones reutilizables para
evitar duplicación de código entre scripts de análisis.
"""

from .device_utils import get_device
from .alpha_utils import generate_alphas
from .image_utils import prepare_image
from .checkpoint_utils import load_checkpoint, save_checkpoint, find_checkpoints
from .plot_config import setup_matplotlib, save_figure
from .config import DATA_DIR, CHECKPOINTS_DIR, DISTRIBUTIONS_DIR, RESULTS_DB, ModelConfig

__all__ = [
    # Device utilities
    'get_device',
    
    # Alpha utilities
    'generate_alphas',
    
    # Image utilities
    'prepare_image',
    
    # Checkpoint utilities
    'load_checkpoint',
    'save_checkpoint',
    'find_checkpoints',
    
    # Plot utilities
    'setup_matplotlib',
    'save_figure',
    
    # Configuration
    'DATA_DIR',
    'CHECKPOINTS_DIR',
    'DISTRIBUTIONS_DIR',
    'RESULTS_DB',
    'ModelConfig',
]
