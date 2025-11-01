"""
Configuración centralizada para análisis de alpha crítico.

Define constantes, paths y configuraciones del modelo Kuramoto.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ============================================================================
# PATHS Y DIRECTORIOS
# ============================================================================

# Directorio raíz del proyecto (relativo a este archivo)
ROOT_DIR = Path(__file__).parent.parent

# Paths de datos
DATA_DIR = Path("./data")
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints_impares"
DISTRIBUTIONS_DIR = ROOT_DIR / "distribuciones_impares"
RESULTS_DB = ROOT_DIR / "resultados_criticalidad.db"
TEMP_DIR = ROOT_DIR / "temp"
GRAFICAS_DIR = ROOT_DIR / "graficas_alpha_c"

# Crear directorios si no existen
for directory in [TEMP_DIR, GRAFICAS_DIR]:
    directory.mkdir(exist_ok=True)


# ============================================================================
# CONFIGURACIÓN DEL MODELO KURAMOTO
# ============================================================================

@dataclass
class ModelConfig:
    """
    Configuración de parámetros para el modelo Kuramoto.
    
    Attributes:
        gamma (float): Fuerza de acoplamiento γ.
        delta_t (float): Paso de integración temporal Δt.
        timesteps (int): Número de pasos temporales T.
        ksize (int): Tamaño de ventana convolucional.
        ch (int): Número de canales.
        img_size (int): Tamaño de imagen (HxW).
        init_omg (float): Escala de frecuencias naturales iniciales.
    """
    gamma: float = 0.7
    delta_t: float = 0.9
    timesteps: int = 100
    ksize: int = 7
    ch: int = 4
    img_size: int = 64
    init_omg: float = 0.1
    
    @classmethod
    def from_args(cls, args):
        """
        Crea una configuración desde argumentos de línea de comandos.
        
        Args:
            args: Namespace de argparse con atributos del modelo.
        
        Returns:
            ModelConfig: Configuración creada.
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in vars(args).items() if k in valid_fields}
        return cls(**kwargs)
    
    def to_dict(self):
        """Convierte la configuración a diccionario."""
        return {
            'gamma': self.gamma,
            'delta_t': self.delta_t,
            'timesteps': self.timesteps,
            'ksize': self.ksize,
            'ch': self.ch,
            'img_size': self.img_size,
            'init_omg': self.init_omg,
        }


# Configuración por defecto
DEFAULT_MODEL_CONFIG = ModelConfig()


# ============================================================================
# CONFIGURACIÓN DE ANÁLISIS DE ALPHA
# ============================================================================

@dataclass
class AlphaAnalysisConfig:
    """
    Configuración para análisis de barrido de alpha.
    
    Attributes:
        alpha_start (float): Valor inicial de alpha.
        alpha_end (float): Valor final de alpha.
        alpha_step (float): Incremento de alpha.
        threshold (float): Umbral para detectar criticalidad.
        window (int): Ventana para promediar parámetro de orden.
    """
    alpha_start: float = 0.0
    alpha_end: float = 0.1
    alpha_step: float = 0.0005
    threshold: float = 0.5
    window: int = 5
    
    @property
    def num_points(self):
        """Número de puntos en el barrido."""
        import numpy as np
        return int(np.floor((self.alpha_end - self.alpha_start) / self.alpha_step)) + 1


# Configuración por defecto para análisis
DEFAULT_ALPHA_CONFIG = AlphaAnalysisConfig()


# ============================================================================
# CONFIGURACIÓN DE VISUALIZACIÓN
# ============================================================================

@dataclass
class PlotConfig:
    """
    Configuración para visualizaciones.
    
    Attributes:
        dpi (int): Resolución de figuras.
        figsize (tuple): Tamaño por defecto de figuras.
        format (str): Formato de salida.
        style (str): Estilo de matplotlib.
    """
    dpi: int = 300
    figsize: tuple = (10, 6)
    format: str = 'png'
    style: Optional[str] = None


# Configuración por defecto para plots
DEFAULT_PLOT_CONFIG = PlotConfig()


# ============================================================================
# CONSTANTES
# ============================================================================

# Clases MNIST
MNIST_CLASSES = list(range(10))
MNIST_ODD_CLASSES = [1, 3, 5, 7, 9]
MNIST_EVEN_CLASSES = [0, 2, 4, 6, 8]

# Tamaños de dataset
MNIST_TRAIN_SIZE = 60000
MNIST_TEST_SIZE = 10000

# Configuración de SQLite
SQLITE_TIMEOUT = 30.0  # segundos
SQLITE_COMMIT_INTERVAL = 100  # commits cada N imágenes


# ============================================================================
# UTILIDADES
# ============================================================================

def get_config_summary():
    """
    Retorna un resumen legible de la configuración actual.
    
    Returns:
        str: Resumen de configuración.
    """
    summary = f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                 CONFIGURACIÓN DE ANÁLISIS                     ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║ Modelo Kuramoto:                                              ║
    ║   γ (gamma):         {DEFAULT_MODEL_CONFIG.gamma:<10}                        ║
    ║   Δt (delta_t):      {DEFAULT_MODEL_CONFIG.delta_t:<10}                        ║
    ║   T (timesteps):     {DEFAULT_MODEL_CONFIG.timesteps:<10}                        ║
    ║   Canales:           {DEFAULT_MODEL_CONFIG.ch:<10}                        ║
    ║   Tamaño imagen:     {DEFAULT_MODEL_CONFIG.img_size}x{DEFAULT_MODEL_CONFIG.img_size}                              ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║ Barrido Alpha:                                                ║
    ║   Rango:             [{DEFAULT_ALPHA_CONFIG.alpha_start}, {DEFAULT_ALPHA_CONFIG.alpha_end}]                        ║
    ║   Paso:              {DEFAULT_ALPHA_CONFIG.alpha_step:<10}                        ║
    ║   Puntos:            {DEFAULT_ALPHA_CONFIG.num_points:<10}                        ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║ Paths:                                                        ║
    ║   Datos:             {str(DATA_DIR):<42} ║
    ║   Resultados DB:     {str(RESULTS_DB.name):<42} ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    return summary.strip()
