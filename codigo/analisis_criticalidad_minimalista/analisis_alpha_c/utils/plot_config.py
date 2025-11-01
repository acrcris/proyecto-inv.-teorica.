"""
Utilidades para configuración de matplotlib y guardado de figuras.
"""

import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt


def setup_matplotlib(backend='Agg', cache_dir='_mplconfig', style=None):
    """
    Configura matplotlib con backend y caché personalizados.
    
    Args:
        backend (str): Backend de matplotlib (default: 'Agg' para entornos sin display).
        cache_dir (str o Path): Directorio para caché de matplotlib (default: '_mplconfig').
                               Si es None, no configura caché.
        style (str o list): Estilo(s) de matplotlib a aplicar (default: None).
    
    Examples:
        >>> setup_matplotlib()  # Backend Agg, caché en _mplconfig
        >>> setup_matplotlib(backend='Qt5Agg', cache_dir=None)  # Backend interactivo
        >>> setup_matplotlib(style='seaborn-v0_8')  # Con estilo
    """
    # Configurar caché si se especifica
    if cache_dir is not None:
        cache_path = Path.cwd() / cache_dir
        try:
            cache_path.mkdir(exist_ok=True)
            os.environ['MPLCONFIGDIR'] = str(cache_path)
        except Exception:
            # Si falla, matplotlib usará su directorio temporal por defecto
            pass
    
    # Configurar backend
    matplotlib.use(backend)
    
    # Aplicar estilo si se especifica
    if style is not None:
        plt.style.use(style)


def save_figure(fig, path, format=None, dpi=300, bbox_inches='tight', **kwargs):
    """
    Guarda una figura con parámetros consistentes.
    
    Args:
        fig (matplotlib.figure.Figure): Figura a guardar.
        path (str o Path): Ruta destino.
        format (str): Formato de salida ('png', 'pdf', 'svg', etc.).
                     Si es None, se infiere de la extensión del archivo.
        dpi (int): Resolución en puntos por pulgada (default: 300).
        bbox_inches (str): Manejo de bordes (default: 'tight').
        **kwargs: Argumentos adicionales para plt.savefig().
    
    Examples:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3])
        >>> save_figure(fig, 'output.png')
        >>> save_figure(fig, 'output.pdf', format='pdf', dpi=150)
    """
    path = Path(path)
    
    # Crear directorio si no existe
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Inferir formato de la extensión si no se especifica
    if format is None:
        format = path.suffix.lstrip('.')
        if not format:
            format = 'png'  # Default
    
    # Guardar figura
    fig.savefig(
        path,
        format=format,
        dpi=dpi,
        bbox_inches=bbox_inches,
        **kwargs
    )


def get_default_figure_params():
    """
    Retorna parámetros por defecto recomendados para figuras.
    
    Returns:
        dict: Diccionario de parámetros rcParams.
    
    Examples:
        >>> params = get_default_figure_params()
        >>> plt.rcParams.update(params)
    """
    return {
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
    }


def apply_default_style():
    """
    Aplica estilo por defecto consistente para todas las visualizaciones.
    
    Examples:
        >>> apply_default_style()
        >>> fig, ax = plt.subplots()  # Usará el estilo por defecto
    """
    params = get_default_figure_params()
    plt.rcParams.update(params)


class FigureContext:
    """
    Context manager para crear y guardar figuras con configuración consistente.
    
    Examples:
        >>> with FigureContext('output.png', figsize=(12, 8)) as (fig, ax):
        ...     ax.plot([1, 2, 3])
        # Figura guardada automáticamente al salir del contexto
    """
    
    def __init__(self, output_path=None, figsize=(10, 6), dpi=100, **save_kwargs):
        """
        Args:
            output_path (str o Path): Ruta donde guardar la figura (opcional).
            figsize (tuple): Tamaño de la figura (default: (10, 6)).
            dpi (int): DPI de la figura (default: 100).
            **save_kwargs: Argumentos adicionales para save_figure().
        """
        self.output_path = output_path
        self.figsize = figsize
        self.dpi = dpi
        self.save_kwargs = save_kwargs
        self.fig = None
        self.ax = None
    
    def __enter__(self):
        """Crea la figura y ejes."""
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        return self.fig, self.ax
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Guarda y cierra la figura."""
        if self.output_path and exc_type is None:
            save_figure(self.fig, self.output_path, **self.save_kwargs)
        plt.close(self.fig)
        return False
