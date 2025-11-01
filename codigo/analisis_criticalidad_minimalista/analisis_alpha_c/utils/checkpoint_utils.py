"""
Utilidades para manejo de checkpoints PyTorch.
"""

from pathlib import Path
from typing import List, Dict, Any
import torch


def load_checkpoint(path, map_location=None, weights_only=False):
    """
    Carga un checkpoint PyTorch con manejo de versiones.
    
    Args:
        path (Path o str): Ruta al archivo checkpoint.
        map_location (str o torch.device): Dispositivo donde cargar (default: None).
        weights_only (bool): Si True, solo carga pesos (PyTorch 2.6+) (default: False).
    
    Returns:
        dict: Datos del checkpoint.
    
    Examples:
        >>> data = load_checkpoint('checkpoint_clase1_0100imgs.pt')
        >>> alphas_c = data['alphas_c']
        >>> clase = data['clase']
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {path}")
    
    try:
        # PyTorch 2.6+ requiere weights_only explícito
        data = torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        # Versiones anteriores de PyTorch
        data = torch.load(path, map_location=map_location)
    
    return data


def save_checkpoint(data, path, create_dirs=True):
    """
    Guarda un checkpoint PyTorch de forma consistente.
    
    Args:
        data (dict): Datos a guardar.
        path (Path o str): Ruta destino.
        create_dirs (bool): Si True, crea directorios necesarios (default: True).
    
    Examples:
        >>> checkpoint = {
        ...     'alphas_c': torch.tensor([0.1, 0.2, 0.3]),
        ...     'clase': 1,
        ...     'num_imgs': 100
        ... }
        >>> save_checkpoint(checkpoint, 'checkpoints/clase1.pt')
    """
    path = Path(path)
    
    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(data, path)


def find_checkpoints(directory, pattern='checkpoint_clase*_*.pt', sort=True):
    """
    Busca archivos de checkpoint en un directorio.
    
    Args:
        directory (Path o str): Directorio donde buscar.
        pattern (str): Patrón glob para buscar (default: 'checkpoint_clase*_*.pt').
        sort (bool): Si True, ordena los resultados (default: True).
    
    Returns:
        List[Path]: Lista de rutas a checkpoints encontrados.
    
    Raises:
        FileNotFoundError: Si el directorio no existe.
        ValueError: Si no se encuentran checkpoints.
    
    Examples:
        >>> checkpoints = find_checkpoints('checkpoints_impares', pattern='*_0100imgs.pt')
        >>> len(checkpoints)
        5
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {directory}")
    
    files = list(directory.glob(pattern))
    
    if not files:
        raise ValueError(f"No se encontraron checkpoints con patrón '{pattern}' en {directory}")
    
    if sort:
        files = sorted(files)
    
    return files


def extract_checkpoint_info(checkpoint_path):
    """
    Extrae información del nombre de archivo del checkpoint.
    
    Args:
        checkpoint_path (Path o str): Ruta al checkpoint.
    
    Returns:
        dict: Información extraída (clase, num_imgs, etc.).
    
    Examples:
        >>> info = extract_checkpoint_info('checkpoint_clase3_0100imgs.pt')
        >>> info
        {'clase': 3, 'num_imgs': 100}
    """
    import re
    
    path = Path(checkpoint_path)
    filename = path.stem  # Sin extensión
    
    info = {}
    
    # Extraer número de clase
    match_clase = re.search(r'clase(\d+)', filename)
    if match_clase:
        info['clase'] = int(match_clase.group(1))
    
    # Extraer número de imágenes
    match_imgs = re.search(r'(\d+)imgs', filename)
    if match_imgs:
        info['num_imgs'] = int(match_imgs.group(1))
    
    return info


def get_checkpoint_metadata(checkpoint_path):
    """
    Carga un checkpoint y retorna solo sus metadatos (sin tensores grandes).
    
    Args:
        checkpoint_path (Path o str): Ruta al checkpoint.
    
    Returns:
        dict: Metadatos del checkpoint.
    """
    data = load_checkpoint(checkpoint_path, weights_only=False)
    
    metadata = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            metadata[key] = {
                'type': 'tensor',
                'shape': list(value.shape),
                'dtype': str(value.dtype),
                'device': str(value.device)
            }
        else:
            metadata[key] = value
    
    return metadata
