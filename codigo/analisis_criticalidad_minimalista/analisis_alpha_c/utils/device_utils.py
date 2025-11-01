"""
Utilidades para detección y configuración de dispositivos (CPU, CUDA, MPS).
"""

import torch


def get_device(device_arg='auto'):
    """
    Detecta el mejor dispositivo disponible para PyTorch.
    
    Args:
        device_arg (str): Dispositivo explícito ('cuda', 'mps', 'cpu') o 'auto'
                         para detección automática.
    
    Returns:
        torch.device: Dispositivo seleccionado.
    
    Examples:
        >>> device = get_device('auto')  # Detección automática
        >>> device = get_device('cuda')  # Forzar CUDA
        >>> device = get_device('cpu')   # Forzar CPU
    """
    if device_arg != 'auto':
        return torch.device(device_arg)
    
    # Prioridad: CUDA > MPS > CPU
    if torch.cuda.is_available():
        return torch.device('cuda')
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    
    return torch.device('cpu')


def print_device_info(device):
    """
    Imprime información sobre el dispositivo utilizado.
    
    Args:
        device (torch.device): Dispositivo a describir.
    """
    device_names = {
        'cuda': '✅ Usando GPU: CUDA',
        'mps': '✅ Usando GPU: Metal Performance Shaders (MPS)',
        'cpu': '⚠️  Usando CPU (sin aceleración GPU)'
    }
    
    device_type = device.type
    print(device_names.get(device_type, f'✅ Usando dispositivo: {device_type}'))
    
    if device_type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
