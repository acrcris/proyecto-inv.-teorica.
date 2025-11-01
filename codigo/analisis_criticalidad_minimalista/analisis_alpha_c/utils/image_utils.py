"""
Utilidades para preparación y procesamiento de imágenes.
"""

import torch
import torch.nn.functional as F


def prepare_image(img, img_size=64, ch=4, device='cpu', add_batch_dim=False):
    """
    Prepara una imagen para el modelo Kuramoto.
    
    Realiza interpolación si es necesario, replica canales y transfiere al dispositivo.
    
    Args:
        img (torch.Tensor): Imagen de entrada [C, H, W] o [H, W].
        img_size (int): Tamaño objetivo (default: 64).
        ch (int): Número de canales a replicar (default: 4).
        device (str o torch.device): Dispositivo destino (default: 'cpu').
        add_batch_dim (bool): Si True, añade dimensión de batch [1, ch, H, W] (default: False).
    
    Returns:
        torch.Tensor: Imagen preparada [ch, H, W] o [1, ch, H, W] si add_batch_dim=True.
    
    Examples:
        >>> img = torch.randn(1, 28, 28)  # MNIST original
        >>> prepared = prepare_image(img, img_size=64, ch=4, device='cuda')
        >>> prepared.shape
        torch.Size([4, 64, 64])
        
        >>> prepared_batch = prepare_image(img, img_size=64, ch=4, add_batch_dim=True)
        >>> prepared_batch.shape
        torch.Size([1, 4, 64, 64])
    """
    # Convertir device a torch.device si es string
    if isinstance(device, str):
        device = torch.device(device)
    
    # Asegurar que img tenga al menos 2 dimensiones [H, W]
    if img.dim() == 2:
        img = img.unsqueeze(0)  # [1, H, W]
    elif img.dim() == 1:
        raise ValueError(f"Imagen debe tener al menos 2 dimensiones, recibida: {img.shape}")
    
    # Tomar solo el primer canal si tiene múltiples
    if img.shape[0] > 1:
        img = img[0:1]  # [1, H, W]
    
    # Interpolar si el tamaño no coincide
    if img.shape[-1] != img_size or img.shape[-2] != img_size:
        img = F.interpolate(
            img.unsqueeze(0),  # [1, 1, H, W]
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False
        )[0]  # [1, H, W]
    
    # Replicar canales
    img_channels = img.repeat(ch, 1, 1)  # [ch, H, W]
    
    # Transferir a dispositivo
    img_channels = img_channels.to(device)
    
    # Añadir dimensión de batch si se solicita
    if add_batch_dim:
        img_channels = img_channels.unsqueeze(0)  # [1, ch, H, W]
    
    return img_channels


def normalize_image(img, mean=0.5, std=0.5):
    """
    Normaliza una imagen.
    
    Args:
        img (torch.Tensor): Imagen a normalizar.
        mean (float o tuple): Media para normalización.
        std (float o tuple): Desviación estándar para normalización.
    
    Returns:
        torch.Tensor: Imagen normalizada.
    """
    return (img - mean) / std


def denormalize_image(img, mean=0.5, std=0.5):
    """
    Desnormaliza una imagen.
    
    Args:
        img (torch.Tensor): Imagen normalizada.
        mean (float o tuple): Media usada en normalización.
        std (float o tuple): Desviación estándar usada en normalización.
    
    Returns:
        torch.Tensor: Imagen desnormalizada.
    """
    return img * std + mean
