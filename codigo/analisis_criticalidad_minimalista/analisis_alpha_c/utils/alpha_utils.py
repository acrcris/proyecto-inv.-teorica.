"""
Utilidades para generación y manejo de valores de alpha (α).
"""

import numpy as np


def generate_alphas(start, end, step):
    """
    Genera un array de valores alpha para barrido.
    
    Args:
        start (float): Valor inicial de alpha.
        end (float): Valor final de alpha.
        step (float): Incremento entre valores.
    
    Returns:
        np.ndarray: Array de valores alpha.
    
    Raises:
        ValueError: Si los parámetros no son válidos.
    
    Examples:
        >>> alphas = generate_alphas(0.0, 2.0, 0.05)
        >>> len(alphas)
        41
        >>> alphas[0], alphas[-1]
        (0.0, 2.0)
    """
    # Validación de parámetros
    if start >= end:
        raise ValueError(f"start ({start}) debe ser menor que end ({end})")
    
    if step <= 0:
        raise ValueError(f"step ({step}) debe ser mayor que 0")
    
    if step > (end - start):
        raise ValueError(f"step ({step}) es mayor que el rango ({end - start})")
    
    # Generar array
    count = int(np.floor((end - start) / step)) + 1
    alphas = start + step * np.arange(count, dtype=np.float64)
    
    # Asegurar que el último valor no exceda 'end' debido a errores de punto flotante
    alphas = np.clip(alphas, start, end)
    
    return alphas


def find_alpha_critical(alphas, order_params, threshold=0.5, method='interpolate'):
    """
    Encuentra el valor crítico de alpha donde el parámetro de orden cruza un umbral.
    
    Args:
        alphas (np.ndarray): Valores de alpha.
        order_params (np.ndarray): Parámetros de orden correspondientes.
        threshold (float): Umbral del parámetro de orden (default: 0.5).
        method (str): Método de búsqueda ('interpolate', 'nearest', 'first').
    
    Returns:
        float: Valor crítico de alpha, o None si no se encuentra.
    
    Examples:
        >>> alphas = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        >>> order_params = np.array([0.1, 0.3, 0.45, 0.6, 0.8])
        >>> find_alpha_critical(alphas, order_params, threshold=0.5)
        0.25  # Interpolado entre 0.2 y 0.3
    """
    if len(alphas) != len(order_params):
        raise ValueError("alphas y order_params deben tener la misma longitud")
    
    # Buscar cruces del umbral
    crosses = np.where(np.diff(np.sign(order_params - threshold)))[0]
    
    if len(crosses) == 0:
        return None  # No hay cruce
    
    # Tomar el primer cruce
    idx = crosses[0]
    
    if method == 'nearest':
        # Retornar el alpha más cercano al umbral
        if abs(order_params[idx] - threshold) < abs(order_params[idx + 1] - threshold):
            return alphas[idx]
        else:
            return alphas[idx + 1]
    
    elif method == 'first':
        # Retornar el primer alpha que supera el umbral
        return alphas[idx + 1]
    
    elif method == 'interpolate':
        # Interpolación lineal entre los dos puntos
        alpha1, alpha2 = alphas[idx], alphas[idx + 1]
        r1, r2 = order_params[idx], order_params[idx + 1]
        
        # Interpolación: alpha_c = alpha1 + (threshold - r1) * (alpha2 - alpha1) / (r2 - r1)
        if r2 != r1:  # Evitar división por cero
            alpha_c = alpha1 + (threshold - r1) * (alpha2 - alpha1) / (r2 - r1)
            return alpha_c
        else:
            return alpha1
    
    else:
        raise ValueError(f"Método desconocido: {method}")
