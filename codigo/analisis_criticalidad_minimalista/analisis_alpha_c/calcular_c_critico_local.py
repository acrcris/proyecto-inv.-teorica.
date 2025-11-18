#!/usr/bin/env python3
"""
Script para calcular C_crítico de imágenes MNIST localmente.

Características:
- Reproducibilidad total: semilla fija y estado inicial consistente
- Base de datos SQLite con una tabla por clase (0-9)
- Capacidad de reinicio: detecta imágenes ya procesadas
- Progreso persistente: checkpoints en cada imagen
- ✨ OPTIMIZADO PARA APPLE SILICON (M1/M2/M3) ✨

Optimizaciones MPS (Metal Performance Shaders):
- Detección automática de GPU Apple Silicon
- Cálculo del parámetro de orden sin números complejos (más rápido en MPS)
- Gestión inteligente de memoria GPU
- Liberación periódica de caché MPS
- Fallback automático a CPU si MPS falla

Uso:
    # Detección automática del mejor dispositivo (recomendado)
    python calcular_c_critico_local.py --clases 0 1 2 --limite 100
    
    # Forzar uso de MPS (Metal)
    python calcular_c_critico_local.py --clases 0 --device mps
    
    # Procesar todas las clases
    python calcular_c_critico_local.py --all
"""

import os
import sys
import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random
import json

# Añadir path del módulo kuramoto del análisis_criticalidad_minimalista
SCRIPT_DIR = Path(__file__).parent
ANALISIS_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ANALISIS_ROOT))

# Importar desde el módulo local de análisis_criticalidad_minimalista
from kuramoto import KBlock


# ============================================================================
# CONFIGURACIÓN GLOBAL (REPRODUCIBILIDAD)
# ============================================================================

SEED = 1  # Semilla fija como en el notebook

def set_seed(seed):
    """Configura todas las semillas para reproducibilidad total."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Optimización para MPS (Metal Performance Shaders en Mac)
    if torch.backends.mps.is_available():
        # MPS no tiene semilla específica, pero podemos configurar el fallback
        torch.use_deterministic_algorithms(False)  # MPS no soporta todas las ops determinísticas


def cleanup_mps_resources():
    """Limpia recursos de MPS para evitar semaphore leaks."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()
        # Forzar garbage collection
        import gc
        gc.collect()


# ============================================================================
# PARÁMETROS DEL MODELO (IDÉNTICOS AL NOTEBOOK)
# ============================================================================

PARAMS = {
    'seed': SEED,
    'ch': 3,      # Canales del estado
    'n': 3,       # Dimensión del oscilador
    'h': 64,      # Alto de la resolución
    'w': 64,      # Ancho de la resolución
    'T': 30,      # Pasos temporales
    'gamma': 0.7,
    'del_t': 0.9,
    'ksize': 3,
    'init_omg': 0.1,
}

# Rango de valores de C a evaluar
C_RANGE = np.arange(0.0, 2.0, 0.01)


# ============================================================================
# FUNCIONES DE ANÁLISIS
# ============================================================================

def kuramoto_order_parameter(xs):
    """
    Calcula el parámetro de orden de Kuramoto r(t) para cada paso de tiempo.
    Optimizado para MPS (Metal) en Mac.
    
    Args:
        xs: Lista de estados [T, B, C, H, W]
    
    Returns:
        np.array: Valores de r(t) para cada paso temporal
    """
    r_values = []

    for x in xs:
        # x: [B, C, H, W] donde C son las componentes del oscilador
        x = x[0]  # Tomamos primer batch

        # Calcula la fase de cada oscilador (asumiendo n=3, usar primeras 2 componentes)
        theta = torch.atan2(x[1], x[0])  # fase en radianes

        # OPTIMIZACIÓN MPS: Operaciones complejas pueden ser lentas en MPS
        # Usar representación polar directamente
        if x.device.type == 'mps':
            # Calcular magnitud del parámetro de orden sin números complejos
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            r_x = cos_theta.mean()
            r_y = sin_theta.mean()
            r = torch.sqrt(r_x**2 + r_y**2)
        else:
            # Método original para CUDA/CPU (soporta complejos nativamente)
            complex_phase = torch.exp(1j * theta)
            order_param = complex_phase.mean()
            r = torch.abs(order_param)
        
        r_values.append(r.item())

    return np.array(r_values)


def calculate_c_critical(img, kblock, x_init, device, c_range, ch=3, h=64, w=64, 
                        T=30, gamma=0.7, del_t=0.9):
    """
    Calcula el valor crítico C_c para una imagen.
    
    Args:
        img: Imagen de entrada [1, H, W]
        kblock: Modelo KBlock
        x_init: Estado inicial [1, ch, h, w]
        device: Dispositivo (cuda/cpu/mps)
        c_range: Rango de valores de C a evaluar
        ch, h, w: Dimensiones de los canales y resolución
        T: Pasos temporales
        gamma, del_t: Parámetros de integración
    
    Returns:
        dict: Diccionario con c_critical, R_final, y otros datos
    """
    # Preprocesar imagen
    img_resized = TF.resize(img, [h, w])
    img_channels = img_resized.repeat(ch, 1, 1)
    img_channels = img_channels.to(device)
    
    R_final = []
    
    # OPTIMIZACIÓN MPS: Procesar en lotes más pequeños si es necesario
    # MPS puede tener problemas de memoria con bucles largos
    if device.type == 'mps':
        # Liberar caché de MPS antes de comenzar
        torch.mps.empty_cache()
    
    # Evaluar para cada valor de C
    for c_val in c_range:
        x = x_init.clone().to(device)
        c = img_channels.unsqueeze(0) * c_val
        
        with torch.no_grad():
            x, xs, es = kblock(x, c, T=T, gamma=gamma, del_t=del_t, 
                              return_xs=True, return_es=True)
        
        R = kuramoto_order_parameter(xs)
        R_final.append(R[-1])
        
        # OPTIMIZACIÓN MPS: Liberar memoria periódicamente
        if device.type == 'mps' and len(R_final) % 50 == 0:
            torch.mps.empty_cache()
    
    R_final = np.array(R_final)
    
    # Calcular C_crítico como máximo de la derivada
    df = np.gradient(R_final, c_range)
    i_c = np.argmax(df)
    c_critical = c_range[i_c]
    
    return {
        'c_critical': float(c_critical),
        'R_final': R_final.tolist(),
        'c_range': c_range.tolist(),
        'max_derivative_idx': int(i_c)
    }


# ============================================================================
# BASE DE DATOS SQLite
# ============================================================================

def create_database(db_path):
    """
    Crea la base de datos SQLite con tablas para cada clase de MNIST (0-9).
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Crear una tabla por cada clase
    for clase in range(10):
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS clase_{clase} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_idx INTEGER UNIQUE NOT NULL,
            c_critical REAL NOT NULL,
            max_derivative_idx INTEGER,
            timestamp TEXT NOT NULL,
            seed INTEGER,
            n INTEGER,
            ch INTEGER,
            T INTEGER,
            gamma REAL,
            del_t REAL
        )
        ''')
    
    conn.commit()
    conn.close()
    print(f"✅ Base de datos creada/verificada: {db_path}")


def save_result(db_path, clase, image_idx, result, params):
    """
    Guarda un resultado en la base de datos.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    
    cursor.execute(f'''
    INSERT OR REPLACE INTO clase_{clase} 
    (image_idx, c_critical, max_derivative_idx, timestamp, seed, n, ch, T, gamma, del_t)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (image_idx, result['c_critical'], result['max_derivative_idx'], 
          timestamp, params['seed'], params['n'], params['ch'], 
          params['T'], params['gamma'], params['del_t']))
    
    conn.commit()
    conn.close()


def get_processed_images(db_path, clase):
    """
    Obtiene los índices de imágenes ya procesadas para una clase.
    
    Returns:
        set: Conjunto de índices ya procesados
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute(f'SELECT image_idx FROM clase_{clase}')
        processed = set(row[0] for row in cursor.fetchall())
    except sqlite3.OperationalError:
        # Tabla no existe aún
        processed = set()
    
    conn.close()
    return processed


def get_statistics(db_path, clase):
    """
    Obtiene estadísticas de C_crítico para una clase.
    
    Returns:
        dict: Estadísticas (mean, std, min, max, median, n)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute(f'SELECT c_critical FROM clase_{clase}')
        c_criticals = np.array([row[0] for row in cursor.fetchall()])
        
        if len(c_criticals) > 0:
            stats = {
                'n_images': len(c_criticals),
                'mean': float(np.mean(c_criticals)),
                'std': float(np.std(c_criticals)),
                'min': float(np.min(c_criticals)),
                'max': float(np.max(c_criticals)),
                'median': float(np.median(c_criticals))
            }
        else:
            stats = None
    except sqlite3.OperationalError:
        stats = None
    
    conn.close()
    return stats


# ============================================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================================

def procesar_clase(clase, dataset, imagenes_por_clase, kblock, x_init, device, 
                  db_path, limite=None, verbose=True):
    """
    Procesa todas las imágenes de una clase.
    
    Args:
        clase: Número de clase (0-9)
        dataset: Dataset de MNIST
        imagenes_por_clase: Diccionario con índices por clase
        kblock: Modelo KBlock
        x_init: Estado inicial
        device: Dispositivo
        db_path: Ruta de la base de datos
        limite: Número máximo de imágenes a procesar (None = todas)
        verbose: Mostrar información detallada
    
    Returns:
        dict: Estadísticas de la clase procesada
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"PROCESANDO CLASE {clase}")
        print(f"{'='*60}")
    
    indices = imagenes_por_clase[clase]
    if limite:
        indices = indices[:limite]
    
    # Verificar imágenes ya procesadas (AUTO-RESTART)
    processed_images = get_processed_images(db_path, clase)
    indices_pendientes = [idx for idx in indices if idx not in processed_images]
    
    if verbose:
        print(f"✅ Imágenes ya procesadas: {len(processed_images)}")
        print(f"⏳ Imágenes pendientes: {len(indices_pendientes)}")
    
    if len(indices_pendientes) == 0:
        if verbose:
            print(f"✨ Clase {clase} ya está completamente procesada. Saltando...")
        stats = get_statistics(db_path, clase)
        return stats
    
    # Procesar imágenes pendientes
    for i, idx in enumerate(tqdm(indices_pendientes, desc=f"Clase {clase}", disable=not verbose)):
        img, label = dataset[idx]
        
        # Calcular C_crítico
        result = calculate_c_critical(
            img, kblock, x_init, device, C_RANGE,
            ch=PARAMS['ch'], h=PARAMS['h'], w=PARAMS['w'],
            T=PARAMS['T'], gamma=PARAMS['gamma'], del_t=PARAMS['del_t']
        )
        
        # Guardar en base de datos (checkpoint automático)
        save_result(db_path, clase, idx, result, PARAMS)
        
        # LIMPIEZA AGRESIVA: Cada 100 imágenes limpiamos recursos MPS
        if device.type == 'mps' and (i + 1) % 100 == 0:
            cleanup_mps_resources()
    
    # Limpieza final
    if device.type == 'mps':
        cleanup_mps_resources()
    
    # Obtener estadísticas finales
    stats = get_statistics(db_path, clase)
    
    if verbose and stats:
        print(f"\n📊 Estadísticas Clase {clase}:")
        print(f"  Imágenes procesadas: {stats['n_images']}")
        print(f"  C_crítico medio: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Rango: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Mediana: {stats['median']:.4f}")
    
    return stats


def visualizar_resultados(db_path, clases, output_dir):
    """
    Genera visualizaciones de los resultados.
    
    Args:
        db_path: Ruta de la base de datos
        clases: Lista de clases a visualizar
        output_dir: Directorio de salida para gráficos
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Distribución por clase
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    resultados_resumen = {}
    
    for clase in range(10):
        if clase not in clases:
            axes[clase].text(0.5, 0.5, f'Clase {clase}\nNo procesada', 
                           ha='center', va='center', fontsize=12)
            axes[clase].axis('off')
            continue
        
        stats = get_statistics(db_path, clase)
        
        if stats is None:
            axes[clase].text(0.5, 0.5, f'Clase {clase}\nSin datos', 
                           ha='center', va='center', fontsize=12)
            axes[clase].axis('off')
            continue
        
        resultados_resumen[clase] = stats
        
        # Obtener valores de C_crítico
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f'SELECT c_critical FROM clase_{clase}')
        c_vals = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        axes[clase].hist(c_vals, bins=30, edgecolor='black', alpha=0.7)
        axes[clase].axvline(stats['mean'], 
                          color='r', linestyle='--', linewidth=2,
                          label=f"Media: {stats['mean']:.3f}")
        axes[clase].set_title(f"Clase {clase}")
        axes[clase].set_xlabel("C_crítico")
        axes[clase].set_ylabel("Frecuencia")
        axes[clase].legend()
        axes[clase].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'distribucion_c_critical_por_clase.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Gráfico guardado: {plot_path}")
    plt.close()
    
    # Comparación de medias
    if resultados_resumen:
        clases_procesadas = sorted(resultados_resumen.keys())
        medias = [resultados_resumen[c]['mean'] for c in clases_procesadas]
        stds = [resultados_resumen[c]['std'] for c in clases_procesadas]
        
        plt.figure(figsize=(12, 6))
        plt.errorbar(clases_procesadas, medias, yerr=stds, fmt='o-', capsize=5, 
                    markersize=8, linewidth=2, capthick=2)
        plt.xlabel('Clase', fontsize=12)
        plt.ylabel('C_crítico medio', fontsize=12)
        plt.title('Comparación de C_crítico entre clases de MNIST', fontsize=14)
        plt.xticks(clases_procesadas)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = output_dir / 'comparacion_medias_clases.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {plot_path}")
        plt.close()
    
    # Guardar resumen en JSON
    if resultados_resumen:
        json_path = output_dir / 'resumen_c_critical.json'
        with open(json_path, 'w') as f:
            json.dump(resultados_resumen, f, indent=2)
        print(f"✅ Resumen guardado: {json_path}")
    
    return resultados_resumen


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Calcular C_crítico para imágenes MNIST localmente',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Procesar clases 0 y 1 completas
  python calcular_c_critico_local.py --clases 0 1
  
  # Procesar primeras 100 imágenes de clase 5
  python calcular_c_critico_local.py --clases 5 --limite 100
  
  # Procesar todas las clases completas (60,000 imágenes)
  python calcular_c_critico_local.py --all
  
  # Continuar procesamiento interrumpido (auto-restart)
  python calcular_c_critico_local.py --clases 0 1 2 3 4 5 6 7 8 9
        """
    )
    
    parser.add_argument('--clases', type=int, nargs='+', 
                       help='Clases a procesar (0-9)')
    parser.add_argument('--all', action='store_true',
                       help='Procesar todas las clases')
    parser.add_argument('--limite', type=int, default=None,
                       help='Límite de imágenes por clase (None = todas)')
    parser.add_argument('--db', type=str, 
                       default='resultados_c_critical/mnist_critical_tot.db',
                       help='Ruta de la base de datos SQLite')
    parser.add_argument('--output', type=str, 
                       default='resultados_c_critical',
                       help='Directorio de salida para resultados')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Dispositivo a usar')
    parser.add_argument('--no-viz', action='store_true',
                       help='No generar visualizaciones')
    
    args = parser.parse_args()
    
    # Validar argumentos
    if not args.all and not args.clases:
        parser.error("Debes especificar --clases o --all")
    
    if args.all:
        clases_a_procesar = list(range(10))
    else:
        clases_a_procesar = args.clases
        for c in clases_a_procesar:
            if c < 0 or c > 9:
                parser.error(f"Clase {c} inválida. Debe estar entre 0 y 9")
    
    # Configurar reproducibilidad
    set_seed(SEED)
    print(f"🔧 Semilla configurada: {SEED}")
    
    # Configurar dispositivo con optimizaciones MPS
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🖥️  Usando dispositivo: CUDA")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"🖥️  Usando dispositivo: MPS (Metal Performance Shaders)")
            print(f"   ✅ Optimizado para Apple Silicon (M1/M2/M3)")
            print(f"   ⚡ Aceleración GPU nativa de Mac")
            # Configurar optimizaciones para MPS
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Mejor gestión de memoria
        else:
            device = torch.device('cpu')
            print(f"🖥️  Usando dispositivo: CPU")
            print(f"   ⚠️  No se detectó GPU. El procesamiento será más lento.")
    else:
        device = torch.device(args.device)
        if args.device == 'mps':
            print(f"🖥️  Usando dispositivo: MPS (Metal Performance Shaders)")
            print(f"   ✅ Optimizado para Apple Silicon")
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        else:
            print(f"🖥️  Usando dispositivo: {device}")
    
    print(f"📦 PyTorch version: {torch.__version__}")
    
    # Verificación adicional para MPS
    if device.type == 'mps':
        print(f"🔍 Verificando compatibilidad MPS...")
        try:
            # Test simple de MPS
            test_tensor = torch.randn(10, 10, device=device)
            _ = test_tensor @ test_tensor.T
            print(f"   ✅ MPS funcionando correctamente")
        except Exception as e:
            print(f"   ⚠️  Advertencia MPS: {e}")
            print(f"   💡 Cambiando a CPU por seguridad...")
            device = torch.device('cpu')
    
    # Crear directorio de salida
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / Path(args.db).name
    
    # Crear base de datos
    create_database(db_path)
    
    # Cargar dataset MNIST
    print("\n📥 Cargando dataset MNIST...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Directorio de datos (en el nivel del proyecto)
    data_dir = ANALISIS_ROOT.parent.parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    train_dataset = torchvision.datasets.MNIST(
        root=str(data_dir), 
        train=True, 
        download=True, 
        transform=transform
    )
    
    print(f"✅ Dataset cargado: {len(train_dataset)} imágenes")
    
    # Organizar por clases
    imagenes_por_clase = {i: [] for i in range(10)}
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        imagenes_por_clase[label].append(i)
    
    print("\n📊 Distribución por clase:")
    for clase in range(10):
        print(f"  Clase {clase}: {len(imagenes_por_clase[clase])} imágenes")
    
    # Inicializar modelo (ESTADO INICIAL CONSISTENTE)
    print(f"\n🧠 Inicializando modelo KBlock con parámetros:")
    for k, v in PARAMS.items():
        print(f"  {k}: {v}")
    
    set_seed(SEED)  # Re-setear justo antes de crear el modelo
    
    kblock = KBlock(
        n=PARAMS['n'], 
        ch=PARAMS['ch'], 
        connectivity='conv', 
        T=PARAMS['T'], 
        ksize=PARAMS['ksize'],
        init_omg=PARAMS['init_omg'], 
        c_norm=None, 
        use_omega_c=False
    ).to(device)
    
    # Estado inicial (SE REUTILIZA para todas las imágenes - REPRODUCIBILIDAD)
    x_init = torch.randn(1, PARAMS['ch'], PARAMS['h'], PARAMS['w'])
    
    print(f"✅ Modelo inicializado en {device}")
    print(f"✅ Estado inicial creado: {x_init.shape}")
    print(f"✅ Rango de C: {len(C_RANGE)} valores [{C_RANGE[0]:.2f}, {C_RANGE[-1]:.2f}]")
    
    # Procesar cada clase
    print("\n" + "="*60)
    print("INICIANDO PROCESAMIENTO")
    print("="*60)
    print(f"Clases a procesar: {clases_a_procesar}")
    if args.limite:
        print(f"Límite por clase: {args.limite} imágenes")
    else:
        print("Procesando TODAS las imágenes de cada clase")
    print(f"Base de datos: {db_path}")
    print("="*60)
    
    resultados = {}
    
    for clase in clases_a_procesar:
        stats = procesar_clase(
            clase, train_dataset, imagenes_por_clase, 
            kblock, x_init, device, db_path, 
            limite=args.limite, verbose=True
        )
        resultados[clase] = stats
    
    print("\n" + "="*60)
    print("PROCESAMIENTO COMPLETADO")
    print("="*60)
    
    # Resumen final
    print("\n📊 RESUMEN FINAL:")
    for clase in sorted(resultados.keys()):
        stats = resultados[clase]
        if stats:
            print(f"\nClase {clase}:")
            print(f"  N: {stats['n_images']}")
            print(f"  Media: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Rango: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  Mediana: {stats['median']:.4f}")
    
    # Visualizaciones
    if not args.no_viz:
        print("\n📈 Generando visualizaciones...")
        visualizar_resultados(db_path, clases_a_procesar, output_dir)
    
    print(f"\n✅ Resultados guardados en: {output_dir}")
    print(f"   📊 Base de datos: {db_path}")
    print(f"   📈 Gráficos: {output_dir}/*.png")
    print(f"   📄 Resumen: {output_dir}/resumen_c_critical.json")
    print("\n🎉 ¡Procesamiento completado exitosamente!")


if __name__ == '__main__':
    main()
