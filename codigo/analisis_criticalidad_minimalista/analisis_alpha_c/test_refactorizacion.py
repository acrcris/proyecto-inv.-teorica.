#!/usr/bin/env python3
"""
test_refactorizacion.py

Suite de pruebas para validar que la refactorización NO cambia los resultados.
Compara los cálculos de α_c antes y después de usar utils/.

Uso:
    python test_refactorizacion.py --verbose
"""
import sys
from pathlib import Path
import torch
import numpy as np
import argparse

# Asegurar que encuentra los módulos
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.loader import MNISTLoader
from kuramoto.modelo import KBlock
from analisis.criticalidad import KuramotoMetrics
import torch.nn.functional as F

# Importar las funciones refactorizadas desde la carpeta local utils/
from analisis_alpha_c.utils.device_utils import get_device
from analisis_alpha_c.utils.alpha_utils import generate_alphas
from analisis_alpha_c.utils.image_utils import prepare_image


# ============================================================================
# IMPLEMENTACIONES ORIGINALES (para comparación)
# ============================================================================

def _prepare_device_original(explicit: str | None = None) -> torch.device:
    """Implementación original de preparación de dispositivo."""
    if explicit is not None:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _generate_alphas_original(start: float, end: float, step: float) -> np.ndarray:
    """Implementación original de generación de alphas."""
    count = int(np.floor((end - start) / step)) + 1
    return start + step * np.arange(count, dtype=np.float64)


def _prepare_image_original(img: torch.Tensor, img_size: int, ch: int, device: torch.device) -> torch.Tensor:
    """Implementación original de preparación de imagen."""
    if img.shape[-1] != img_size:
        img = F.interpolate(img.unsqueeze(0), size=(img_size, img_size), 
                          mode="bilinear", align_corners=False)[0]
    img_channels = img.repeat(ch, 1, 1)
    return img_channels.to(device)


# ============================================================================
# FUNCIÓN DE CÁLCULO DE ALPHA CRÍTICO (compartida)
# ============================================================================

def calculate_alpha_critical(
    kblock: KBlock,
    c_base: torch.Tensor,
    alphas: np.ndarray,
    timesteps: int,
    gamma: float,
    delta_t: float,
    window: int,
    device: torch.device
) -> tuple[float, np.ndarray]:
    """
    Calcula alpha crítico para una imagen.
    Retorna: (alpha_c, order_curve)
    """
    order_values = []
    
    for alpha in alphas:
        c_scaled = c_base * float(alpha)
        x0 = torch.randn_like(c_base, device=device)
        
        # Agregar dimensión batch (B=1)
        c_batch = c_scaled.unsqueeze(0)  # (ch, h, w) -> (1, ch, h, w)
        x0_batch = x0.unsqueeze(0)       # (ch, h, w) -> (1, ch, h, w)
        
        with torch.no_grad():
            _, xs = kblock(
                x0_batch,
                c_batch,
                T=timesteps,
                gamma=gamma,
                del_t=delta_t,
                return_xs=True,
            )
        
        r_series = KuramotoMetrics.order_parameter(xs)
        if len(r_series) == 0:
            order_values.append(0.0)
            continue
        
        tail = r_series[-window:] if window < len(r_series) else r_series
        order_values.append(float(np.mean(tail)))
    
    order_curve = np.asarray(order_values, dtype=np.float64)
    
    # Encontrar alpha crítico
    if len(order_curve) < 2:
        alpha_c = float("nan")
    else:
        gradient = np.gradient(order_curve, alphas)
        idx = int(np.argmax(gradient))
        alpha_c = float(alphas[idx])
    
    return alpha_c, order_curve


# ============================================================================
# TESTS
# ============================================================================

class TestRefactorizacion:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.tolerance = 1e-10  # Tolerancia para comparaciones float
        
    def log(self, msg):
        if self.verbose:
            print(f"  {msg}")
    
    def assert_equal(self, a, b, name, tolerance=None):
        """Verifica igualdad con tolerancia."""
        tol = tolerance or self.tolerance
        
        if isinstance(a, (int, float, np.floating)):
            equal = abs(a - b) < tol
        elif isinstance(a, np.ndarray):
            equal = np.allclose(a, b, atol=tol)
        elif isinstance(a, torch.Tensor):
            equal = torch.allclose(a, b, atol=tol)
        elif isinstance(a, torch.device):
            equal = str(a) == str(b)
        else:
            equal = a == b
        
        if equal:
            self.passed += 1
            self.log(f"✓ {name}: PASS")
        else:
            self.failed += 1
            print(f"✗ {name}: FAIL")
            print(f"  Original: {a}")
            print(f"  Refactorizado: {b}")
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                print(f"  Max diff: {np.max(np.abs(a - b))}")
        
        return equal
    
    def test_device_detection(self):
        """Test 1: Detección de dispositivo."""
        print("\n🔍 Test 1: Detección de Dispositivo")
        
        # Auto detection
        device_orig = _prepare_device_original(None)
        device_new = get_device('auto')
        self.assert_equal(device_orig, device_new, "Auto detection")
        
        # CPU explícito
        device_orig = _prepare_device_original('cpu')
        device_new = get_device('cpu')
        self.assert_equal(device_orig, device_new, "CPU explícito")
        
        # MPS si está disponible
        if torch.backends.mps.is_available():
            device_orig = _prepare_device_original('mps')
            device_new = get_device('mps')
            self.assert_equal(device_orig, device_new, "MPS explícito")
    
    def test_alpha_generation(self):
        """Test 2: Generación de alphas."""
        print("\n🔍 Test 2: Generación de Alphas")
        
        test_cases = [
            (0.0, 1.0, 0.1),
            (0.0, 2.0, 0.05),
            (0.0, 0.1, 0.0005),
            (0.5, 1.5, 0.01),
        ]
        
        for start, end, step in test_cases:
            alphas_orig = _generate_alphas_original(start, end, step)
            alphas_new = generate_alphas(start, end, step)
            
            self.assert_equal(
                len(alphas_orig), len(alphas_new),
                f"Cantidad alphas [{start}, {end}] step={step}"
            )
            self.assert_equal(
                alphas_orig, alphas_new,
                f"Valores alphas [{start}, {end}] step={step}"
            )
    
    def test_image_preparation(self):
        """Test 3: Preparación de imagen."""
        print("\n🔍 Test 3: Preparación de Imagen")
        
        device = get_device('auto')
        
        # Test con diferentes tamaños de imagen
        test_cases = [
            (28, 64, 4),  # MNIST -> 64x64, 4 canales
            (32, 64, 4),  # Imagen más grande
            (64, 64, 4),  # Sin redimensionamiento
            (28, 32, 8),  # Más canales
        ]
        
        for img_h, target_size, channels in test_cases:
            # Crear imagen de prueba
            img = torch.randn(1, img_h, img_h)
            
            # Preparar con ambas implementaciones
            # IMPORTANTE: Usar la misma imagen para ambos
            img_orig = _prepare_image_original(img.clone(), target_size, channels, device)
            img_new = prepare_image(img.clone(), target_size, channels, device)
            
            self.assert_equal(
                img_orig.shape, img_new.shape,
                f"Shape {img_h}x{img_h} -> {target_size}x{target_size}, ch={channels}"
            )
            
            # Verificar que los valores son idénticos
            self.assert_equal(
                img_orig, img_new,
                f"Valores {img_h}x{img_h} -> {target_size}x{target_size}, ch={channels}",
                tolerance=1e-6  # Tolerancia ligeramente mayor para interpolación
            )
    
    def test_alpha_critical_calculation(self):
        """Test 4: Cálculo de alpha crítico completo."""
        print("\n🔍 Test 4: Cálculo de Alpha Crítico (End-to-End)")
        
        # Configuración
        torch.manual_seed(42)
        np.random.seed(42)
        
        device = get_device('auto')
        img_size = 64
        ch = 4
        timesteps = 50
        gamma = 0.7
        delta_t = 0.9
        window = 5
        
        # Cargar una imagen real de MNIST
        loader = MNISTLoader(root=str(Path("../data")), img_size=28)
        dataset = loader.train_dataset
        img, label = dataset[0]  # Primera imagen
        
        print(f"  Usando imagen clase {label}")
        
        # Crear modelo
        kblock = KBlock(
            n=ch,
            ch=ch,
            connectivity="conv",
            T=timesteps,
            ksize=3,
            init_omg=0.1,
            c_norm=None,
            use_omega_c=False,
        ).to(device)
        kblock.eval()
        
        # Generar alphas
        alphas_orig = _generate_alphas_original(0.0, 0.1, 0.005)
        alphas_new = generate_alphas(0.0, 0.1, 0.005)
        
        self.assert_equal(alphas_orig, alphas_new, "Alphas generados")
        
        # Preparar imagen CON LA MISMA SEMILLA
        torch.manual_seed(42)
        c_base_orig = _prepare_image_original(img.clone(), img_size, ch, device)
        
        torch.manual_seed(42)
        c_base_new = prepare_image(img.clone(), img_size, ch, device)
        
        self.assert_equal(c_base_orig, c_base_new, "Imagen preparada", tolerance=1e-6)
        
        # Calcular alpha crítico con SEMILLA FIJA para cada método
        print("  Calculando alpha crítico con implementación original...")
        torch.manual_seed(42)
        np.random.seed(42)
        alpha_c_orig, curve_orig = calculate_alpha_critical(
            kblock, c_base_orig, alphas_orig,
            timesteps, gamma, delta_t, window, device
        )
        
        print("  Calculando alpha crítico con implementación refactorizada...")
        torch.manual_seed(42)
        np.random.seed(42)
        alpha_c_new, curve_new = calculate_alpha_critical(
            kblock, c_base_new, alphas_new,
            timesteps, gamma, delta_t, window, device
        )
        
        print(f"\n  Alpha crítico original: {alpha_c_orig:.6f}")
        print(f"  Alpha crítico refactorizado: {alpha_c_new:.6f}")
        
        # Comparar resultados
        self.assert_equal(
            curve_orig, curve_new,
            "Curva R(α) completa",
            tolerance=1e-5
        )
        
        self.assert_equal(
            alpha_c_orig, alpha_c_new,
            "Alpha crítico final",
            tolerance=1e-6
        )
    
    def run_all(self):
        """Ejecuta todos los tests."""
        print("=" * 70)
        print("  SUITE DE VALIDACIÓN - REFACTORIZACIÓN")
        print("=" * 70)
        print("\nValidando que la refactorización NO cambia los resultados...")
        
        self.test_device_detection()
        self.test_alpha_generation()
        self.test_image_preparation()
        self.test_alpha_critical_calculation()
        
        # Resumen
        print("\n" + "=" * 70)
        print("  RESUMEN")
        print("=" * 70)
        print(f"✅ Tests pasados: {self.passed}")
        print(f"❌ Tests fallidos: {self.failed}")
        
        if self.failed == 0:
            print("\n🎉 ÉXITO: Todos los tests pasaron!")
            print("   La refactorización preserva exactamente los resultados originales.")
            return 0
        else:
            print(f"\n⚠️  ADVERTENCIA: {self.failed} test(s) fallaron")
            print("   Revisar las diferencias arriba.")
            return 1


def main():
    parser = argparse.ArgumentParser(
        description="Validar que la refactorización preserva los resultados"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mostrar todos los resultados de tests"
    )
    args = parser.parse_args()
    
    tester = TestRefactorizacion(verbose=args.verbose)
    exit_code = tester.run_all()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
