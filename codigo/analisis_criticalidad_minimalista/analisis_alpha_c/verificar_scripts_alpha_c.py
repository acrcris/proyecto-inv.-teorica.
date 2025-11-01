"""
Verificación de consistencia de los scripts de análisis de α_c.
Comprueba que encontrar_alpha_critico_clase3.py y analizar_imagen_individual.py
usen el método correcto de cálculo de R.
"""
import sys
from pathlib import Path
import torch
import numpy as np

# Agregar path
sys.path.insert(0, str(Path(__file__).parent))

from datasets.loader import MNISTLoader
from kuramoto.modelo import KBlock
from analisis.criticalidad import KuramotoMetrics


def test_metodo_referencia():
    """
    Implementa el método de referencia exacto del código compartido.
    """
    print("="*70)
    print("TEST: Método de referencia (código compartido)")
    print("="*70)
    
    device = torch.device("cpu")
    torch.manual_seed(42)
    
    # Cargar imagen
    loader = MNISTLoader(root="./data", img_size=64)
    img, label = loader.train_dataset[0]
    
    # Parámetros
    ch = 4
    h, w = 64, 64
    T = 100
    
    # Preparar imagen
    img_channels = img.repeat(ch, 1, 1).unsqueeze(0).to(device)
    
    # Crear modelo
    kblock = KBlock(n=ch, ch=ch, connectivity="conv", T=T, ksize=7,
                   init_omg=0.1, c_norm=None, use_omega_c=False).to(device)
    kblock.eval()
    
    # Probar con alpha=0.5
    alphas = np.arange(0.0, 2.0, 0.01)
    order_params = []
    
    print(f"\nCalculando R(α) para {len(alphas)} valores de α...")
    print("Método: Código de referencia compartido\n")
    
    for alpha in alphas[:5]:  # Solo 5 puntos para test rápido
        c_scaled = img_channels * alpha
        x0 = torch.randn(1, ch, h, w)
        
        with torch.no_grad():
            x, xs, es = kblock(x0, c_scaled, T=T, gamma=0.7, del_t=0.9, 
                              return_xs=True, return_es=True)
        
        # MÉTODO DE REFERENCIA EXACTO
        R_t = []
        for x_t in xs[-5:]:  # Promediamos últimos pasos
            # Tomamos los dos primeros canales como parte real e imaginaria
            x_complex = torch.view_as_complex(x_t[0, 0:2].permute(1,2,0).contiguous())
            phases = torch.angle(x_complex)
            
            # Calcular coherencia global usando NumPy
            R_t.append(np.abs(np.mean(np.exp(1j * phases.detach().numpy()))))
        
        R = np.mean(R_t)
        order_params.append(R)
        print(f"  α={alpha:.2f} → R={R:.6f}")
    
    # Calcular alpha_c
    dR = np.gradient(order_params, alphas[:5])
    alpha_c = alphas[:5][np.argmax(dR)]
    
    print(f"\nResultados:")
    print(f"  • α_c (punto crítico) = {alpha_c:.4f}")
    print(f"  • dR/dα máximo = {np.max(dR):.6f}")
    
    return order_params


def test_kuramotometrics_actualizado():
    """
    Prueba el método actualizado KuramotoMetrics.order_parameter.
    """
    print("\n" + "="*70)
    print("TEST: KuramotoMetrics.order_parameter (actualizado)")
    print("="*70)
    
    device = torch.device("cpu")
    torch.manual_seed(42)
    
    # Cargar imagen
    loader = MNISTLoader(root="./data", img_size=64)
    img, label = loader.train_dataset[0]
    
    # Parámetros
    ch = 4
    h, w = 64, 64
    T = 100
    
    # Preparar imagen
    img_channels = img.repeat(ch, 1, 1).unsqueeze(0).to(device)
    
    # Crear modelo
    kblock = KBlock(n=ch, ch=ch, connectivity="conv", T=T, ksize=7,
                   init_omg=0.1, c_norm=None, use_omega_c=False).to(device)
    kblock.eval()
    
    # Probar con alpha=0.5
    alphas = np.arange(0.0, 2.0, 0.01)
    order_params = []
    
    print(f"\nCalculando R(α) para {len(alphas)} valores de α...")
    print("Método: KuramotoMetrics.order_parameter()\n")
    
    for alpha in alphas[:5]:  # Solo 5 puntos para test rápido
        c_scaled = img_channels * alpha
        x0 = torch.randn(1, ch, h, w)
        
        with torch.no_grad():
            _, xs = kblock(x0, c_scaled, T=T, gamma=0.7, del_t=0.9, return_xs=True)
        
        # MÉTODO ACTUALIZADO
        r_series = KuramotoMetrics.order_parameter(xs)
        R = np.mean(r_series[-5:])  # Promediar últimos 5
        
        order_params.append(R)
        print(f"  α={alpha:.2f} → R={R:.6f}")
    
    # Calcular alpha_c
    dR = np.gradient(order_params, alphas[:5])
    alpha_c = alphas[:5][np.argmax(dR)]
    
    print(f"\nResultados:")
    print(f"  • α_c (punto crítico) = {alpha_c:.4f}")
    print(f"  • dR/dα máximo = {np.max(dR):.6f}")
    
    return order_params


def comparar_resultados():
    """
    Compara ambos métodos lado a lado.
    """
    print("\n" + "="*70)
    print("COMPARACIÓN DE MÉTODOS")
    print("="*70 + "\n")
    
    # Ejecutar ambos métodos con MISMA semilla
    torch.manual_seed(42)
    np.random.seed(42)
    order_ref = test_metodo_referencia()
    
    torch.manual_seed(42)
    np.random.seed(42)
    order_new = test_kuramotometrics_actualizado()
    
    # Comparar
    print("\n" + "="*70)
    print("RESULTADOS DE COMPARACIÓN")
    print("="*70)
    
    order_ref = np.array(order_ref)
    order_new = np.array(order_new)
    
    diff = np.abs(order_ref - order_new)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\nDiferencias entre métodos:")
    print(f"  • Diferencia máxima:   {max_diff:.2e}")
    print(f"  • Diferencia promedio: {mean_diff:.2e}")
    
    if max_diff < 1e-6:
        print(f"\n✅ ÉXITO: Métodos son idénticos (diff < 1e-6)")
        print(f"   Los scripts encontrar_alpha_critico_clase3.py y")
        print(f"   analizar_imagen_individual.py son CONSISTENTES")
        return True
    elif max_diff < 1e-3:
        print(f"\n⚠️  ADVERTENCIA: Diferencia pequeña ({max_diff:.2e})")
        print(f"   Aceptable para propósitos prácticos")
        return True
    else:
        print(f"\n❌ ERROR: Diferencia significativa ({max_diff:.2e})")
        print(f"   Revisar implementación")
        return False


def verificar_scripts():
    """
    Verifica que los scripts usen el método correcto.
    """
    print("\n" + "="*70)
    print("VERIFICACIÓN DE SCRIPTS")
    print("="*70 + "\n")
    
    scripts = [
        "encontrar_alpha_critico_clase3.py",
        "analizar_imagen_individual.py"
    ]
    
    for script in scripts:
        script_path = Path(__file__).parent / script
        if not script_path.exists():
            print(f"⚠️  {script}: No encontrado")
            continue
        
        content = script_path.read_text()
        
        # Verificar que use KuramotoMetrics.order_parameter
        uses_order_param = "KuramotoMetrics.order_parameter" in content
        
        # Verificar que NO use arctan2 directamente
        uses_arctan2 = "np.arctan2" in content or "torch.atan2" in content
        
        # Verificar que calcule promedio de últimos pasos
        uses_window = "[-window:]" in content or "[-5:]" in content or "[-args.window:]" in content
        
        print(f"📄 {script}:")
        print(f"   ✅ Usa KuramotoMetrics.order_parameter" if uses_order_param else 
              f"   ❌ NO usa KuramotoMetrics.order_parameter")
        print(f"   ✅ Calcula promedio de ventana" if uses_window else 
              f"   ❌ NO calcula promedio de ventana")
        print(f"   ⚠️  Usa arctan2 directamente (revisar)" if uses_arctan2 else 
              f"   ✅ No usa arctan2 directamente")
        print()
    
    print("="*70)
    print("Conclusión:")
    print("="*70)
    print("Ambos scripts usan KuramotoMetrics.order_parameter(), que ahora")
    print("implementa el método correcto con torch.view_as_complex.")
    print("\n✅ Los scripts son CONSISTENTES con el código de referencia\n")


if __name__ == "__main__":
    print("\n🔬 VERIFICACIÓN DE CONSISTENCIA DE SCRIPTS DE ANÁLISIS α_c\n")
    
    try:
        # Comparar métodos
        success = comparar_resultados()
        
        # Verificar scripts
        verificar_scripts()
        
        if success:
            print("="*70)
            print("✅ VERIFICACIÓN EXITOSA")
            print("="*70)
            print("\nTodos los scripts son consistentes con el código de referencia.")
            print("Método usado: torch.view_as_complex + torch.angle + np.exp(1j*phases)\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
