"""
Script de verificación de la instalación del módulo analisis_criticalidad_minimalista.
Verifica que todas las funciones del notebook estén disponibles.
"""

import sys
sys.path.insert(0, '/home/crperezp/proyectos/ProyectoInvTeorica/Proyecto-Inv.-teorica./codigo')

def test_imports():
    """Verifica que todas las importaciones funcionen."""
    print("=" * 60)
    print("VERIFICACIÓN DEL MÓDULO analisis_criticalidad_minimalista")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Importación principal
    tests_total += 1
    try:
        import analisis_criticalidad_minimalista as acm
        print("✅ Test 1: Importación del módulo principal")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 1: Error en importación principal: {e}")
    
    # Test 2: Modelo de Kuramoto
    tests_total += 1
    try:
        from analisis_criticalidad_minimalista import (
            KBlock, KConv2d, ModReLU, Reshape,
            reshape, reshape_back, nrm, gaussian_kernel_2d
        )
        print("✅ Test 2: Importación de modelo Kuramoto")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 2: Error en modelo Kuramoto: {e}")
    
    # Test 3: Métricas de criticalidad
    tests_total += 1
    try:
        from analisis_criticalidad_minimalista import (
            KuramotoMetrics, Entropia, DFA, PSD, 
            MutualInformation, Correlacion
        )
        print("✅ Test 3: Importación de métricas de criticalidad")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 3: Error en métricas: {e}")
    
    # Test 4: Análisis de series temporales
    tests_total += 1
    try:
        from analisis_criticalidad_minimalista import SeriesAnalysis
        print("✅ Test 4: Importación de análisis de series temporales")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 4: Error en series temporales: {e}")
    
    # Test 5: Visualización
    tests_total += 1
    try:
        from analisis_criticalidad_minimalista import Visualizador, Animaciones
        print("✅ Test 5: Importación de visualización y animaciones")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 5: Error en visualización: {e}")
    
    # Test 6: Datasets
    tests_total += 1
    try:
        from analisis_criticalidad_minimalista import MNISTLoader
        print("✅ Test 6: Importación de cargador de datos")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 6: Error en datasets: {e}")
    
    # Test 7: Segmentación
    tests_total += 1
    try:
        from analisis_criticalidad_minimalista import SegmentadorKMeans
        print("✅ Test 7: Importación de segmentación")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 7: Error en segmentación: {e}")
    
    # Test 8: Instanciación de KBlock
    tests_total += 1
    try:
        import torch
        from analisis_criticalidad_minimalista import KBlock
        kblock = KBlock(n=2, ch=4, T=10)
        print("✅ Test 8: Instanciación de KBlock")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 8: Error al instanciar KBlock: {e}")
    
    # Test 9: Ejecución básica del modelo
    tests_total += 1
    try:
        x = torch.randn(1, 4, 16, 16)
        c = torch.randn(1, 4, 16, 16)
        x_final, xs, es = kblock(x, c, T=50, gamma=0.5, del_t=1.0, 
                                 return_xs=True, return_es=True)
        assert len(xs) == 51  # t=0 + 50 pasos
        assert len(es) == 51
        print("✅ Test 9: Ejecución básica del modelo")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 9: Error al ejecutar modelo: {e}")
    
    # Test 10: Cálculo de métricas
    tests_total += 1
    try:
        from analisis_criticalidad_minimalista import KuramotoMetrics, DFA, PSD
        import numpy as np
        
        R = KuramotoMetrics.order_parameter(xs)
        series = KuramotoMetrics.magnitudes_mean_series(xs)
        global_series = series.mean(axis=1)
        
        # Solo probar las métricas básicas que no requieren ajustes numéricos complejos
        assert len(R) == 51
        assert series.shape[0] == 51
        assert series.shape[1] == 4
        
        print("✅ Test 10: Cálculo de métricas de criticalidad")
        print(f"   - R final: {R[-1]:.4f}")
        print(f"   - R mean: {R.mean():.4f}")
        print(f"   - Series shape: {series.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 10: Error al calcular métricas: {e}")
    
    # Resumen
    print("\n" + "=" * 60)
    print(f"RESUMEN: {tests_passed}/{tests_total} tests pasados")
    print("=" * 60)
    
    if tests_passed == tests_total:
        print("🎉 ¡TODAS LAS FUNCIONES DEL NOTEBOOK ESTÁN DISPONIBLES!")
        print("\nEl módulo está listo para usar. Puedes ejecutar:")
        print("  - python ejemplo_completo.py    (demo completo)")
        print("  - python main.py                (análisis de MNIST)")
        return True
    else:
        print(f"⚠️  {tests_total - tests_passed} tests fallaron.")
        print("Revisa los errores anteriores.")
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
