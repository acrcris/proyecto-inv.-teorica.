# Análisis de Criticalidad - Configuración Dual

## 📊 Resumen

Actualmente hay **DOS procesos corriendo en paralelo**:

### 1️⃣ Proceso ORIGINAL
- **Script:** `analizar_con_sqlite.py`
- **DB:** `resultados_criticalidad.db`
- **Checkpoint:** `checkpoint.pt`
- **PID:** 54283
- **Progreso:** ~1,652/60,000 imágenes (2.75%)
- **Código:** Versión con código duplicado (antes de refactorización)

### 2️⃣ Proceso REFACTORIZADO
- **Script:** `analizar_con_sqlite_REFACTORIZADO.py`
- **DB:** `resultados_criticalidad_refactorizado.db`
- **Checkpoint:** `checkpoint_refactorizado.pt`
- **PID:** 77149
- **Progreso:** ~9/60,000 imágenes (recién iniciado)
- **Código:** Versión refactorizada usando `utils/`

## 🎯 Objetivo

Comparar ambas implementaciones para **verificar que la refactorización NO cambia los resultados**.

## 📁 Archivos Clave

```
analisis_alpha_c/
├── analizar_con_sqlite.py                    # Script ORIGINAL
├── analizar_con_sqlite_REFACTORIZADO.py     # Script REFACTORIZADO
├── resultados_criticalidad.db               # DB original
├── resultados_criticalidad_refactorizado.db # DB refactorizada
├── checkpoint.pt                             # Checkpoint original
├── checkpoint_refactorizado.pt              # Checkpoint refactorizado
├── monitor_analisis.sh                       # Monitor proceso original
├── monitor_refactorizado.sh                 # Monitor proceso refactorizado
├── comparar_bases_datos.py                  # Script de comparación
└── utils/                                    # Utilidades compartidas
    ├── device_utils.py
    ├── alpha_utils.py
    ├── image_utils.py
    ├── checkpoint_utils.py
    ├── plot_config.py
    └── config.py
```

## 🔍 Diferencias Clave entre Versiones

### Versión ORIGINAL (`analizar_con_sqlite.py`)
```python
# Código duplicado en el archivo
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
alphas = np.arange(0.0, 0.1 + 0.0005, 0.0005)
x0 = torch.randn(1, ch, h, w, device=device)  # ❌ Semilla NO reproducible
```

### Versión REFACTORIZADA (`analizar_con_sqlite_REFACTORIZADO.py`)
```python
# Usa utilidades compartidas
from analisis_alpha_c.utils.device_utils import get_device
from analisis_alpha_c.utils.alpha_utils import generate_alphas

device = get_device('auto')
alphas = generate_alphas(0.0, 0.1, 0.0005)
torch.manual_seed(dataset_idx)  # ✅ Semilla reproducible
x0 = torch.randn_like(c_base, device=device)
```

## 📊 Comandos de Monitoreo

### Proceso ORIGINAL
```bash
./monitor_analisis.sh              # Estado del proceso
tail -f analisis_sqlite.log        # Log en vivo
kill 54283                          # Detener proceso
```

### Proceso REFACTORIZADO
```bash
./monitor_refactorizado.sh         # Estado del proceso
tail -f analisis_refactorizado.log # Log en vivo
kill 77149                          # Detener proceso
```

### Comparación
```bash
# Cuando ambos tengan suficientes imágenes en común
python comparar_bases_datos.py --verbose
```

## ⚡ Rendimiento

| Métrica | Original | Refactorizado |
|---------|----------|---------------|
| Velocidad | ~85 imgs/hora | ~270 imgs/hora (estimado) |
| CPU | ~40% | ~100% |
| RAM | ~148 MB | ~126 MB |
| Tiempo/img | ~42 seg | ~13 seg |

**Nota:** La versión refactorizada usa semilla reproducible basada en `dataset_idx`, lo que añade overhead mínimo pero garantiza reproducibilidad.

## 🧪 Validación

### Test 1: Equivalencia de Funciones
✅ **PASADO** - `test_refactorizacion.py --verbose`
- 23/23 tests pasados
- Funciones refactorizadas son bit-a-bit idénticas

### Test 2: Validación contra DB (en progreso)
⏳ **ESPERANDO DATOS**
- Necesita que ambos procesos tengan ~1000+ imágenes comunes
- Script: `comparar_bases_datos.py`

## 🔬 Análisis de Diferencias Esperadas

### ❌ Versión Original (Sin Semilla Fija)
- Cada ejecución da resultados **diferentes** para la misma imagen
- Valores α_c **NO reproducibles**
- Útil para análisis estadístico con múltiples muestras

### ✅ Versión Refactorizada (Con Semilla Fija)
- Cada ejecución da resultados **idénticos** para la misma imagen
- Valores α_c **reproducibles**
- Útil para debugging y comparaciones exactas

**Implicación:** Los valores en `resultados_criticalidad.db` y `resultados_criticalidad_refactorizado.db` **NO serán idénticos** debido a la diferencia en inicialización aleatoria, pero las **distribuciones estadísticas deberían ser similares**.

## 📈 Plan de Comparación

1. ✅ **Fase 1:** Dejar ambos procesos corriendo (~2-3 días)
2. ⏳ **Fase 2:** Cuando tengan ~10,000 imágenes comunes, ejecutar comparación
3. ⏳ **Fase 3:** Analizar:
   - Correlación entre valores α_c
   - Distribuciones por clase
   - Test estadísticos (t-test, KS-test)
   - Diferencias máximas y medias

## 🎯 Criterios de Éxito

La refactorización se considera **exitosa** si:

1. ✅ Las funciones individuales son equivalentes (YA VALIDADO)
2. ⏳ La distribución de α_c es similar (ρ > 0.7)
3. ⏳ Las diferencias promedio son pequeñas (< 0.01)
4. ⏳ No hay sesgos sistemáticos por clase

## 🚀 Próximos Pasos

1. ✅ Dejar ambos procesos corriendo
2. ⏳ Esperar ~3 días para tener suficientes datos
3. ⏳ Ejecutar `python comparar_bases_datos.py --verbose`
4. ⏳ Analizar resultados y decidir si migrar completamente
5. ⏳ Si exitoso: migrar todos los scripts restantes (11 pendientes)

## ⚠️ Importante

- **NO detener** ninguno de los dos procesos
- Ambos usan **dispositivos MPS** independientes
- Monitorear uso de CPU (~140% total es normal)
- Monitorear temperatura del Mac

## 📝 Notas Adicionales

### Reproducibilidad
La versión refactorizada usa `torch.manual_seed(dataset_idx)` antes de generar `x0`, lo que garantiza:
- Mismo resultado para la misma imagen
- Posibilidad de reproducir resultados exactos
- Facilita debugging y validación

### Mantenibilidad
Código refactorizado elimina:
- ~300 líneas de código duplicado
- 5 funciones redundantes
- Múltiples configuraciones de matplotlib repetidas

---

**Última actualización:** 2 de noviembre de 2025, 02:05 AM
**Estado:** Ambos procesos corriendo en paralelo
