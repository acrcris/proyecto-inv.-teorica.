# 📊 Progreso de Migración - Fase 2

**Fecha:** 2 de noviembre de 2025, 01:15 AM  
**Carpeta:** `codigo/analisis_criticalidad_minimalista/analisis_alpha_c/`

---

## ✅ Scripts Migrados (3/17)

### 1. `encontrar_alpha_critico.py` ✅
**Cambios aplicados:**
- ✅ Importar `get_device, generate_alphas, prepare_image` desde `utils`
- ✅ Eliminar funciones duplicadas `_prepare_device`, `_generate_alphas`, `_prepare_image`
- ✅ Reemplazar llamadas a funciones locales por `utils.*`
- ✅ Remover import `torch.nn.functional as F` (no usado)

**Reducción:** ~35 líneas de código duplicado

### 2. `encontrar_alpha_critico_clase3.py` ✅
**Cambios aplicados:**
- ✅ Importar `get_device, generate_alphas, prepare_image` desde `utils`
- ✅ Eliminar funciones duplicadas `_prepare_device`, `_generate_alphas`, `_prepare_image`
- ✅ Reemplazar llamadas: `device = get_device(args.device or 'auto')`
- ✅ Reemplazar llamadas: `alphas = generate_alphas(...)`
- ✅ Reemplazar llamadas: `c_base = prepare_image(...)`

**Reducción:** ~35 líneas de código duplicado

### 3. `visualizar_puntos_criticos.py` ✅
**Cambios aplicados:**
- ✅ Importar `setup_matplotlib, save_figure` desde `utils`
- ✅ Eliminar configuración manual de matplotlib (backend + cache)
- ✅ Reemplazar `plt.savefig(...)` por `save_figure(fig, ...)` (3 ocurrencias)
- ✅ Simplificar imports

**Reducción:** ~20 líneas de configuración duplicada

---

## ⏸️ Scripts Pendientes (14/17)

### 📊 Visualización/Gráficas (7 scripts)
1. **`visualizar_alpha_critico.py`**
   - Patrón: matplotlib setup + device
   - Migración: setup_matplotlib(), get_device(), save_figure()

2. **`visualizar_clase_unica.py`**
   - Patrón: matplotlib setup + device + prepare_image
   - Migración: setup_matplotlib(), get_device(), prepare_image(), save_figure()

3. **`generar_graficas_checkpoints.py`**
   - Patrón: matplotlib setup + load checkpoints
   - Migración: setup_matplotlib(), load_checkpoint(), save_figure()

4. **`generar_graficas_pdf.py`**
   - Patrón: matplotlib setup + PDF export
   - Migración: setup_matplotlib(), save_figure(format='pdf')

5. **`generar_graficas_por_clase.py`**
   - Patrón: matplotlib setup + device
   - Migración: setup_matplotlib(), get_device(), save_figure()

6. **`graficar_distribucion_criticalidad.py`**
   - Patrón: matplotlib setup simple
   - Migración: setup_matplotlib(), save_figure()

7. **`graficar_distribuciones_checkpoints_impares.py`**
   - Patrón: matplotlib setup + load checkpoints
   - Migración: setup_matplotlib(), load_checkpoint(), save_figure()

### 🔬 Análisis (4 scripts)
8. **`analizar_imagen_individual.py`**
   - Patrón: device + prepare_image + matplotlib
   - Migración: get_device(), prepare_image(), setup_matplotlib(), save_figure()

9. **`analizar_incremental_clases_impares.py`**
   - Patrón: device + load/save checkpoints
   - Migración: get_device(), load_checkpoint(), save_checkpoint()

10. **`test_single_image_referencia.py`**
    - Patrón: device + prepare_image
    - Migración: get_device(), prepare_image()

11. **`verificar_consistencia.py`**
    - Patrón: load checkpoints
    - Migración: load_checkpoint()

### 🛠️ Utilidades (3 scripts)
12. **`ejemplo_uso_utils.py`** ⭐ Ya usa utils (ejemplo)
    - No requiere migración

13. **`verificar_scripts_alpha_c.py`**
    - Script de auditoría
    - Migración mínima o ninguna

14. **`analizar_con_sqlite.py`** ⚠️ **CRÍTICO**
    - Script activo del análisis
    - Ya tiene timeout/retry mejorado
    - Migración: device + generate_alphas + prepare_image
    - **NOTA:** NO migrar mientras esté corriendo

---

## 📈 Estadísticas

| Métrica | Valor |
|---------|-------|
| **Scripts migrados** | 3/17 (17.6%) |
| **Código eliminado** | ~90 líneas |
| **Funciones duplicadas eliminadas** | 6 |
| **Usos de utils agregados** | 9 |

### Reducción de Código por Categoría

| Patrón Eliminado | Ocurrencias | Líneas Ahorradas |
|------------------|-------------|------------------|
| `_prepare_device()` | 2 | ~20 |
| `_generate_alphas()` | 2 | ~10 |
| `_prepare_image()` | 2 | ~20 |
| Matplotlib setup | 1 | ~15 |
| `plt.savefig()` → `save_figure()` | 3 | ~0 (simplificación) |
| Imports reducidos | 3 | ~5 |

---

## 🎯 Plan de Continuación

### Fase 2A: Scripts de Visualización (Prioridad Alta)
```bash
# Batch 1: Visualización simple (2-3 horas)
1. visualizar_alpha_critico.py
2. visualizar_clase_unica.py
3. graficar_distribucion_criticalidad.py

# Batch 2: Generación de gráficas (2 horas)
4. generar_graficas_checkpoints.py
5. generar_graficas_pdf.py
6. generar_graficas_por_clase.py
7. graficar_distribuciones_checkpoints_impares.py
```

### Fase 2B: Scripts de Análisis (Prioridad Media)
```bash
# Batch 3: Análisis individual (1-2 horas)
8. analizar_imagen_individual.py
9. test_single_image_referencia.py

# Batch 4: Análisis con checkpoints (1 hora)
10. analizar_incremental_clases_impares.py
11. verificar_consistencia.py
```

### Fase 2C: Scripts Críticos (Después del análisis)
```bash
# Batch 5: Solo cuando NO esté corriendo (1 hora)
12. analizar_con_sqlite.py
```

---

## 🚀 Siguiente Acción Recomendada

### Opción 1: Continuar Migración Manual
Migrar los 7 scripts de visualización restantes uno por uno siguiendo el mismo patrón.

### Opción 2: Crear Script de Migración Automática
Escribir un script Python que aplique las transformaciones automáticamente usando AST.

### Opción 3: Pausar y Reiniciar Análisis
1. Detener migración temporalmente
2. Reiniciar `analizar_con_sqlite.py`
3. Continuar migración después del análisis (~13 días)

---

## 💡 Recomendación

**Opción 1** - Continuar migración manual para los 7 scripts de visualización.

**Razón:**
- Patrones muy similares (fácil de migrar)
- No afectan el análisis en ejecución
- Se completa en ~4-5 horas
- Validación inmediata de beneficios

**Después:**
- Reiniciar análisis SQLite
- Migrar scripts de análisis después de que termine

---

## 📝 Comandos para Testing

```bash
# Verificar imports
cd analisis_alpha_c
python -c "from utils import get_device, generate_alphas, prepare_image, setup_matplotlib, save_figure; print('✅ Imports OK')"

# Test rápido de scripts migrados
python encontrar_alpha_critico.py --help
python encontrar_alpha_critico_clase3.py --help
python visualizar_puntos_criticos.py --help
```

---

## ✅ Estado de Ejecución

- ✅ Fase 1 completada (utils/ creados)
- ⏸️ Fase 2 en progreso (3/17 scripts migrados - 17.6%)
- ⏸️ Análisis SQLite pausado (1,579/60,000 imágenes)
- 📊 Listo para continuar migración o reiniciar análisis
