# Guía de Migración a Utilidades Refactorizadas

**Fecha:** 1 de noviembre de 2025  
**Versión:** 1.0  
**Estado:** Fase 1 completada

---

## 📋 Resumen

Esta guía explica cómo migrar los scripts existentes para usar las nuevas utilidades centralizadas en `utils/`.

### Estructura Creada

```
analisis_alpha_c/
├── utils/
│   ├── __init__.py           # Exports principales
│   ├── device_utils.py       # get_device(), print_device_info()
│   ├── alpha_utils.py        # generate_alphas(), find_alpha_critical()
│   ├── image_utils.py        # prepare_image(), normalize/denormalize
│   ├── checkpoint_utils.py   # load/save_checkpoint(), find_checkpoints()
│   ├── plot_config.py        # setup_matplotlib(), save_figure()
│   └── config.py             # ModelConfig, paths, constantes
├── ejemplo_uso_utils.py      # Ejemplos de uso
└── MIGRACION.md              # Este archivo
```

---

## 🔄 Cambios por Categoría

### 1. Detección de Dispositivo

**Antes:**
```python
def _prepare_device(explicit: str | None = None) -> torch.device:
    if explicit is not None:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = _prepare_device(args.device)
```

**Después:**
```python
from utils import get_device

device = get_device(args.device or 'auto')
```

**Archivos a migrar:**
- `encontrar_alpha_critico.py`
- `encontrar_alpha_critico_clase3.py`
- `analizar_imagen_individual.py`
- `test_single_image_referencia.py`
- `visualizar_puntos_criticos.py`
- `visualizar_clase_unica.py`
- `generar_graficas_por_clase.py`

---

### 2. Generación de Alphas

**Antes:**
```python
def _generate_alphas(start: float, end: float, step: float) -> np.ndarray:
    count = int(np.floor((end - start) / step)) + 1
    return start + step * np.arange(count, dtype=np.float64)

alphas = _generate_alphas(args.alpha_start, args.alpha_end, args.alpha_step)
```

**Después:**
```python
from utils import generate_alphas

alphas = generate_alphas(args.alpha_start, args.alpha_end, args.alpha_step)
```

**Archivos a migrar:**
- `encontrar_alpha_critico.py`
- `encontrar_alpha_critico_clase3.py`

---

### 3. Preparación de Imágenes

**Antes:**
```python
def _prepare_image(img: torch.Tensor, img_size: int, ch: int, device: torch.device) -> torch.Tensor:
    if img.shape[-1] != img_size:
        img = F.interpolate(img.unsqueeze(0), size=(img_size, img_size), 
                          mode="bilinear", align_corners=False)[0]
    img_channels = img.repeat(ch, 1, 1)
    return img_channels.to(device)

img_prepared = _prepare_image(img, 64, 4, device)
```

**Después:**
```python
from utils import prepare_image

img_prepared = prepare_image(img, img_size=64, ch=4, device=device)
```

**Archivos a migrar:**
- `encontrar_alpha_critico.py`
- `encontrar_alpha_critico_clase3.py`
- `analizar_imagen_individual.py`
- `test_single_image_referencia.py`
- `visualizar_puntos_criticos.py`

---

### 4. Configuración de Matplotlib

**Antes:**
```python
def _setup_mpl_cache():
    try:
        mplconfigdir = Path.cwd() / "_mplconfig"
        mplconfigdir.mkdir(exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mplconfigdir)
    except Exception:
        pass

_setup_mpl_cache()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

**Después:**
```python
from utils import setup_matplotlib

setup_matplotlib(backend='Agg', cache_dir='_mplconfig')
import matplotlib.pyplot as plt
```

**Archivos a migrar:**
- `graficar_distribuciones_checkpoints_impares.py`
- `visualizar_puntos_criticos.py`
- `visualizar_clase_unica.py`
- `generar_graficas_checkpoints.py`
- `generar_graficas_por_clase.py`
- `generar_graficas_pdf.py`
- `analizar_imagen_individual.py`
- `graficar_distribucion_criticalidad.py`

---

### 5. Guardado de Figuras

**Antes:**
```python
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=150)
```

**Después:**
```python
from utils import save_figure

save_figure(fig, output_file)  # PNG, DPI 300
save_figure(fig, filename, format='pdf', dpi=150)  # PDF custom
```

**Archivos a migrar:** Todos los scripts de visualización

---

### 6. Manejo de Checkpoints

**Antes:**
```python
data = torch.load(checkpoint_path, weights_only=False)
alphas_c = data['alphas_c']
clase = data['clase']

torch.save(data, checkpoint_path)

files = sorted(checkpoints_dir.glob('checkpoint_clase*_0100imgs.pt'))
```

**Después:**
```python
from utils import load_checkpoint, save_checkpoint, find_checkpoints

data = load_checkpoint(checkpoint_path)
alphas_c = data['alphas_c']
clase = data['clase']

save_checkpoint(data, checkpoint_path)

files = find_checkpoints(checkpoints_dir, pattern='checkpoint_clase*_0100imgs.pt')
```

**Archivos a migrar:**
- `graficar_distribuciones_checkpoints_impares.py`
- `generar_graficas_checkpoints.py`
- `analizar_incremental_clases_impares.py`
- `verificar_consistencia.py`

---

### 7. Configuración del Modelo

**Antes:**
```python
parser.add_argument("--gamma", type=float, default=0.7)
parser.add_argument("--delta-t", type=float, default=0.9)
parser.add_argument("--timesteps", type=int, default=100)
parser.add_argument("--ksize", type=int, default=7)
parser.add_argument("--ch", type=int, default=4)
parser.add_argument("--img-size", type=int, default=64)
parser.add_argument("--init-omg", type=float, default=0.1)

# ... luego usar args.gamma, args.delta_t, etc.
```

**Después:**
```python
from utils import ModelConfig

# Opción 1: Usar config por defecto
config = ModelConfig()

# Opción 2: Crear desde args (mantiene compatibilidad CLI)
config = ModelConfig.from_args(args)

# Usar config.gamma, config.delta_t, etc.
```

**Archivos a migrar:**
- `encontrar_alpha_critico.py`
- `encontrar_alpha_critico_clase3.py`
- `analizar_imagen_individual.py`
- `test_single_image_referencia.py`

---

## 📝 Plan de Migración por Script

### Prioridad Alta (Scripts más usados)

#### 1. `analizar_con_sqlite.py` ⚠️ NO TOCAR AHORA
**Razón:** Proceso en ejecución (85/60,000 imágenes, ~43 días restantes)  
**Acción:** Migrar después de completar ejecución

#### 2. `encontrar_alpha_critico.py` y `encontrar_alpha_critico_clase3.py`
**Cambios necesarios:**
- [ ] Importar `from utils import get_device, generate_alphas, prepare_image`
- [ ] Reemplazar `_prepare_device()` → `get_device()`
- [ ] Reemplazar `_generate_alphas()` → `generate_alphas()`
- [ ] Reemplazar `_prepare_image()` → `prepare_image()`
- [ ] (Opcional) Usar `ModelConfig` para parámetros

**Estimado:** 15-20 minutos por script

#### 3. `visualizar_alpha_critico.py`
**Cambios necesarios:**
- [ ] Importar `from utils import setup_matplotlib, save_figure`
- [ ] Reemplazar setup matplotlib → `setup_matplotlib()`
- [ ] Reemplazar `plt.savefig()` → `save_figure()`

**Estimado:** 10 minutos

### Prioridad Media

#### 4-6. Scripts de generación de gráficas
- `generar_graficas_checkpoints.py`
- `generar_graficas_por_clase.py`
- `generar_graficas_pdf.py`

**Cambios comunes:**
- [ ] Setup matplotlib
- [ ] Manejo de checkpoints
- [ ] Guardado de figuras

**Estimado:** 15 minutos cada uno

#### 7-8. Scripts de visualización
- `visualizar_puntos_criticos.py`
- `visualizar_clase_unica.py`

**Estimado:** 15 minutos cada uno

### Prioridad Baja

Resto de scripts de análisis y verificación.

---

## ✅ Checklist de Migración

Para cada script que migres:

1. **Crear rama de git:**
   ```bash
   git checkout -b refactor/migrate-<nombre-script>
   ```

2. **Backup del script original:**
   ```bash
   cp script.py script.py.bak
   ```

3. **Agregar imports:**
   ```python
   from utils import (
       get_device,
       generate_alphas,
       prepare_image,
       setup_matplotlib,
       save_figure,
       ModelConfig,
       # ... otros según necesidad
   )
   ```

4. **Reemplazar funciones duplicadas:**
   - Eliminar definiciones locales de `_prepare_device`, `_generate_alphas`, etc.
   - Reemplazar llamadas por versiones de utils

5. **Actualizar configuración matplotlib:**
   - Eliminar código de setup manual
   - Agregar `setup_matplotlib()` al inicio

6. **Actualizar guardado de figuras:**
   - Reemplazar `plt.savefig()` por `save_figure()`

7. **Testear el script:**
   ```bash
   python script.py --help  # Verificar argumentos
   python script.py [args]  # Ejecutar con parámetros de prueba
   ```

8. **Comparar resultados:**
   - Verificar que genera mismos outputs
   - Comparar tiempos de ejecución

9. **Commit y push:**
   ```bash
   git add script.py
   git commit -m "refactor: migrate script.py to use utils/"
   git push origin refactor/migrate-<nombre-script>
   ```

10. **Eliminar backup si todo OK:**
    ```bash
    rm script.py.bak
    ```

---

## 🧪 Testing

### Test Rápido de Utilidades

```bash
# Ejecutar ejemplo de uso
cd analisis_alpha_c
python ejemplo_uso_utils.py
```

**Output esperado:**
```
======================================================================
EJEMPLO DE USO DE UTILIDADES REFACTORIZADAS
======================================================================

1. Configuración de dispositivo:
   Dispositivo seleccionado: mps

2. Generación de alphas:
   Generados 21 valores de alpha
   Rango: [0.000000, 0.100000]
...
```

### Test Individual de Módulos

```python
# Test device_utils
from utils import get_device
device = get_device('auto')
assert device.type in ['cuda', 'mps', 'cpu']

# Test alpha_utils
from utils import generate_alphas
alphas = generate_alphas(0.0, 1.0, 0.1)
assert len(alphas) == 11
assert alphas[0] == 0.0
assert alphas[-1] == 1.0

# Test image_utils
from utils import prepare_image
import torch
img = torch.randn(1, 28, 28)
prepared = prepare_image(img, img_size=64, ch=4, device='cpu')
assert prepared.shape == (4, 64, 64)
```

---

## 📊 Progreso de Migración

### Estado Actual: Fase 1 Completada ✅

- [x] Crear estructura `utils/`
- [x] Implementar `device_utils.py`
- [x] Implementar `alpha_utils.py`
- [x] Implementar `image_utils.py`
- [x] Implementar `checkpoint_utils.py`
- [x] Implementar `plot_config.py`
- [x] Implementar `config.py`
- [x] Crear script de ejemplo
- [x] Crear guía de migración

### Siguiente: Fase 2 - Migración Gradual

Scripts a migrar (orden sugerido):

1. [ ] `encontrar_alpha_critico.py`
2. [ ] `encontrar_alpha_critico_clase3.py`
3. [ ] `visualizar_alpha_critico.py`
4. [ ] `generar_graficas_checkpoints.py`
5. [ ] `generar_graficas_por_clase.py`
6. [ ] `generar_graficas_pdf.py`
7. [ ] `visualizar_puntos_criticos.py`
8. [ ] `visualizar_clase_unica.py`
9. [ ] `graficar_distribuciones_checkpoints_impares.py`
10. [ ] `graficar_distribucion_criticalidad.py`
11. [ ] `analizar_imagen_individual.py`
12. [ ] `test_single_image_referencia.py`
13. [ ] `verificar_scripts_alpha_c.py`
14. [ ] `verificar_consistencia.py`
15. [ ] `analizar_incremental_clases_impares.py`
16. [ ] `analizar_con_sqlite.py` ⚠️ (después de finalizar ejecución)

---

## 🔗 Referencias

- **Análisis de redundancia:** `docs/01-11-2025/redundancia_analisis_alpha_c.md`
- **Ejemplos de uso:** `ejemplo_uso_utils.py`
- **Configuración:** `utils/config.py`
- **README principal:** `README.md`

---

## 💡 Consejos

1. **No migres todo a la vez:** Hazlo gradualmente, script por script
2. **Testea cada migración:** Verifica que el output no cambia
3. **Mantén backups:** Guarda `.bak` hasta confirmar que funciona
4. **Documenta cambios:** Usa commits descriptivos
5. **Pregunta si dudas:** Revisa ejemplos en `ejemplo_uso_utils.py`

---

**¡Buena suerte con la migración! 🚀**
