# Análisis de Redundancia - analisis_alpha_c

**Fecha:** 1 de noviembre de 2025  
**Carpeta analizada:** `codigo/analisis_criticalidad_minimalista/analisis_alpha_c/`  
**Total de archivos .py:** 16 scripts

---

## 📊 Resumen Ejecutivo

Se identificaron **5 categorías principales de redundancia** con diferentes niveles de impacto. La redundancia más significativa se encuentra en:

1. **Funciones helper duplicadas** (preparación de dispositivos, generación de alphas, preparación de imágenes)
2. **Configuración de matplotlib** (8 archivos con código idéntico)
3. **Importaciones y configuraciones** (paths, módulos)
4. **Patrones de carga de checkpoints** (4 archivos)
5. **Parámetros de modelo repetidos** (gamma, delta_t, timesteps)

**Impacto estimado:** ~30-40% de código duplicado que podría consolidarse en utilidades compartidas.

---

## 🔍 Categorías de Redundancia Detalladas

### 1. ⚙️ Funciones Helper Duplicadas (ALTA PRIORIDAD)

#### 1.1 Preparación de Dispositivo
**Archivos afectados:** 7 scripts
- `encontrar_alpha_critico.py`
- `encontrar_alpha_critico_clase3.py`
- `analizar_imagen_individual.py`
- `test_single_image_referencia.py`
- `visualizar_puntos_criticos.py`
- `visualizar_clase_unica.py`
- `generar_graficas_por_clase.py`

**Código duplicado:**
```python
def _prepare_device(explicit: str | None = None) -> torch.device:
    if explicit is not None:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

**Recomendación:**
- Crear `utils/device_utils.py` con `get_device(device_arg='auto')`
- Ya existe implementación similar en `analizar_con_sqlite.py` (línea 118)

#### 1.2 Generación de Alphas
**Archivos afectados:** 2 scripts
- `encontrar_alpha_critico.py`
- `encontrar_alpha_critico_clase3.py`

**Código duplicado:**
```python
def _generate_alphas(start: float, end: float, step: float) -> np.ndarray:
    count = int(np.floor((end - start) / step)) + 1
    return start + step * np.arange(count, dtype=np.float64)
```

**Recomendación:**
- Mover a `utils/alpha_utils.py`
- Agregar validación de parámetros (start < end, step > 0)

#### 1.3 Preparación de Imágenes
**Archivos afectados:** 5 scripts
- `encontrar_alpha_critico.py`
- `encontrar_alpha_critico_clase3.py`
- `analizar_imagen_individual.py`
- `test_single_image_referencia.py`
- `visualizar_puntos_criticos.py`

**Código duplicado:**
```python
def _prepare_image(img: torch.Tensor, img_size: int, ch: int, device: torch.device) -> torch.Tensor:
    if img.shape[-1] != img_size:
        img = F.interpolate(img.unsqueeze(0), size=(img_size, img_size), 
                          mode="bilinear", align_corners=False)[0]
    img_channels = img.repeat(ch, 1, 1)
    return img_channels.to(device)
```

**Recomendación:**
- Mover a `datasets/preprocessing.py` o `utils/image_utils.py`
- Agregar opciones para modo de interpolación configurable

---

### 2. 🎨 Configuración de Matplotlib (PRIORIDAD MEDIA)

#### 2.1 Configuración Idéntica de Backend
**Archivos afectados:** 8 scripts de visualización
- `graficar_distribuciones_checkpoints_impares.py`
- `visualizar_puntos_criticos.py`
- `visualizar_clase_unica.py`
- `generar_graficas_checkpoints.py`
- `generar_graficas_por_clase.py`
- `generar_graficas_pdf.py`
- `analizar_imagen_individual.py`
- `graficar_distribucion_criticalidad.py`

**Patrón duplicado:**
```python
import matplotlib
matplotlib.use("Agg")  # Backend no interactivo
import matplotlib.pyplot as plt
```

**Variante con configuración de caché:**
```python
def _setup_mpl_cache():
    """Configurar caché de Matplotlib en carpeta local y backend Agg."""
    try:
        mplconfigdir = Path.cwd() / "_mplconfig"
        mplconfigdir.mkdir(exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mplconfigdir)
    except Exception:
        pass  # Dejar que Matplotlib use tmp
```

**Recomendación:**
- Crear `utils/plot_config.py` con:
  - `setup_matplotlib_backend(backend='Agg', cache_dir='_mplconfig')`
  - `get_default_figure_params()` para estilos consistentes
- Usar `plt.style.use()` para estilos globales consistentes

#### 2.2 Parámetros de Guardado Repetidos
**Patrón común:**
```python
plt.savefig(output_file, dpi=300, bbox_inches='tight')  # Visualizaciones
plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=150)  # PDFs
```

**Recomendación:**
- Función helper `save_figure(fig, path, format='png', dpi=300, **kwargs)`

---

### 3. 📦 Importaciones y Paths (PRIORIDAD BAJA)

#### 3.1 Importaciones Comunes
**Todos los scripts tienen:**
```python
import sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kuramoto.modelo import KBlock
from datasets.loader import MNISTLoader
from analisis.criticalidad import KuramotoMetrics
```

**Observación:**
- El `sys.path.insert(0, ...)` es necesario para imports relativos
- No es redundancia crítica, es patrón necesario

**Recomendación:**
- Considerar usar `__init__.py` en carpeta `analisis_alpha_c` para facilitar imports
- Crear alias de importación si se usa frecuentemente

#### 3.2 Paths de Datos Repetidos
```python
default=Path("./data")  # MNIST root
default=Path("codigo/analisis_criticalidad_minimalista/checkpoints_impares")
default=Path("distribuciones_impares")
```

**Recomendación:**
- Crear `config.py` con constantes:
  ```python
  DATA_DIR = Path("./data")
  CHECKPOINTS_DIR = Path("checkpoints_impares")
  DISTRIBUTIONS_DIR = Path("distribuciones_impares")
  ```

---

### 4. 💾 Manejo de Checkpoints (PRIORIDAD MEDIA)

#### 4.1 Carga de Checkpoints PyTorch
**Archivos afectados:**
- `graficar_distribuciones_checkpoints_impares.py`
- `generar_graficas_checkpoints.py`
- `analizar_incremental_clases_impares.py`
- `verificar_consistencia.py`

**Patrón duplicado:**
```python
# Carga con manejo de weights_only
data = torch.load(checkpoint_path, weights_only=False)
alphas_c = data['alphas_c']
clase = data['clase']
num_imgs = data.get('num_imgs', len(alphas_c))
```

**Recomendación:**
- Crear `utils/checkpoint_utils.py` con:
  ```python
  def load_checkpoint(path: Path) -> dict:
      """Carga checkpoint con manejo de PyTorch 2.6+"""
      return torch.load(path, weights_only=False)
  
  def save_checkpoint(data: dict, path: Path):
      """Guarda checkpoint de forma consistente"""
      torch.save(data, path)
  ```

#### 4.2 Búsqueda de Checkpoints
**Patrón común:**
```python
files = sorted(checkpoints_dir.glob('checkpoint_clase*_0100imgs.pt'))
if not files:
    raise FileNotFoundError(f"No se encontraron archivos *_0100imgs.pt")
```

**Recomendación:**
- Función `find_checkpoints(dir: Path, pattern: str = 'checkpoint_clase*_*.pt') -> List[Path]`

---

### 5. 🔧 Parámetros de Modelo (PRIORIDAD BAJA)

#### 5.1 Configuración de KBlock Repetida
**Valores comunes en múltiples scripts:**
```python
gamma = 0.7
delta_t = 0.9
timesteps = 100 (o 50)
ksize = 7
ch = 4
img_size = 64
init_omg = 0.1
```

**Archivos con configuraciones similares:**
- `encontrar_alpha_critico.py`
- `encontrar_alpha_critico_clase3.py`
- `analizar_imagen_individual.py`
- `test_single_image_referencia.py`
- `analizar_con_sqlite.py`

**Recomendación:**
- Crear dataclass o NamedTuple para configuración:
  ```python
  from dataclasses import dataclass
  
  @dataclass
  class KuramotoConfig:
      gamma: float = 0.7
      delta_t: float = 0.9
      timesteps: int = 100
      ksize: int = 7
      ch: int = 4
      img_size: int = 64
      init_omg: float = 0.1
      
      @classmethod
      def from_args(cls, args):
          return cls(**{k: v for k, v in vars(args).items() if k in cls.__annotations__})
  ```

---

## 📈 Análisis Cuantitativo

### Redundancia por Tipo

| Tipo de Redundancia | Archivos Afectados | Líneas Duplicadas | Prioridad |
|---------------------|-------------------|-------------------|-----------|
| Funciones helper    | 7-8               | ~150-200          | ALTA      |
| Config matplotlib   | 8                 | ~80-100           | MEDIA     |
| Carga checkpoints   | 4                 | ~40-60            | MEDIA     |
| Parámetros modelo   | 5                 | ~30-50            | BAJA      |
| Imports/paths       | 16                | ~100-150          | BAJA      |
| **TOTAL**           | **16**            | **~400-560**      | -         |

### Scripts con Mayor Redundancia

1. **`encontrar_alpha_critico.py`** y **`encontrar_alpha_critico_clase3.py`**
   - Funcionalidad casi idéntica (~70% código compartido)
   - Diferencia: uno itera todas las clases, otro solo clase 3
   - **Recomendación:** Unificar con argumento `--target-classes`

2. **`visualizar_clase_unica.py`** y **`visualizar_puntos_criticos.py`**
   - Setup matplotlib idéntico
   - Funciones helper duplicadas
   - **Recomendación:** Extraer funciones comunes a módulo de visualización

3. **`generar_graficas_*.py`** (3 scripts)
   - Configuración matplotlib repetida
   - Patrones de guardado similares
   - **Recomendación:** Clase base `GraficadorBase` con métodos compartidos

---

## 🎯 Plan de Refactorización Recomendado

### Fase 1: Utilidades Básicas (2-3 horas)

#### Crear estructura:
```
analisis_alpha_c/
├── utils/
│   ├── __init__.py
│   ├── device_utils.py      # get_device()
│   ├── alpha_utils.py        # generate_alphas()
│   ├── image_utils.py        # prepare_image()
│   ├── checkpoint_utils.py   # load/save_checkpoint()
│   ├── plot_config.py        # setup_matplotlib(), save_figure()
│   └── config.py             # Constantes globales
```

#### Contenido de `utils/device_utils.py`:
```python
import torch

def get_device(device_arg='auto'):
    """Detecta el mejor dispositivo disponible."""
    if device_arg != 'auto':
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
```

#### Contenido de `utils/plot_config.py`:
```python
import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

def setup_matplotlib(backend='Agg', cache_dir='_mplconfig'):
    """Configura backend y caché de matplotlib."""
    if cache_dir:
        cache_path = Path.cwd() / cache_dir
        cache_path.mkdir(exist_ok=True)
        os.environ['MPLCONFIGDIR'] = str(cache_path)
    matplotlib.use(backend)

def save_figure(fig, path, format='png', dpi=300, **kwargs):
    """Guarda figura con parámetros consistentes."""
    defaults = {'bbox_inches': 'tight'}
    defaults.update(kwargs)
    fig.savefig(path, format=format, dpi=dpi, **defaults)
```

### Fase 2: Consolidación de Scripts (3-4 horas)

1. **Unificar `encontrar_alpha_critico*.py`:**
   ```bash
   # Nuevo script unificado
   python encontrar_alpha_critico.py --classes all
   python encontrar_alpha_critico.py --classes 3
   python encontrar_alpha_critico.py --classes 1,3,5,7,9
   ```

2. **Crear clase base para graficadores:**
   ```python
   # utils/base_graficador.py
   class GraficadorBase:
       def __init__(self):
           setup_matplotlib()
       
       def save(self, fig, path, **kwargs):
           save_figure(fig, path, **kwargs)
   ```

3. **Extraer funciones comunes de visualización:**
   - `plot_distribution(alphas, title, output)` - usado en 3+ scripts
   - `plot_curves_by_class(data_dict, output)` - usado en 2+ scripts

### Fase 3: Configuración Centralizada (1-2 horas)

#### `utils/config.py`:
```python
from dataclasses import dataclass
from pathlib import Path

# Paths
DATA_DIR = Path("./data")
CHECKPOINTS_DIR = Path("checkpoints_impares")
DISTRIBUTIONS_DIR = Path("distribuciones_impares")
RESULTS_DIR = Path("resultados_criticalidad.db")

# Constantes de modelo
@dataclass
class ModelConfig:
    gamma: float = 0.7
    delta_t: float = 0.9
    timesteps: int = 100
    ksize: int = 7
    ch: int = 4
    img_size: int = 64
    init_omg: float = 0.1
    
    DEFAULT = None  # Se inicializa en __post_init__

def __post_init__(self):
    ModelConfig.DEFAULT = ModelConfig()
```

---

## ✅ Beneficios Esperados

### Reducción de Código
- **Antes:** ~2,500-3,000 líneas totales
- **Después:** ~1,800-2,200 líneas
- **Reducción:** 25-30%

### Mantenibilidad
- ✅ Cambios en lógica de device detection: 1 archivo vs 7 archivos
- ✅ Actualización de matplotlib config: 1 archivo vs 8 archivos
- ✅ Cambio en parámetros por defecto: 1 archivo vs 5 archivos

### Consistencia
- ✅ Misma lógica de preparación de imágenes en todos los scripts
- ✅ Mismo manejo de checkpoints
- ✅ Mismos parámetros de guardado de figuras

### Testing
- ✅ Tests unitarios concentrados en `utils/`
- ✅ Fácil validación de funciones críticas

---

## 🚨 Riesgos y Consideraciones

### Riesgos Menores
1. **Cambios en imports:**
   - Todos los scripts necesitarán actualizar imports
   - Mitigación: Hacer cambios en rama separada, testear uno por uno

2. **Compatibilidad con scripts en ejecución:**
   - El script SQLite activo (`analizar_con_sqlite.py`) no debería modificarse mientras corre
   - Mitigación: Esperar a que termine el proceso actual

### Dependencias
- Los scripts refactorizados dependerán de `utils/`
- Si se mueven archivos, actualizar paths relativos

---

## 📝 Siguiente Paso Recomendado

**Opción 1: Refactorización Completa (recomendada para largo plazo)**
- Implementar Fase 1 completa
- Crear PR para revisión
- Migrar scripts gradualmente

**Opción 2: Quick Wins (recomendada para corto plazo)**
- Solo crear `utils/device_utils.py` y `utils/plot_config.py`
- Actualizar los 3 scripts más usados:
  - `analizar_con_sqlite.py` (activo)
  - `visualizar_alpha_critico.py`
  - `encontrar_alpha_critico.py`

**Opción 3: Mantener Status Quo**
- No refactorizar mientras el análisis esté corriendo
- Documentar redundancia para futuras decisiones
- **Esta es la opción más segura dado que hay un proceso de 43 días en ejecución**

---

## 🔗 Referencias

- Scripts analizados: 16 archivos .py en `analisis_alpha_c/`
- Módulos base: `kuramoto/`, `datasets/`, `analisis/`, `utils/` (del proyecto principal)
- Documentación: `README.md` en `analisis_alpha_c/`

**Nota:** Este análisis se realizó el 1 de noviembre de 2025, mientras el script `analizar_con_sqlite.py` está procesando 60,000 imágenes MNIST (85/60,000 completadas).
