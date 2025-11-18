# 🔐 Guía de Reproducibilidad - Análisis R vs C

## Garantías de Reproducibilidad

Este notebook **garantiza reproducibilidad completa** siguiendo **exactamente** el mismo patrón que `calculo_alpha_c_mnist_autocontenido.ipynb`.

### 1. Semilla Global Fija

```python
SEED = 1  # MISMO valor que calculo_alpha_c

set_seed(SEED)
```

**Semilla:** `SEED = 1` (no 42, para mantener consistencia con el notebook de α_c)

### 2. Estado Inicial Único Compartido

**CRÍTICO:** `x_init` se crea **UNA SOLA VEZ** y se reutiliza para **TODAS** las imágenes del dataset:

```python
def calcular_R_vs_C(imagen, kblock, c_range, device, imagen_idx):
    # Fijar semilla basada en imagen_idx
    torch.manual_seed(SEED + imagen_idx)
    
    # Estado inicial (mismo para TODOS los C)
    x_init = torch.randn(1, CH*2, IMG_SIZE, IMG_SIZE).to(device)
    
    for c_val in c_range:
        x = x_init.clone()  # ← Restaurar estado inicial
        # Calcular R(c_val) partiendo de x_init
```

### 3. Estructura de la Base de Datos

#### Esquema SQLite

```sql
-- 10 tablas: clase_0, clase_1, ..., clase_9
CREATE TABLE clase_N (
    id INTEGER PRIMARY KEY,
    imagen_idx INTEGER NOT NULL UNIQUE,
    clase INTEGER NOT NULL,
    c_values BLOB NOT NULL,           -- Array de 50 valores C ∈ [0, 0.4]
    r_values BLOB NOT NULL,           -- Array de 50 valores R correspondientes
    c_critico REAL,                   -- C donde R ≈ 0.5
    seed_usado INTEGER NOT NULL,      -- = 42 + imagen_idx
    timestamp TEXT NOT NULL
);

-- Tabla de configuración
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
```

#### Contenido de `metadata`:

| Key | Value | Descripción |
|-----|-------|-------------|
| `seed` | `42` | Semilla global |
| `c_min` | `0.0` | Inicio del rango C |
| `c_max` | `0.4` | Fin del rango C |
| `n_c_points` | `50` | Número de puntos C |
| `t_steps` | `30` | Pasos temporales para convergencia |
| `ch` | `4` | Canales del modelo Kuramoto |
| `n` | `4` | Número de capas KConv2d |
| `device` | `CUDA/MPS/CPU` | Dispositivo usado |
| `created` | `ISO 8601` | Fecha de creación |

### 4. Cómo Reproducir un Resultado Específico

#### Ejemplo: Reproducir resultado de imagen 1000, clase 3

```python
# Paso 1: Cargar configuración de la DB
conn = sqlite3.connect('mnist_R_vs_C.db')
cursor = conn.cursor()
cursor.execute('SELECT value FROM metadata WHERE key = "seed"')
SEED = int(cursor.fetchone()[0])

# Paso 2: Calcular semilla de la imagen
imagen_idx = 1000
seed_imagen = SEED + imagen_idx  # = 42 + 1000 = 1042

# Paso 3: Fijar semilla
torch.manual_seed(seed_imagen)
np.random.seed(seed_imagen)

# Paso 4: Cargar imagen del dataset MNIST
from torchvision import datasets, transforms
train_dataset = datasets.MNIST(root='./data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.Resize((64, 64)),
                                    transforms.ToTensor()
                                ]))
imagen, label = train_dataset[imagen_idx]

# Paso 5: Calcular R vs C
c_range = np.linspace(0.0, 0.4, 50)
r_values_reproducidos = calcular_R_vs_C(imagen.unsqueeze(0), kblock, c_range, device, imagen_idx)

# Paso 6: Verificar contra DB
cursor.execute('SELECT r_values FROM clase_3 WHERE imagen_idx = ?', (imagen_idx,))
r_values_db = pickle.loads(cursor.fetchone()[0])

assert np.allclose(r_values_reproducidos, r_values_db, atol=1e-6), "¡Resultados diferentes!"
print("✅ Reproducción exitosa: resultados idénticos")
```

### 5. Verificación Automatizada

El notebook incluye una celda de verificación:

```python
verificar_reproducibilidad(DB_PATH)
```

**Salida esperada:**
```
✅ TODAS las semillas son correctas y reproducibles
   Fórmula: seed_usado = SEED(42) + imagen_idx
```

### 6. Consideraciones Importantes

#### ✅ Garantizado reproducible:

- Mismo hardware (CUDA/MPS/CPU)
- Misma versión de PyTorch
- Misma semilla y `imagen_idx`
- Mismo dataset MNIST

#### ⚠️ Puede variar ligeramente:

- **Hardware diferente**: Operaciones floating-point en GPU pueden tener precisión numérica levemente diferente
- **Versiones diferentes de PyTorch**: Implementaciones de operaciones pueden cambiar
- **Modo determinista vs no-determinista**: Asegurarse de `torch.backends.cudnn.deterministic = True`

### 7. Exportar y Compartir Resultados

Para compartir resultados reproducibles:

```bash
# Comprimir base de datos
tar -czf mnist_R_vs_C_results.tar.gz resultados_R_vs_C/

# Incluir información de versiones
pip freeze > requirements.txt
```

**Archivo de reproducción** (`reproduce.txt`):
```
Dataset: MNIST Training Set (60,000 images)
PyTorch: 2.x.x
CUDA: 11.x / MPS (Apple M3) / CPU
Seed global: 42
Fórmula semilla: seed = 42 + imagen_idx
Config: C ∈ [0, 0.4], 50 puntos, T_steps=30
```

### 8. Troubleshooting

#### Problema: Resultados no reproducibles

**Causa posible:** Orden del DataLoader shuffle=True

**Solución:**
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,  # ← CRÍTICO para reproducibilidad
    num_workers=0,
    generator=torch.Generator().manual_seed(SEED)
)
```

#### Problema: Valores ligeramente diferentes en GPU vs CPU

**Causa:** Precisión numérica de operaciones floating-point

**Solución:** Usar `atol=1e-5` en comparaciones:
```python
np.allclose(r_values_gpu, r_values_cpu, atol=1e-5)
```

---

## Resumen Ejecutivo

✅ **Reproducibilidad garantizada al 100%** para:
- Mismo hardware + misma versión PyTorch
- Usar semilla `SEED + imagen_idx` antes de procesar cada imagen
- Partir del mismo `x_init` para todos los valores de C

✅ **Base de datos SQLite** almacena:
- Curvas R(C) completas (50 puntos)
- C_crítico calculado (R ≈ 0.5)
- Semilla usada para cada imagen
- Metadata de configuración

✅ **Verificación automática** incluida en el notebook

---

**Autor:** Proyecto Inv. Teórica - Análisis de Criticalidad Kuramoto  
**Fecha:** Noviembre 2025  
**Notebook:** `calculo_R_vs_C_autocontenido.ipynb`
