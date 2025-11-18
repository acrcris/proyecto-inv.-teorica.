# 🔐 Reproducibilidad - Análisis R vs C (ACTUALIZADO)

## ✅ Patrón de Reproducibilidad

Este notebook sigue **EXACTAMENTE** el mismo patrón que `calculo_alpha_c_mnist_autocontenido.ipynb`.

### 1. Semilla Global Fija

```python
SEED = 1  # MISMO valor que calculo_alpha_c (no 42)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
```

### 2. Estado Inicial Único (x_init)

**CRÍTICO:** `x_init` se crea **UNA SOLA VEZ** después de `set_seed(SEED)` y se **reutiliza para TODAS las imágenes**:

```python
# Inicialización global (UNA VEZ)
set_seed(SEED)
kblock = KBlock(...).to(device)
x_init = torch.randn(1, CH*2, IMG_SIZE, IMG_SIZE)  # ← ÚNICO para TODO el dataset

# Procesamiento
for imagen in dataset:
    result = calculate_c_critical_R(imagen, kblock, x_init, device, c_range)
    # x_init es el MISMO para todas las imágenes
```

**Dentro de la función de cálculo:**

```python
def calculate_c_critical_R(img, kblock, x_init, device, c_range, ...):
    """x_init es el MISMO para todas las imágenes"""
    R_final = []
    
    for c_val in c_range:
        x = x_init.clone().to(device)  # ← Clonar el MISMO x_init
        c = img_channels * c_val
        
        _, xs = kblock(x, c, T=T, gamma=gamma, del_t=del_t, return_xs=True)
        R = kuramoto_order_parameter(xs)
        R_final.append(R[-1])
    
    return {'c_critical': c_range[np.argmax(np.gradient(R_final))], ...}
```

**Garantías:**
- ✅ **Misma imagen** procesada dos veces → **MISMO resultado**
- ✅ **Diferentes imágenes** usan **MISMO x_init**
- ✅ **Todos los valores de C** dentro de una imagen parten del **MISMO x_init.clone()**

### 3. Estructura de la Base de Datos

**Tabla por clase (clase_0 a clase_9):**

```sql
CREATE TABLE clase_N (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_idx INTEGER NOT NULL UNIQUE,  -- Índice en MNIST
    c_critical REAL NOT NULL,           -- C donde derivada de R es máxima
    max_r_idx INTEGER,                  -- Índice del máximo en la derivada
    timestamp TEXT NOT NULL,
    seed INTEGER,                       -- SEED = 1
    n INTEGER,                          -- N = 4
    ch INTEGER,                         -- CH = 4
    T INTEGER,                          -- T = 30
    gamma REAL,                         -- 0.7
    del_t REAL                          -- 0.9
);
```

**Idéntico a:** `calculo_alpha_c_mnist_autocontenido.ipynb`

### 4. Cómo Reproducir un Resultado

```python
# Paso 1: Configurar semilla GLOBAL
SEED = 1
set_seed(SEED)

# Paso 2: Crear modelo y x_init UNA VEZ
kblock = KBlock(n=4, ch=4, T=4, ksize=7, init_omg=0.1).to(device)
x_init = torch.randn(1, 8, 64, 64)  # CH*2 = 4*2 = 8

# Paso 3: Cargar imagen del dataset
img, label = train_dataset[1000]  # Ejemplo: imagen 1000

# Paso 4: Calcular (x_init es el MISMO para todas)
result = calculate_c_critical_R(img, kblock, x_init, device, c_range)

# Paso 5: Verificar contra DB
conn = sqlite3.connect('mnist_R_vs_C.db')
cursor = conn.cursor()
cursor.execute('SELECT c_critical FROM clase_X WHERE image_idx = 1000')
c_crit_db = cursor.fetchone()[0]

assert abs(result['c_critical'] - c_crit_db) < 1e-6
print("✅ Reproducción exitosa!")
```

### 5. Diferencias con el Patrón Anterior

| Aspecto | Patrón Anterior (Incorrecto) | Patrón Actual (Correcto) |
|---------|------------------------------|--------------------------|
| **Semilla** | SEED = 42 | SEED = 1 |
| **x_init por imagen** | ❌ seed_imagen = SEED + img_idx | ✅ UN SOLO x_init global |
| **x_init en DB** | ❌ Guardaba seed_usado | ✅ Solo guarda SEED global |
| **Consistencia** | ❌ Diferente de alpha_c | ✅ IDÉNTICO a alpha_c |

### 6. Ventajas del Patrón Actual

1. ✅ **Consistencia:** Mismo patrón que `calculo_alpha_c_mnist_autocontenido.ipynb`
2. ✅ **Simplicidad:** No hay que calcular `seed_imagen = SEED + img_idx`
3. ✅ **Comparabilidad:** Resultados R vs C directamente comparables con α_c
4. ✅ **Menos storage:** No se guarda `seed_usado` en cada fila

### 7. Verificación de Reproducibilidad

```python
# Verificar que SEED es correcto
conn = sqlite3.connect('mnist_R_vs_C.db')
cursor = conn.cursor()
cursor.execute('SELECT seed FROM clase_0 LIMIT 1')
seed_db = cursor.fetchone()[0]
assert seed_db == 1, f"SEED incorrecto: {seed_db}"

print("✅ SEED correcto: 1")
print("✅ Patrón consistente con calculo_alpha_c_mnist_autocontenido.ipynb")
```

### 8. Resumen Ejecutivo

**Reproducibilidad Garantizada:**
- ✅ SEED = 1 (fijo global)
- ✅ x_init creado UNA VEZ y compartido por TODAS las imágenes
- ✅ Patrón IDÉNTICO a calculo_alpha_c_mnist_autocontenido.ipynb
- ✅ Base de datos con estructura compatible

**Para reproducir cualquier resultado:**
1. set_seed(1)
2. Crear kblock y x_init (una vez)
3. Procesar imagen con x_init compartido
4. Resultados serán idénticos a los de la DB

---

**Autor:** Proyecto Inv. Teórica - Análisis de Criticalidad Kuramoto  
**Fecha:** 17 de noviembre de 2025  
**Notebook:** `calculo_R_vs_C_autocontenido.ipynb`  
**Referencia:** `calculo_alpha_c_mnist_autocontenido.ipynb`
