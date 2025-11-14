# 🔄 Sistema de Reinicio Automático

## ✅ Funcionalidad Implementada

Ambos notebooks ahora incluyen **reinicio automático** que permite:

1. **Detectar imágenes ya procesadas** en la base de datos SQLite
2. **Saltar automáticamente** imágenes duplicadas
3. **Continuar desde donde se quedó** si el proceso se interrumpe
4. **Preservar todos los resultados** anteriores

---

## 🔧 Cómo Funciona

### Nueva Función: `get_processed_images()`

```python
def get_processed_images(db_path, clase):
    """
    Obtiene los índices de imágenes ya procesadas para una clase.
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
```

### Lógica de Procesamiento Actualizada

```python
for clase in CLASES_A_PROCESAR:
    # ✅ Verificar qué imágenes ya están procesadas
    processed_images = get_processed_images(DB_PATH, clase)
    indices_pendientes = [idx for idx in indices if idx not in processed_images]
    
    print(f"✅ Imágenes ya procesadas: {len(processed_images)}")
    print(f"⏳ Imágenes pendientes: {len(indices_pendientes)}")
    
    # ✅ Saltar si la clase ya está completa
    if len(indices_pendientes) == 0:
        print(f"✨ Clase {clase} ya está completamente procesada. Saltando...")
        # Cargar estadísticas desde BD
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f'SELECT c_critical FROM clase_{clase}')
        c_criticals = np.array([row[0] for row in cursor.fetchall()])
        conn.close()
    else:
        # ✅ Procesar solo las imágenes pendientes
        for idx in tqdm(indices_pendientes, desc=f"Clase {clase}"):
            # ... calcular y guardar ...
        
        # ✅ Cargar TODOS los resultados al final
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f'SELECT c_critical FROM clase_{clase}')
        c_criticals = np.array([row[0] for row in cursor.fetchall()])
        conn.close()
```

---

## 📊 Ejemplo de Uso

### Escenario 1: Primera Ejecución

```
PROCESANDO CLASE 0
============================================================
✅ Imágenes ya procesadas: 0
⏳ Imágenes pendientes: 5923

Clase 0: 100%|██████████| 5923/5923 [20:15<00:00, 4.87it/s]

📊 Estadísticas Clase 0:
  Imágenes procesadas: 5923
  C_crítico medio: 0.1523 ± 0.0345
```

### Escenario 2: Se Interrumpe en la Imagen 3000

```
⚠️ CONEXIÓN PERDIDA / KERNEL CRASHED / TIMEOUT
```

### Escenario 3: Reiniciar Notebook

```
PROCESANDO CLASE 0
============================================================
✅ Imágenes ya procesadas: 3000
⏳ Imágenes pendientes: 2923

Clase 0: 100%|██████████| 2923/2923 [10:08<00:00, 4.81it/s]

📊 Estadísticas Clase 0:
  Imágenes procesadas: 5923
  C_crítico medio: 0.1523 ± 0.0345
```

### Escenario 4: Clase Ya Completada

```
PROCESANDO CLASE 0
============================================================
✅ Imágenes ya procesadas: 5923
⏳ Imágenes pendientes: 0
✨ Clase 0 ya está completamente procesada. Saltando...

📊 Estadísticas Clase 0:
  Imágenes procesadas: 5923
  C_crítico medio: 0.1523 ± 0.0345
```

---

## 🔥 Ventajas

### ✅ Tolerante a Fallos

- **Desconexión de Colab**: Solo pierdes el progreso de la imagen actual
- **Kernel crash**: Todos los datos guardados en SQLite están a salvo
- **Timeout**: Simplemente reinicia y continúa

### ✅ Flexible

- **Cambiar límites**: Puedes modificar `LIMITE_POR_CLASE` entre ejecuciones
- **Pausas intencionales**: Detén cuando quieras y retoma después
- **Múltiples sesiones**: Ejecuta en diferentes momentos sin perder progreso

### ✅ Eficiente

- **No duplica cálculos**: Detecta automáticamente qué está hecho
- **Sin sobrescritura**: `INSERT OR REPLACE` protege contra duplicados
- **Progreso granular**: Guarda después de cada imagen, no al final

---

## 🧪 Cómo Probar

### Test 1: Interrumpir Intencionalmente

```python
# Modificar temporalmente la celda de procesamiento:

for idx in tqdm(indices_pendientes, desc=f"Clase {clase}"):
    if idx > imagenes_por_clase[clase][50]:  # Detener después de 50 imágenes
        print("🛑 DETENIENDO INTENCIONALMENTE PARA TEST")
        break
    
    img, label = train_dataset[idx]
    result = calculate_c_critical(...)
    save_result(DB_PATH, clase, idx, result, PARAMS)
```

**Resultado esperado:**
1. Primera ejecución: Procesa 50 imágenes
2. Segunda ejecución: Detecta 50 ya procesadas, continúa desde imagen 51

### Test 2: Verificar Base de Datos

```python
import sqlite3

conn = sqlite3.connect('/content/mnist_c_critical_autocontenido.db')
cursor = conn.cursor()

# Ver cuántas imágenes procesadas por clase
for clase in range(10):
    cursor.execute(f'SELECT COUNT(*) FROM clase_{clase}')
    count = cursor.fetchone()[0]
    print(f"Clase {clase}: {count} imágenes procesadas")

conn.close()
```

### Test 3: Simular Timeout de Colab

```python
# En Colab, ejecuta esto en otra celda mientras procesa:
import time
time.sleep(43200)  # 12 horas - forzará timeout de Colab
```

**Resultado esperado:**
- Colab desconecta después de 12 horas
- Al reconectar y ejecutar, continúa automáticamente

---

## 🚨 Importante

### ⚠️ Misma Base de Datos

Asegúrate de usar el **mismo archivo de base de datos** al reiniciar:

```python
# ✅ CORRECTO (mismo path)
DB_PATH = '/content/mnist_c_critical_autocontenido.db'

# ❌ INCORRECTO (path diferente)
DB_PATH = '/content/mnist_c_critical_autocontenido_v2.db'  # Empezará desde cero
```

### ⚠️ Mismos Parámetros

Si cambias los parámetros del modelo, deberías usar una **nueva base de datos**:

```python
# Si cambias esto:
PARAMS = {
    'ch': 5,  # ⚠️ Antes era 3
    'T': 50,  # ⚠️ Antes era 30
    ...
}

# Usa nueva BD:
DB_PATH = '/content/mnist_c_critical_ch5_T50.db'
```

### ⚠️ En Colab: Persistencia

Para **garantizar que la BD sobreviva** desconexiones en Colab:

```python
# Opción 1: Google Drive
from google.colab import drive
drive.mount('/content/drive')
DB_PATH = '/content/drive/MyDrive/mnist_c_critical.db'

# Opción 2: Descargar periódicamente
from google.colab import files
# Después de cada clase:
files.download(DB_PATH)
```

---

## 📝 Resumen

### Cambios en los Notebooks

1. ✅ **Nueva función**: `get_processed_images(db_path, clase)`
2. ✅ **Verificación automática**: Antes de procesar cada clase
3. ✅ **Skip inteligente**: Salta clases ya completadas
4. ✅ **Estadísticas completas**: Siempre carga todos los datos desde BD

### Beneficios

- 🔄 **Reinicio automático**: Sin intervención manual
- 💾 **Sin pérdida de datos**: SQLite guarda después de cada imagen
- ⚡ **Eficiente**: Solo procesa lo que falta
- 🎯 **Confiable**: Funciona en interrupciones inesperadas

### Flujo Actualizado

```
Iniciar notebook
    ↓
Cargar MNIST
    ↓
Para cada clase:
    ↓
    Verificar imágenes en BD ← 🆕 NUEVO
    ↓
    ¿Ya está completa?
    ├─ Sí → Cargar estadísticas y saltar
    └─ No → Procesar solo pendientes
        ↓
        Guardar después de cada imagen ← 🆕 CRÍTICO
        ↓
        Cargar todos los resultados al final
    ↓
Generar visualizaciones
    ↓
Descargar BD completa
```

---

## 🎯 Próximos Pasos Recomendados

1. **Prueba rápida**: Ejecuta con `LIMITE_POR_CLASE = 10`
2. **Interrumpe intencionalmente**: Para después de 5 imágenes
3. **Reinicia**: Verifica que continúa desde imagen 6
4. **Si funciona**: Cambia a `LIMITE_POR_CLASE = None` para dataset completo

¡Ahora los notebooks son **completamente robustos** ante interrupciones! 🎉
