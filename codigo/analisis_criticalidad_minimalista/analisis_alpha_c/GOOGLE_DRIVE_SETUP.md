# 💾 Integración con Google Drive para Persistencia de Datos

## 🎯 Objetivo

Ambos notebooks ahora guardan **automáticamente** todos los resultados en Google Drive, garantizando que:

1. ✅ **Los datos sobreviven** a desconexiones de Colab
2. ✅ **Puedes retomar** el procesamiento en cualquier momento
3. ✅ **Acceso desde cualquier dispositivo** (PC, móvil, tablet)
4. ✅ **No pierdes trabajo** si Colab se desconecta

---

## 📂 Estructura en Google Drive

Después de ejecutar los notebooks, tendrás estos directorios en tu Drive:

```
Google Drive/
└── MyDrive/
    ├── MNIST_C_Critical/           ← Versión autocontenida
    │   ├── mnist_c_critical_autocontenido.db
    │   ├── resumen_c_critical.json
    │   ├── distribucion_c_critical_por_clase.png
    │   └── comparacion_medias_clases.png
    │
    └── MNIST_C_Critical_AKOrN/     ← Versión con librería
        ├── mnist_c_critical_akorn.db
        ├── resumen_c_critical_akorn.json
        ├── distribucion_c_critical_por_clase_akorn.png
        └── comparacion_medias_clases_akorn.png
```

---

## 🚀 Cómo Funciona

### Primera Celda: Montar Google Drive

**Versión Autocontenida:**
```python
# Celda 1 del notebook
# Detecta automáticamente si estás en Colab o local

try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_PATH = '/content/drive/MyDrive/MNIST_C_Critical'
else:
    DRIVE_PATH = '.'  # Si ejecutas localmente
```

**Versión AKOrN:**
```python
# Celda 0 del notebook
from google.colab import drive
drive.mount('/content/drive')
DRIVE_PATH = '/content/drive/MyDrive/MNIST_C_Critical_AKOrN'
```

### Base de Datos en Drive

```python
# La ruta de la base de datos ahora apunta a Drive
DB_PATH = os.path.join(DRIVE_PATH, 'mnist_c_critical_autocontenido.db')

# Cada vez que guardas un resultado:
save_result(DB_PATH, clase, idx, result, PARAMS)
# ↓
# Se escribe DIRECTAMENTE en Google Drive
# ✅ Persistencia garantizada
```

### Gráficos y JSON en Drive

```python
# Todos los archivos se guardan en Drive
plot_path = os.path.join(DRIVE_PATH, 'distribucion_c_critical_por_clase.png')
json_path = os.path.join(DRIVE_PATH, 'resumen_c_critical.json')

plt.savefig(plot_path, dpi=150)
json.dump(resultados_resumen, open(json_path, 'w'))
```

---

## 🔄 Escenarios de Uso

### Escenario 1: Primera Ejecución

```python
# Ejecutas en Colab
1. Celda 1: Montar Drive → Autorizar acceso
2. Celda 2-9: Setup y configuración
3. Celda 10: Procesamiento
   - Clase 0: 2000/5923 imágenes procesadas...
   - ⚠️ COLAB SE DESCONECTA (timeout, internet, etc.)
```

**Resultado:**
- ✅ 2000 imágenes guardadas en Drive
- ✅ Base de datos intacta: `clase_0` tiene 2000 registros
- ❌ Perdiste solo la imagen #2001 (la que estaba procesando)

### Escenario 2: Reiniciar Después de Desconexión

```python
# Vuelves a abrir el notebook (minutos, horas o días después)
1. Celda 1: Montar Drive → Mismo directorio
2. Celda 2-9: Setup (rápido)
3. Celda 10: Procesamiento
   ✅ Imágenes ya procesadas: 2000
   ⏳ Imágenes pendientes: 3923
   
   # Continúa desde imagen #2001 automáticamente
```

**Ventajas:**
- 🚀 No reprocesa las 2000 anteriores
- 💾 Lee desde la BD en Drive
- ⚡ Continúa inmediatamente

### Escenario 3: Cambiar de Dispositivo

```python
# Día 1: Procesaste en laptop (clase 0-4 completas)
# Día 2: Abres Colab desde tu teléfono
1. Monta el mismo Drive
2. Ejecuta el notebook
   ✅ Clases 0-4: 29,615 imágenes (saltadas)
   ⏳ Clases 5-9: 30,385 imágenes (continúa)
```

### Escenario 4: Múltiples Sesiones Paralelas

⚠️ **NO RECOMENDADO** pero funciona:
```python
# Notebook A en Colab: Procesa clases 0-4
CLASES_A_PROCESAR = range(0, 5)

# Notebook B en otro Colab: Procesa clases 5-9
CLASES_A_PROCESAR = range(5, 10)

# Ambos escriben en el MISMO Drive
# ✅ SQLite maneja concurrencia por tabla
# ✅ No hay conflictos (diferentes clases)
```

---

## 📊 Monitoreo del Progreso

### Desde Google Drive Web

1. Ve a: https://drive.google.com/drive/my-drive
2. Navega a `MNIST_C_Critical/`
3. Clic derecho en `mnist_c_critical_autocontenido.db` → Ver detalles
4. **Última modificación**: Te dice cuándo se guardó el último dato

### Desde el Notebook (nueva celda)

```python
# Agregar esta celda para ver progreso en tiempo real
import sqlite3

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("📊 Progreso actual:")
total_procesadas = 0
for clase in range(10):
    try:
        cursor.execute(f'SELECT COUNT(*) FROM clase_{clase}')
        count = cursor.fetchone()[0]
        print(f"  Clase {clase}: {count}/~6000 imágenes")
        total_procesadas += count
    except:
        print(f"  Clase {clase}: 0 imágenes")

print(f"\n✅ Total: {total_procesadas}/60,000 imágenes procesadas")
print(f"   Progreso: {total_procesadas/60000*100:.1f}%")

conn.close()
```

### Desde Otro Notebook (mientras procesa)

Puedes abrir un **segundo notebook** solo para monitorear:

```python
# Notebook de monitoreo
from google.colab import drive
drive.mount('/content/drive')

import sqlite3
import time

DB_PATH = '/content/drive/MyDrive/MNIST_C_Critical/mnist_c_critical_autocontenido.db'

while True:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for clase in range(10):
        try:
            cursor.execute(f'SELECT COUNT(*) FROM clase_{clase}')
            count = cursor.fetchone()[0]
            print(f"Clase {clase}: {count}", end="  ")
        except:
            print(f"Clase {clase}: 0", end="  ")
    print()
    
    conn.close()
    time.sleep(60)  # Actualiza cada minuto
```

---

## 🛡️ Ventajas de Seguridad

### 1. Backup Automático
- ✅ Google Drive tiene versionado automático
- ✅ Puedes recuperar versiones anteriores si algo falla
- ✅ Clic derecho → "Ver versiones" en cualquier archivo

### 2. Sin Límite de Tiempo
- ✅ Colab free: 12 horas máximo → Reinicia y continúa
- ✅ Colab Pro: 24 horas → Más tiempo pero igual robusto
- ✅ Los datos siempre están en Drive

### 3. Acceso Universal
```python
# Desde cualquier lugar:
- 💻 Laptop con Colab
- 📱 Teléfono/tablet (solo monitoreo)
- 🖥️ Otro PC
- 🌐 Cualquier navegador

# Mismo Drive, mismo progreso
```

---

## ⚠️ Consideraciones Importantes

### Cuota de Drive

```
- Drive gratis: 15 GB
- Base de datos MNIST completa: ~500 MB - 1 GB
- Gráficos: ~5 MB
- JSON: <1 MB

Total estimado: ~1 GB (bien dentro del límite)
```

Si tienes poco espacio:
```python
# Opción 1: Procesa por clases
CLASES_A_PROCESAR = [0, 1, 2]  # Solo 3 clases
# Después descarga y borra, luego procesa [3, 4, 5]

# Opción 2: Reduce resolución
PARAMS = {
    'h': 32,  # En vez de 64
    'w': 32,
    # ... otros params
}
```

### Velocidad de Escritura

Drive es un poco más lento que disco local:

```python
# Tiempo por imagen:
- Disco local:     0.2-0.3 segundos
- Google Drive:    0.3-0.5 segundos

# Para 60,000 imágenes:
- Local:  ~3-4 horas
- Drive:  ~4-6 horas

# Pero... con Drive no pierdes nada si se desconecta
```

### Primera Autorización

La primera vez que ejecutes, verás:

```
Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?...

Enter your authorization code:
```

1. Copia la URL
2. Pégala en tu navegador
3. Inicia sesión con tu cuenta Google
4. Copia el código que te da
5. Pégalo en Colab

Después de esto, Drive queda montado durante toda la sesión.

---

## 🧪 Prueba Rápida

Para verificar que funciona:

```python
# Celda de prueba (ejecutar después de montar Drive)
import os

# Verificar que Drive está montado
assert os.path.exists('/content/drive/MyDrive'), "❌ Drive no montado"

# Crear archivo de prueba
test_path = os.path.join(DRIVE_PATH, 'test.txt')
with open(test_path, 'w') as f:
    f.write("Prueba exitosa!")

# Verificar que se creó
assert os.path.exists(test_path), "❌ No se pudo escribir en Drive"

print("✅ Google Drive funcionando correctamente")
print(f"✅ Archivo de prueba creado: {test_path}")
print("✅ Puedes verificarlo en: https://drive.google.com/drive/my-drive")

# Limpieza
os.remove(test_path)
```

---

## 📋 Checklist Pre-Ejecución

Antes de ejecutar el procesamiento completo:

- [ ] ✅ Google Drive montado correctamente
- [ ] ✅ Directorio `MNIST_C_Critical` creado
- [ ] ✅ Variable `DRIVE_PATH` apunta a Drive
- [ ] ✅ `DB_PATH` incluye `DRIVE_PATH`
- [ ] ✅ Tienes espacio suficiente (~2 GB libre)
- [ ] ✅ Ejecutaste la celda de prueba exitosamente

---

## 🎉 Resumen

**Antes (sin Drive):**
```
Colab local → Procesa → BD en /content → Se desconecta → ❌ TODO PERDIDO
```

**Ahora (con Drive):**
```
Colab → Monta Drive → Procesa → BD en Drive → Se desconecta → ✅ TODO GUARDADO
                                     ↓
                              Reinicia → Continúa desde donde quedó
```

**Beneficios clave:**
1. 🔄 **Reinicio automático** desde última imagen
2. 💾 **Persistencia garantizada** en Drive
3. 🌐 **Acceso universal** desde cualquier dispositivo
4. 🛡️ **Backup automático** con versionado
5. 📊 **Monitoreo en tiempo real** desde otro notebook

¡Ahora puedes procesar el dataset completo sin miedo a perder progreso! 🚀
