# 🚀 Inicio Rápido: vscode-colab para AKOrN

## 🎯 Qué es vscode-colab

**vscode-colab** te permite ejecutar notebooks de Jupyter en los **servidores de Google Colab** (con GPU gratuita) pero trabajando directamente desde **VSCode en tu Mac**, sin tener que:
- ❌ Subir archivos manualmente a Colab
- ❌ Usar la interfaz web de Colab
- ❌ Copiar/pegar código entre VSCode y navegador

---

## ⚡ Setup Inicial (Solo primera vez - 2 minutos)

### Paso 1: Instalar Extensión
```bash
code --install-extension ms-toolsai.vscode-jupyter-colab
```

O desde VSCode:
1. `Cmd+Shift+X` (abrir extensiones)
2. Buscar: **"Colab"**
3. Instalar: **"Colab" by Google**

### Paso 2: Verificar Instalación
1. Abrir cualquier `.ipynb` en VSCode
2. Click en "Select Kernel" (arriba derecha)
3. Debería aparecer **"Connect to Colab"** en la lista

✅ Si aparece → Instalación exitosa  
❌ Si no aparece → Reinicia VSCode y vuelve a intentar

---

## 🚀 Uso Diario (30 segundos)

### Entrenar AKOrN con GPU de Colab

1. **Abrir notebook**:
   ```
   codigo/akorn/train_mnist_colab.ipynb
   ```

2. **Conectar a Colab**:
   - Click en "Select Kernel" (esquina superior derecha)
   - Seleccionar: **"Connect to Colab"**
   - Primera vez: Autenticar con Google (aceptar permisos)
   - Esperar ~30 segundos a que conecte

3. **Verificar conexión**:
   - Verás: **"🟢 Colab (Python 3.10)"** arriba
   - Significa: Conectado a servidor Colab con GPU

4. **Ejecutar celdas**:
   - Click ▶️ en cada celda (o `Shift+Enter`)
   - Outputs aparecen directamente en VSCode
   - Código corre en GPU de Colab (no en tu Mac)

5. **Iniciar entrenamiento**:
   - Ejecutar celdas 1-6 secuencialmente
   - Celda 6 inicia entrenamiento de 18 horas
   - ⚠️ **Mantén VSCode abierto** (no cierres la ventana)

---

## 📊 Flujo de Trabajo Visual

```
┌──────────────────────────────────────────────────────────────┐
│  TU MAC (VSCode)                                             │
│                                                              │
│  1. Abres: train_mnist_colab.ipynb                          │
│  2. Select Kernel → "Connect to Colab"                      │
│  3. Click ▶️ en celda de código                             │
│                                                              │
│         ↓ (código se envía via internet)                    │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│  SERVIDORES GOOGLE COLAB (GPU T4 gratuita)                  │
│                                                              │
│  1. Recibe tu código                                         │
│  2. Ejecuta en GPU T4                                        │
│  3. Entrena AKOrN (forward, backward, update)               │
│                                                              │
│         ↓ (outputs regresan via internet)                   │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│  TU MAC (VSCode)                                             │
│                                                              │
│  4. Ves outputs/resultados directamente en VSCode           │
│  5. TensorBoard se abre en panel de VSCode                  │
│  6. Checkpoints se guardan en Google Drive                  │
└──────────────────────────────────────────────────────────────┘
```

**Ventaja clave**: Trabajas en tu editor familiar (VSCode) pero con GPU potente (Colab) 🎉

---

## 🔑 Comandos Esenciales

### Conectar/Desconectar
```
Cmd+Shift+P → "Colab: Connect"     (conectar)
Cmd+Shift+P → "Colab: Disconnect"  (desconectar)
Cmd+Shift+P → "Colab: Sign Out"    (cerrar sesión)
```

### Estado de Conexión
| Indicador | Significado |
|-----------|-------------|
| 🟢 "Colab (Python 3.10)" | ✅ Conectado y listo |
| 🟡 "Connecting..." | ⏳ Conectando... |
| 🔴 "Disconnected" | ❌ Desconectado |
| ⚙️ "Busy" | 🔄 Ejecutando celda |

---

## ⚠️ Limitaciones Importantes

### 1. Archivos NO se sincronizan automáticamente
**Problema**: vscode-colab solo sube el notebook, no tu proyecto completo

**Solución**: Clonar repo en Colab (incluido en celda 2 del notebook)
```python
!git clone https://github.com/TuUsuario/TuRepo.git
%cd TuRepo/codigo/akorn
```

### 2. Mantener conexión activa
**Problema**: Si cierras VSCode, se pierde conexión

**Solución**: 
- Mantén VSCode abierto durante entrenamiento
- Guarda checkpoints frecuentes (cada 25 epochs)
- Usa Google Drive para guardar checkpoints

### 3. Tiempo límite (Colab Free)
**Problema**: Colab Free desconecta después de ~12 horas inactivas

**Solución**:
- Ejecutar script anti-idle (incluido en notebook)
- O usar **Colab Pro** ($10/mes, sin límite)

---

## 🛠️ Troubleshooting Rápido

### No aparece "Connect to Colab"
```bash
# Reinstalar extensión
code --uninstall-extension ms-toolsai.vscode-jupyter-colab
code --install-extension ms-toolsai.vscode-jupyter-colab

# Reiniciar VSCode
```

### "Authentication failed"
```
Cmd+Shift+P → "Colab: Sign Out"
# Volver a conectar y re-autenticar
```

### Conexión muy lenta
- **Normal**: 30 seg - 1 min primera vez
- **Lento**: Verifica tu internet
- **Tip**: Cierra carpetas grandes en VSCode antes de conectar

### GPU no detectada
```python
# Ejecutar en celda 1 del notebook:
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")
```

Si dice "No GPU":
1. Abre [colab.research.google.com](https://colab.research.google.com) en navegador
2. Runtime → Change runtime type → GPU (T4)
3. Regresa a VSCode y reconecta

---

## 📈 Ejemplo Completo: Entrenar AKOrN

```bash
# 1. Abrir notebook en VSCode
open codigo/akorn/train_mnist_colab.ipynb

# 2. En VSCode:
#    - Click "Select Kernel" → "Connect to Colab"
#    - Autenticar (primera vez)

# 3. Ejecutar celdas secuencialmente (click ▶️):
#    Celda 1: Verificar GPU ✓
#    Celda 2: Clonar repo
#    Celda 3: Instalar dependencias
#    Celda 4: Descargar MNIST
#    Celda 5: Montar Drive (opcional)
#    Celda 6: ¡Iniciar entrenamiento! 🚀

# 4. Monitorear:
#    - Outputs en VSCode
#    - TensorBoard en panel lateral
#    - Checkpoints en Drive cada 25 epochs

# 5. Después de 18 horas:
#    - Celda 11: Evaluar modelo final
#    - Celda 13: Descargar resultados
```

---

## 🎓 Comparación: Métodos de Trabajo

| Característica | Colab Web (navegador) | vscode-colab |
|----------------|----------------------|--------------|
| **Editor** | Interfaz Colab | VSCode ✓ |
| **GPU** | Gratis (T4) | Gratis (T4) ✓ |
| **Atajos teclado** | Limitados | VSCode completo ✓ |
| **Archivos locales** | Subir manual | Clonar repo |
| **Multi-archivo** | Difícil | Fácil ✓ |
| **Debugging** | Básico | VSCode avanzado ✓ |
| **Sincronización** | Manual | Automática (notebook) ✓ |

**Recomendación**: Usa **vscode-colab** si:
- ✅ Prefieres trabajar en VSCode
- ✅ Necesitas editar múltiples archivos
- ✅ Quieres atajos de teclado familiares
- ✅ Tienes buena conexión a internet

---

## 🚦 Estado Actual

### ✅ Ya está listo para usar
- `codigo/akorn/train_mnist_colab.ipynb` ← Notebook configurado
- Celdas incluyen clonar repo automáticamente
- Script anti-idle para mantener sesión activa
- Guardado automático en Drive cada 25 epochs

### 📝 Checklist antes de iniciar entrenamiento

- [ ] Extensión vscode-colab instalada
- [ ] Notebook abierto en VSCode
- [ ] Kernel "Connect to Colab" seleccionado
- [ ] Autenticado con Google (primera vez)
- [ ] Celda 1 muestra "CUDA disponible: True"
- [ ] Google Drive montado (celda 5)
- [ ] Mac conectado a corriente (no batería)
- [ ] VSCode se mantendrá abierto 18 horas

**Tiempo de ejecución total**: ~20 horas (setup + entrenamiento)

---

## 📚 Documentación Adicional

- **Guía completa**: `GUIA_COLAB_GPU.md` (troubleshooting detallado)
- **Notebook**: `train_mnist_colab.ipynb` (instrucciones en celdas)
- **Parámetros**: `AKORN_RESUMEN_EJECUTIVO.md` (explicación parámetros)

---

**Última actualización**: Noviembre 2025  
**Estado**: ✅ Listo para ejecutar  
**Próximo paso**: Abre `train_mnist_colab.ipynb` y conecta a Colab 🚀
