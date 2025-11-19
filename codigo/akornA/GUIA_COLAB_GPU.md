# Guía Rápida: Entrenar AKOrN en Google Colab con GPU (desde VSCode)

## 🎯 Objetivo
Entrenar AKOrN con configuración **Máxima Precisión** (~99% accuracy) usando GPU gratuita de Google Colab, trabajando directamente desde VSCode sin subir archivos manualmente.

---

## 📋 Requisitos Previos

### 1. Cuenta Google Colab
- **Colab Free**: GPU T4 (~18-20 horas de entrenamiento)
- **Colab Pro** ($10/mes): GPU V100/A100, tiempo ilimitado (~10-12 horas)
- **Colab Pro+** ($50/mes): GPU A100, prioridad máxima (~6-8 horas)

### 2. Extensión VSCode-Colab (REQUERIDA)
```bash
# Instalar extensión en VSCode
code --install-extension ms-toolsai.vscode-jupyter-colab
```

O instalar desde VSCode:
1. `Cmd+Shift+P` → "Extensions: Install Extensions"
2. Buscar: **"Colab"**
3. Instalar: **"Colab" por Google**

### 3. Autenticación Google
- La extensión te pedirá autenticarte con tu cuenta de Google
- Acepta permisos para acceder a Colab
- Se guardará la sesión para futuros usos

### 4. Google Drive (Opcional pero recomendado)
- Para guardar checkpoints automáticamente
- Evita perder progreso si la sesión se desconecta

---

## 🚀 Inicio Rápido (Desde VSCode)

### Paso 1: Abrir Notebook en VSCode
1. En VSCode, abre `codigo/akorn/train_mnist_colab.ipynb`
2. VSCode detectará automáticamente que es un notebook Jupyter

### Paso 2: Seleccionar Kernel de Colab
1. Click en "Select Kernel" (esquina superior derecha del notebook)
2. Selecciona **"Connect to Colab"** o **"Google Colab"**
3. Se abrirá ventana de autenticación de Google → **Autorizar**
4. Espera a que se conecte (~10-30 segundos)
5. Verás "Colab (Python 3.x)" como kernel activo ✓

### Paso 3: Habilitar GPU en Colab
**IMPORTANTE**: Desde VSCode no puedes cambiar el runtime directamente. Debes:

**Opción A - Desde VSCode (Recomendado)**:
1. La primera celda del notebook verificará GPU automáticamente
2. Si no hay GPU, el notebook te dará instrucciones

**Opción B - Forzar GPU manualmente**:
1. Abre el notebook en navegador: [colab.research.google.com](https://colab.research.google.com)
2. Runtime → Change runtime type → GPU (T4)
3. Regresa a VSCode y reconecta

### Paso 4: Ejecutar Setup (Celdas 1-5 desde VSCode)
Ejecuta cada celda haciendo click en el ícono ▶️ a la izquierda:

- **Celda 1**: Verificar GPU ✅
  ```
  Esperado: "CUDA disponible: True"
            "GPU: Tesla T4"
  ```

- **Celda 2**: Clonar repositorio
  ```python
  # Solo necesario si trabajas desde Colab remoto
  # OMITIR si usas vscode-colab (archivos locales se sincronizan)
  ```

- **Celda 3**: Instalar dependencias (~2 min)
  ```
  Verás barras de progreso de pip install
  ```

- **Celda 4**: Descargar MNIST (~30 seg)
  ```
  Verás "✅ Train: 60000 imágenes"
  ```

- **Celda 5**: Montar Google Drive (opcional)
  ```
  Click en link → Autorizar → Copiar código
  ```

**Tiempo total setup**: ~3-5 minutos

### Paso 5: Iniciar Entrenamiento (Celda 6)
```python
!python train_classification.py mnist_maxima_precision \
    --data mnist \
    --epochs 200 \
    --n 4 \
    --L 3 \
    --T 5 \
    --ch 128 \
    --batchsize 128 \
    --device cuda
```

**Tiempo**: ~18 horas (GPU T4)

### Paso 6: Monitorear con TensorBoard (Celda 8)
Ejecutar en VSCode:
```python
%load_ext tensorboard
%tensorboard --logdir runs/mnist_maxima_precision
```

**Resultado**: Se abrirá TensorBoard en un panel de VSCode mostrando:
- Training loss (bajando)
- Test accuracy (subiendo hacia 99%)
- Adversarial robustness

---

## 🔑 Características de vscode-colab

### ✅ Ventajas
1. **Trabajas en VSCode**: Editor familiar, atajos de teclado
2. **Archivos locales**: No necesitas subir/descargar archivos
3. **Sincronización automática**: Cambios en notebook se guardan localmente
4. **GPU gratuita**: Colab ejecuta en la nube
5. **Debugging**: Puedes usar variables inspector de VSCode
6. **Multi-archivo**: Editas otros archivos mientras el notebook corre

### ⚠️ Limitaciones
1. **Conexión persistente**: Mantén VSCode abierto durante entrenamiento
2. **Sincronización inicial**: Primera vez puede tardar ~1 min
3. **Archivos grandes**: Checkpoints se guardan en Colab, descarga manual después
4. **Runtime settings**: Cambiar GPU requiere abrir navegador

---

## 📊 Workflow Completo con vscode-colab

```
Tu Mac (VSCode)                    Servidores Colab (GPU)
─────────────────                  ──────────────────────
                                   
1. Abrir notebook      ─────────▶  Conectar a runtime
   train_mnist_colab.ipynb         
                                   
2. Seleccionar kernel  ◀─────────  Autenticar Google
   "Connect to Colab"              
                                   
3. Ejecutar celdas     ─────────▶  Código corre en GPU T4
   (Click ▶️)                       
                                   
4. Ver outputs         ◀─────────  Resultados se muestran
   en VSCode                       en VSCode
                                   
5. TensorBoard abierto ◀─────────  Métricas en tiempo real
   en panel VSCode                 
                                   
6. Checkpoints         ◀─────────  Guardar en Drive
   guardar en Drive                cada 25 epochs
                                   
7. Descargar           ◀─────────  Al finalizar:
   resultados finales              zip y download
```

---

## 🔧 Configuración Inicial (Primera Vez)

### Instalar Extensión vscode-colab

**Método 1 - Desde VSCode**:
1. `Cmd+Shift+X` (abrir extensiones)
2. Buscar: **"Colab"**
3. Instalar: **"Colab" by Google**
4. Reload VSCode

**Método 2 - Terminal**:
```bash
code --install-extension ms-toolsai.vscode-jupyter-colab
```

### Verificar Instalación
1. Abre cualquier `.ipynb`
2. Click en "Select Kernel" (arriba derecha)
3. Deberías ver opción **"Connect to Colab"**

Si no aparece:
- Reinstala extensión
- Reinicia VSCode completamente
- Verifica que Jupyter extension esté instalada

---

## 🚦 Estado de Conexión

Verás indicadores en VSCode:

| Indicador | Significado |
|-----------|-------------|
| 🟢 "Colab (Python 3.10)" | Conectado, listo para ejecutar |
| 🟡 "Connecting..." | Conectando a Colab |
| 🔴 "Disconnected" | Sin conexión, click para reconectar |
| ⚙️ "Busy" | Celda ejecutándose |

---

## 📊 Configuración de Entrenamiento

### Parámetros Máxima Precisión
| Parámetro | Valor | Significado |
|-----------|-------|-------------|
| `--epochs` | 200 | Épocas de entrenamiento |
| `--n` | 4 | Dimensión de osciladores Kuramoto |
| `--L` | 3 | Número de capas Kuramoto |
| `--T` | 5 | Timesteps por capa |
| `--ch` | 128 | Canales base (features) |
| `--batchsize` | 128 | Imágenes por batch |
| `--lr` | 0.0001 | Learning rate (default) |
| `--device` | cuda | Usar GPU |

### Recursos GPU Necesarios
- **Memoria GPU**: ~8-10 GB
- **Tiempo (T4)**: ~18-20 horas
- **Tiempo (V100)**: ~10-12 horas
- **Tiempo (A100)**: ~6-8 horas

---

## 📈 Resultados Esperados

### Accuracy por Epoch
```
Epoch   0:  ~10% (random baseline)
Epoch  10:  ~87%
Epoch  25:  ~92%
Epoch  50:  ~95%
Epoch 100:  ~97%
Epoch 150:  ~98%
Epoch 200:  ~99% ← Meta
```

### Robustez Adversarial
| Ataque | Accuracy Esperada |
|--------|-------------------|
| Clean (sin ataque) | ~99% |
| Random Noise | ~96% |
| FGSM (ε=8/255) | ~90% |
| PGD (ε=8/255) | ~80% |

---

## 💾 Guardado de Checkpoints

### Automático (cada 25 epochs)
```
runs/mnist_maxima_precision/
├── checkpoint_epoch_025.pth  (~2 GB)
├── checkpoint_epoch_050.pth
├── checkpoint_epoch_075.pth
├── checkpoint_epoch_100.pth
├── checkpoint_epoch_125.pth
├── checkpoint_epoch_150.pth
├── checkpoint_epoch_175.pth
└── checkpoint_epoch_200.pth
```

### Copiar a Google Drive
El notebook automáticamente copia checkpoints a:
```
/content/drive/MyDrive/akorn_mnist_maxima/
```

### Descargar Resultados Completos
```bash
# En Colab (celda 13)
!zip -r mnist_maxima_results.zip runs/mnist_maxima_precision/
files.download('mnist_maxima_results.zip')
```

---

## 🔧 Troubleshooting

### ❌ Problema: No aparece opción "Connect to Colab"
**Causa**: Extensión no instalada correctamente

**Solución**:
```bash
# Desinstalar e instalar limpiamente
code --uninstall-extension ms-toolsai.vscode-jupyter-colab
code --install-extension ms-toolsai.vscode-jupyter-colab

# Reiniciar VSCode completamente
# Abrir notebook nuevamente
```

### ❌ Problema: "Authentication failed"
**Causa**: Token de Google expirado o permisos denegados

**Solución**:
1. `Cmd+Shift+P` → "Colab: Sign Out"
2. Volver a conectar → Re-autenticar
3. Asegurar que aceptas **todos** los permisos

### ❌ Problema: Conexión muy lenta al iniciar
**Causa**: Primera conexión sincroniza archivos

**Solución**:
- **Normal**: 30 seg - 2 min primera vez
- **Lento (>5 min)**: Verifica tu conexión a internet
- **Tip**: Cierra archivos/carpetas grandes en VSCode antes de conectar

### ❌ Problema: GPU Out of Memory (OOM)
**Solución 1**: Reducir batch size
```bash
--batchsize 64  # En lugar de 128
```

**Solución 2**: Reducir canales
```bash
--ch 96  # En lugar de 128
```

**Solución 3**: Reducir timesteps
```bash
--T 4  # En lugar de 5
```

### ❌ Problema: Sesión de Colab se desconecta
**Prevención**:
1. **Mantén VSCode abierto**: No cierres la ventana
2. **Evita suspender Mac**: Ajusta Energy Saver
3. Usar Colab Pro (sesiones ilimitadas)
4. Montar Google Drive y guardar checkpoints frecuentemente

**Script Anti-Idle** (ejecutar en celda aparte):
```python
# Mantener sesión activa (ejecutar en paralelo)
import time
from IPython.display import clear_output

while True:
    time.sleep(300)  # Cada 5 minutos
    clear_output()
    print("Session active...")
```

**Recuperación**:
```bash
# Si se desconecta, reconectar y reanudar
--finetune runs/mnist_maxima_precision/checkpoint_epoch_XXX.pth
```

### ❌ Problema: "Cannot connect to Colab runtime"
**Solución**:
1. Verifica internet: `ping google.com`
2. Abre navegador → [colab.research.google.com](https://colab.research.google.com)
3. Crea notebook vacío → Si funciona, problema es con vscode-colab
4. Reinicia VSCode y reconecta

### ❌ Problema: Archivos no se sincronizan a Colab
**Causa**: vscode-colab solo sube el notebook, no carpetas enteras

**Solución - Opción A (Clonar repo en Colab)**:
```python
# En celda del notebook:
!git clone https://github.com/ACRCris/Proyecto-Inv.-teorica..git
%cd Proyecto-Inv.-teorica./codigo/akorn
```

**Solución - Opción B (Subir archivos específicos)**:
```python
# Subir solo archivos necesarios
from google.colab import files
# Arrastra y suelta archivos en la celda
```

**Solución - Opción C (Usar Drive)**:
```python
from google.colab import drive
drive.mount('/content/drive')
# Copiar proyecto a Drive primero
```

### ❌ Problema: "MNIST dataset not found"
**Solución**:
```python
# Ejecutar celda 4 del notebook
import torchvision
trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True)
```

### ❌ Problema: Accuracy no mejora (estancada en ~80%)
**Diagnóstico**:
- Learning rate muy alto → Reducir `--lr 0.00005`
- T muy bajo → Aumentar `--T 6` o `--T 7`
- No suficientes epochs → Continuar entrenamiento

### ❌ Problema: Entrenamiento muy lento (>30 horas)
**Causas posibles**:
1. GPU no habilitada → Verificar en Runtime settings
2. Batch size muy pequeño → Aumentar a 128
3. GPU compartida (Colab Free) → Esperar o usar Colab Pro

---

## 📊 Monitoreo en Tiempo Real

### TensorBoard (Recomendado)
```python
# En Colab
%load_ext tensorboard
%tensorboard --logdir runs/mnist_maxima_precision
```

**Métricas clave**:
- `train/loss` → debe bajar consistentemente
- `train/acc` → debe subir hasta ~99%
- `test/acc` → accuracy en test set
- `test/adv_acc` → robustez adversarial

### Memoria GPU
```python
# Ejecutar en celda aparte (celda 9)
import torch
mem_allocated = torch.cuda.memory_allocated(0) / 1e9
print(f"GPU Memoria: {mem_allocated:.2f} GB")
```

### Logs en Terminal
```bash
# Ver últimas 50 líneas
!tail -n 50 runs/mnist_maxima_precision/train.log
```

---

## 🔄 Flujo Completo de Trabajo

### Primera Vez (Setup completo)
```
1. Abrir train_mnist_colab.ipynb en Colab
2. Runtime → Change runtime → GPU (T4)
3. Ejecutar celdas 1-5 (setup)
4. Ejecutar celda 6 (entrenamiento)
5. Ejecutar celda 8 (TensorBoard)
6. Esperar ~18 horas
7. Ejecutar celda 11 (evaluar)
8. Ejecutar celda 13 (descargar)
```

### Reanudar Entrenamiento Interrumpido
```python
!python train_classification.py mnist_maxima_precision \
    --data mnist \
    --finetune runs/mnist_maxima_precision/checkpoint_epoch_100.pth \
    --epochs 200 \
    --n 4 --L 3 --T 5 --ch 128 --batchsize 128
```

### Solo Evaluar (sin entrenar)
```python
# Celda 11 del notebook
checkpoint = torch.load('checkpoint_epoch_200.pth')
model.load_state_dict(checkpoint['model_state_dict'])
# ... código de evaluación
```

---

## 📁 Estructura de Archivos

### Antes del Entrenamiento
```
codigo/akorn/
├── train_mnist_colab.ipynb  ← Notebook para Colab
├── train_classification.py  ← Script de entrenamiento
├── source/
│   └── models/
│       └── classification/
│           └── knet.py       ← Modelo AKOrN
└── data/                     ← Se crea automáticamente
```

### Después del Entrenamiento
```
codigo/akorn/
├── runs/
│   └── mnist_maxima_precision/
│       ├── checkpoint_epoch_025.pth
│       ├── checkpoint_epoch_050.pth
│       ├── ...
│       ├── checkpoint_epoch_200.pth
│       ├── ema_epoch_200.pth
│       └── events.out.tfevents.*  ← Logs TensorBoard
└── data/
    └── MNIST/
        ├── train/
        └── test/
```

---

## 🎓 Comparación con tu Análisis Kuramoto

| Característica | Tu Análisis | AKOrN (Este entrenamiento) |
|---|---|---|
| **Dataset** | MNIST 60k | MNIST 60k |
| **Dinámicas** | KBlock, T=100 | KLayer ×3, T=5 cada una |
| **Acoplamiento** | α barrido [0, 0.1] | Aprendible (conv) |
| **Kernel size** | 7×7 | [9×9, 7×7, 5×5] |
| **Canales** | 4 | 128 |
| **Objetivo** | Encontrar α_c | Clasificar 0-9 |
| **Output** | α_c escalar | Clase (0-9) |
| **Tiempo (GPU)** | ~62.5 horas | ~18 horas |

---

## 💡 Experimentos Recomendados Post-Entrenamiento

### 1. Correlación α_c vs Timesteps
```python
# Para cada imagen:
# - Obtener α_c de tu análisis SQLite
# - Variar T en AKOrN: [1, 2, 3, 4, 5, 6, 7]
# - Medir confidence
# - Graficar: α_c vs confidence(T)

# Hipótesis: Imágenes con α_c alto requieren más T
```

### 2. Análisis de Parámetro de Orden R(t)
```python
# Modificar KLayer para retornar R(t)
x_final, xs, es = klayer(x, return_xs=True, return_es=True)
R_t = calcular_orden_parameter(xs)

# Comparar con tu R(t) del análisis Kuramoto
```

### 3. Variación de Configuraciones
| Experimento | Modificar | Objetivo |
|-------------|-----------|----------|
| **Exp A** | T=1,2,3,5,7,10 | ¿Más T = mejor? |
| **Exp B** | L=1,2,3,4 | ¿Más capas = mejor? |
| **Exp C** | n=2,3,4,5 | ¿Dimensión óptima? |
| **Exp D** | ch=64,96,128,256 | ¿Trade-off speed/accuracy? |

---

## 📞 Soporte y Referencias

### Documentación Local
- **Análisis técnico**: `/proyecto/ANALISIS_AKORN_MNIST.md`
- **Resumen ejecutivo**: `/proyecto/AKORN_RESUMEN_EJECUTIVO.md`
- **Guía MNIST**: `codigo/akorn/GUIA_MNIST.md`

### Paper Original
- **AKOrN (ICLR 2025)**: https://arxiv.org/abs/2410.13821
- **Kuramoto Networks**: Referencias en el paper

### Código Fuente
- **Repositorio**: https://github.com/ACRCris/Proyecto-Inv.-teorica.
- **Modelo AKOrN**: `codigo/akorn/source/models/classification/knet.py`
- **KLayer**: `codigo/akorn/source/models/classification/klayer.py`

---

## ✅ Checklist Pre-Entrenamiento

Antes de iniciar el entrenamiento de 18 horas, verifica:

- [ ] GPU habilitada en Colab (Runtime → Change runtime)
- [ ] Verificado que GPU es T4/V100/A100 (celda 1)
- [ ] MNIST descargado correctamente (celda 4)
- [ ] Google Drive montado para guardar checkpoints (celda 5)
- [ ] TensorBoard funcionando (celda 8)
- [ ] Memoria GPU suficiente (~10 GB disponibles)
- [ ] Sesión de Colab estable (Pro recomendado)
- [ ] Computadora/pestaña se mantendrá activa (o script anti-idle)

---

## 🎯 Objetivo Final

Al completar este entrenamiento, tendrás:

✅ Modelo AKOrN entrenado con ~99% accuracy en MNIST  
✅ 8 checkpoints guardados (cada 25 epochs)  
✅ Logs TensorBoard con métricas de entrenamiento  
✅ Evaluación de robustez adversarial  
✅ Base para comparar con tu análisis Kuramoto α_c  
✅ Modelo listo para experimentos de correlación  

---

**Tiempo total estimado**: ~20 horas (incluyendo setup)  
**Última actualización**: Noviembre 2025  
**Estado**: Listo para ejecutar en Colab GPU
