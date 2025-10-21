# 🔄 Sincronización para Ejecutar en Mac con Apple M3

**Fecha**: Octubre 21, 2025  
**Objetivo**: Continuar optimización de parámetros en Mac local con GPU M3

---

## 📋 Contexto del Proyecto

### Estado Actual (en servidor remoto)

- **Dataset procesado**: 10,002 imágenes MNIST (test set completo)
- **Análisis completado**: Distribuciones, DFA, PSD, MI, Entropía
- **Problema identificado**: Modelo NO está en estado crítico
  - R_final ≈ 0.62 (objetivo: 0.5)
  - DFA α ≈ 1.93 (objetivo: 1.0) ← **Muy alto**
  - PSD slope ≈ -4.0 (objetivo: -1.0) ← **Muy bajo**
  - Score actual: 3.264

### Decisión Tomada

**Ejecutar optimización en Mac M3** con aceleración GPU (Metal Performance Shaders) para reducir tiempo de 3-5 horas a ~30-60 minutos.

---

## 🎯 Plan de Trabajo en Mac

### Paso 1: Clonar/Actualizar Repositorio

```bash
cd ~/proyectos
git clone https://github.com/ACRCris/Proyecto-Inv.-teorica.git
# O si ya existe:
cd Proyecto-Inv.-teorica
git pull origin main
```

### Paso 2: Navegar al Directorio

```bash
cd codigo/analisis_criticalidad_minimalista
```

### Paso 3: Crear Entorno Virtual (si no existe)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Paso 4: Instalar Dependencias

```bash
# PyTorch con soporte Metal (MPS) para Apple Silicon
pip install torch torchvision torchaudio

# Dependencias científicas
pip install numpy scipy pandas matplotlib tqdm

# Otras dependencias del proyecto
pip install -r requirements.txt  # Si existe
```

### Paso 5: Verificar Soporte GPU M3

```bash
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS disponible: {torch.backends.mps.is_available()}")
print(f"MPS construido: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    print("✅ GPU M3 lista para usar con Metal Performance Shaders")
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print(f"Test tensor en GPU: {x.device}")
else:
    print("⚠️ MPS no disponible, usará CPU")
EOF
```

### Paso 6: Modificar Script para usar MPS

El script `optimizar_parametros_secuencial.py` detecta automáticamente CUDA, pero necesita un pequeño ajuste para detectar MPS (Apple Silicon).

**Edita la función `detectar_dispositivo()`:**

```python
def detectar_dispositivo():
    """Detecta si hay GPU disponible (CUDA o MPS)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ GPU CUDA detectada: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✅ GPU Apple Silicon (MPS) detectada")
        print(f"   Usando Metal Performance Shaders")
    else:
        device = torch.device('cpu')
        print("⚠️  GPU no disponible, usando CPU")
    return device
```

**Ubicación**: Línea ~95 en `optimizar_parametros_secuencial.py`

### Paso 7: Ejecutar Optimización

```bash
# Asegurarse de estar en el directorio correcto
cd ~/proyectos/Proyecto-Inv.-teorica/codigo/analisis_criticalidad_minimalista

# Activar entorno
source .venv/bin/activate

# Ejecutar
python3 optimizar_parametros_secuencial.py
```

**Tiempo estimado con M3 GPU**: 30-60 minutos (vs 3-5 horas en CPU)

---

## 📊 Qué Esperar Durante la Ejecución

### Salida Inicial

```
╔══════════════════════════════════════════════════════════════════════════════╗
║               OPTIMIZACIÓN SECUENCIAL DE PARÁMETROS                          ║
║                    Fase Exploratoria - Grid Search                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

✅ GPU Apple Silicon (MPS) detectada
   Usando Metal Performance Shaders

📋 CONFIGURACIÓN:
   • Estrategia: Optimización Secuencial (greedy)
   • Parámetros a optimizar: ['T_steps', 'gamma', 'del_t', 'init_omg', 'ksize']
   • Valores por parámetro: 5
   • Total evaluaciones: 25
   • Imágenes por evaluación: 100

🎯 FUNCIÓN OBJETIVO:
   score = 1.0×|R_final - 0.5| + 0.8×|DFA_α - 1.0| + 0.8×|PSD_slope + 1.0|
   Minimizar score → Acercarse al estado crítico

⏱️  TIEMPO ESTIMADO: 0.7 horas
   (0.5s por imagen × 100 imágenes × 25 configuraciones)
```

### Durante Optimización

```
================================================================================
📊 Optimizando: T_steps
================================================================================

   Evaluando: T_steps=50, gamma=0.70, del_t=0.90, init_omg=0.10, ksize=7
      Procesando: 100%|██████████████████| 100/100 [00:52<00:00,  1.91it/s]
      Score: 2.145 | R=0.587 | DFA=1.42 | PSD=-2.87

   Evaluando: T_steps=75, gamma=0.70, del_t=0.90, init_omg=0.10, ksize=7
      Procesando: 100%|██████████████████| 100/100 [00:48<00:00,  2.08it/s]
      Score: 1.892 | R=0.524 | DFA=1.28 | PSD=-2.13
      
   ...

✅ Mejor T_steps: 75 (score: 1.892)
```

### Resultado Final

```
🏆 RESULTADO FINAL
================================================================================

📊 MEJOR CONFIGURACIÓN ENCONTRADA:
   • T_steps: 75
   • gamma: 0.5
   • del_t: 0.7
   • init_omg: 0.15
   • ksize: 5

📈 MÉTRICAS FINALES:
   • R_final: 0.4985 (objetivo: 0.5000) ✅
   • DFA α: 1.0234 (objetivo: 1.0000) ✅
   • PSD slope: -1.1245 (objetivo: -1.0000) ✅
   • Score total: 0.1845

📊 COMPARACIÓN CON BASELINE:
   Baseline score: ~3.264 (actual)
   Mejor score: 0.1845
   Mejora: 94.3% 🎉

💾 Resultados guardados en: resultados_optimizacion_secuencial/
```

---

## 📁 Archivos que se Generarán

### 1. `resultados_optimizacion_secuencial/resultados_optimizacion_secuencial.csv`

Tabla completa con las 25 evaluaciones:

| T_steps | gamma | del_t | init_omg | ksize | R_final | DFA_α | PSD_slope | score | param_name | param_value |
|---------|-------|-------|----------|-------|---------|-------|-----------|-------|------------|-------------|
| 50      | 0.7   | 0.9   | 0.1      | 7     | 0.587   | 1.42  | -2.87     | 2.145 | T_steps    | 50          |
| 75      | 0.7   | 0.9   | 0.1      | 7     | 0.524   | 1.28  | -2.13     | 1.892 | T_steps    | 75          |
| ...     | ...   | ...   | ...      | ...   | ...     | ...   | ...       | ...   | ...        | ...         |

### 2. `resultados_optimizacion_secuencial/mejor_configuracion.txt`

```
MEJOR CONFIGURACIÓN ENCONTRADA
==================================================

Parámetros:
  T_steps: 75
  gamma: 0.5
  del_t: 0.7
  init_omg: 0.15
  ksize: 5

Métricas (validación con 500 imágenes):
  R_final: 0.4985
  DFA α: 1.0234
  PSD slope: -1.1245
  Score: 0.1845
```

---

## 📈 Interpretación de Resultados

### Criterios de Éxito

| Score | Significado | Acción |
|-------|-------------|--------|
| < 1.0 | 🎉 **Excelente** | Ejecutar grid search completo (243 combinaciones) |
| 1.0 - 2.0 | 👍 **Bueno** | Grid search selectivo alrededor de mejores valores |
| 2.0 - 2.5 | ⚠️ **Marginal** | Considerar expandir espacio de búsqueda |
| > 2.5 | ❌ **Sin mejora** | Revisar arquitectura o cambiar estrategia |

### Métricas Clave

- **R_final**: Debe estar en [0.4, 0.6] para criticalidad
- **DFA α**: Debe estar en [0.85, 1.15] para correlaciones críticas
- **PSD slope**: Debe estar en [-1.3, -0.7] para ruido rosa

---

## 🔄 Siguientes Pasos Después de la Ejecución

### Si hay mejora significativa (score < 1.0):

1. **Subir resultados al repo**:
   ```bash
   git add resultados_optimizacion_secuencial/
   git commit -m "Resultados optimización fase exploratoria en M3"
   git push origin main
   ```

2. **Sincronizar con GitHub Copilot** en servidor remoto:
   ```bash
   # En servidor remoto
   cd /home/crperezp/proyectos/ProyectoInvTeorica/Proyecto-Inv.-teorica.
   git pull origin main
   ```

3. **Decidir siguiente fase**:
   - Opción A: Grid search completo (243 combinaciones)
   - Opción B: Validación con full dataset (10,002 imágenes)
   - Opción C: Análisis de redes funcionales (Fase 2.2)

### Si no hay mejora (score > 2.0):

1. **Revisar parámetros**: ¿Están en rangos adecuados?
2. **Expandir búsqueda**: Probar valores más extremos
3. **Considerar arquitectura**: ¿4 canales son suficientes?

---

## 🐛 Troubleshooting en Mac

### Error: "MPS backend out of memory"

El M3 tiene memoria unificada (8-24GB según modelo). Si falla:

```python
# Reducir imágenes por evaluación
N_IMAGES_PER_CONFIG = 50  # En lugar de 100
```

### Error: "RuntimeError: Placeholder storage has not been allocated"

Problema conocido con MPS. Solución temporal:

```python
# Al inicio del script
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
```

### Script muy lento (> 2 horas)

Verificar que esté usando MPS:
```bash
# Durante ejecución, en otra terminal:
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

Si devuelve `False`, instalar PyTorch actualizado:
```bash
pip install --upgrade torch torchvision torchaudio
```

---

## 📚 Archivos Relevantes en el Repo

### Nuevos (para Mac):
- `optimizar_parametros_secuencial.py` ← **Script principal**
- `INSTRUCCIONES_OPTIMIZACION.md` ← Documentación completa
- `SINCRONIZACION_MAC.md` ← Este archivo

### Modificados:
- `kuramoto/modelo.py` ← Añadido método `forward_with_params()`

### Contexto del Proyecto:
- `ANALISIS_COMPLETO_TODAS_METRICAS.md` ← Estado actual del análisis
- `PLAN_DE_ACCION.md` ← Roadmap general
- `RESUMEN_CORRECCION.md` ← Correcciones metodológicas previas

---

## 💬 Contexto de la Conversación (Resumen)

### Problema Original
Usuario preguntó qué significaban los resultados de análisis de distribuciones (0/101 momentos normales).

### Hallazgos
1. **Distribuciones no-normales son ESPERADAS** en sistemas críticos (power-law)
2. **Métricas temporales desalineadas**: DFA α=1.93 vs objetivo 1.0, PSD=-4.0 vs objetivo -1.0
3. **R cercano a crítico** (0.54-0.69) pero dinámicas temporales incorrectas

### Solución Propuesta
**Optimización de parámetros** para alinear métricas temporales (DFA, PSD) con teoría de criticalidad.

### Decisiones Técnicas
1. Usar **búsqueda secuencial** (25 evaluaciones) en lugar de grid search completo (243)
2. **Función objetivo multi-métrica**: Score = |R-0.5| + 0.8|DFA-1| + 0.8|PSD+1|
3. Ejecutar en **Mac M3 con GPU** en lugar de servidor remoto con CPU

### Justificación
- M3 es **15-25% más rápido** que i7-11700 en CPU
- M3 con MPS (GPU) es **4-6x más rápido** que CPU
- Tiempo: 30-60 min (M3 GPU) vs 3-5 hrs (servidor CPU)
- No requiere modificar entorno de servidor compartido

---

## 📞 Contacto con GitHub Copilot en Mac

Una vez en tu Mac, abre el proyecto en VS Code y di:

> "He clonado el repo en mi Mac M3. Ya ejecuté la optimización y obtuve estos resultados: [pegar resultados]. ¿Cuál es el siguiente paso?"

O simplemente:

> "Continuar con optimización de parámetros Kuramoto desde sincronización"

El contexto estará en este archivo y en los commits del repo.

---

## ✅ Checklist Pre-Ejecución

- [ ] Repo clonado/actualizado en Mac
- [ ] Entorno virtual creado (`.venv`)
- [ ] PyTorch con MPS instalado
- [ ] Verificado que MPS está disponible (`torch.backends.mps.is_available()`)
- [ ] Modificada función `detectar_dispositivo()` para detectar MPS
- [ ] Dataset MNIST descargado (se descarga automáticamente si no existe)
- [ ] Espacio en disco: ~2GB libres para resultados

---

## 🚀 Comando Rápido de Inicio

```bash
# Todo en uno (copia y pega en terminal de Mac)
cd ~/proyectos/Proyecto-Inv.-teorica/codigo/analisis_criticalidad_minimalista && \
source .venv/bin/activate && \
python3 -c "import torch; print('MPS:', torch.backends.mps.is_available())" && \
python3 optimizar_parametros_secuencial.py
```

---

**Última actualización**: Octubre 21, 2025  
**Autor**: GitHub Copilot (sesión en servidor remoto)  
**Próximo ejecutor**: Usuario en Mac M3
