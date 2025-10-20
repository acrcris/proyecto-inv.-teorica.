<!-- .github/copilot-instructions.md -->
# Instrucciones rápidas para agentes de codificación (AI)

Objetivo: permitir que un agente IA sea productivo inmediatamente explorando la parte experimental y el análisis de criticalidad en este repositorio.

- Contexto rápido
  - Este repositorio mezcla dos líneas principales:
    1) investigación en dinámica de osciladores / análisis de criticalidad: `codigo/analisis_criticalidad_minimalista/` (módulo Python con `main.py`, `ejemplo_completo.py`).
    2) implementación/experimentos de AKOrN (modelos y entrenamiento): `codigo/akorn/` (scripts de entrenamiento/evaluación, `requirements.txt`).

- Qué revisar primero
  - `codigo/analisis_criticalidad_minimalista/README.md` — arquitectura del módulo, ejemplo de uso, scripts de ejemplo y pipeline (`main.py`, `ejemplo_completo.py`).
  - `codigo/akorn/README.md` — instrucciones de conda/entrenamiento, uso de `accelerate`, descarga de datasets (`data/download_*.sh`) y comandos de evaluación (`eval_obj.py`).
  - `codigo/analisis_criticalidad_minimalista/` — examina `kuramoto/`, `analisis/`, `datasets/` y `utils/` para entender las interfaces (KBlock, KConv2d, KuramotoMetrics, Visualizador).

- Entorno y comandos esenciales
  - Entorno virtual: el proyecto asume un venv/conda. Ejemplo en README del módulo:
    - source /home/crperezp/proyectos/ProyectoInvTeorica/Proyecto-Inv.-teorica./codigo/.venv/bin/activate
    - para AKOrN se ofrece un ejemplo con conda `conda create -n akorn python=3.12` y `pip install -r requirements.txt`.
  - Datos: varios scripts `codigo/akorn/data/download_*.sh` descargan datasets (CLEVRTex, pascal, satnet, synths). Revisa `codigo/akorn/data/` antes de ejecutar entrenamientos.
  - Entradas/entrypoints:
    - Análisis Kuramoto: `codigo/analisis_criticalidad_minimalista/main.py` y `ejemplo_completo.py` (ejecutar para reproducir el notebook `kuramoto_pruebas_basico.ipynb`).
    - AKOrN training/eval: `codigo/akorn/train_obj.py`, `train.py`, `eval_obj.py` (uso de `accelerate` para multi-GPU).

- Estilos y convenciones del proyecto
  - Estructura modular: los módulos científicos (kuramoto, analisis, datasets, utils) exponen clases y funciones desde `__init__.py`. Prefiere usar esas interfaces en lugar de acceder a módulos internos.
  - Scripts de reproducción: `ejemplo_completo.py` y `main.py` actúan como tests/herramientas de demostración—útiles para validar cambios rápidamente.
  - Naming: variables como `T`, `ch`, `n`, `ksize`, `gamma` aparecen frecuentemente en firmas de funciones; preserva esas convenciones para compatibilidad.

- Integraciones y dependencias notables
  - PyTorch (torch, torchvision) y numpy/scipy. AKOrN usa `accelerate` para multi-GPU.
  - Visualización y análisis: matplotlib, scikit-learn, einops.
  - Datasets externos: descarga por scripts en `codigo/akorn/data/`.

- Patterns útiles y ejemplos concretos
  - Dinámica: `KBlock.__call__` devuelve (x_final, xs, es) si se piden `return_xs`/`return_es`. Ejemplo en `README.md` del módulo.
  - Métricas: `KuramotoMetrics.order_parameter(xs)` y clases DFA/PSD/Entropia ubicadas en `analisis/`.
  - Visualizador: `Visualizador.plot_energy(es)` y `Animaciones.animate_dynamics(xs, channel=0, filename='canal.gif')`.

- Qué evitar / límites
  - No inventes nuevos datasets ni paths; usa los scripts de descarga existentes y las rutas relativas dentro de `codigo/akorn/data`.
  - Evita cambiar las firmas públicas en `analisis_criticalidad_minimalista` sin actualizar `__init__.py` y los scripts de demostración.

- Tareas de alto valor que un agente puede hacer primero
  1. Ejecutar `ejemplo_completo.py` en el entorno de desarrollo para verificar que la canalización reproduce el notebook (usa el venv definido en `codigo/`).
 2. Añadir tests pequeños: un test de integridad que construya `KBlock` con `T=10` y verifique la longitud de `xs` (véase sección "Testing" en `README.md`).
 3. Documentar cualquier cambio en `README.md` del módulo y en `codigo/akorn/README.md` si afectas comandos de entrenamiento/descarga.

- Referencias clave (usa estas rutas cuando hagas cambios)
  - `codigo/analisis_criticalidad_minimalista/README.md`
  - `codigo/analisis_criticalidad_minimalista/main.py`
  - `codigo/analisis_criticalidad_minimalista/ejemplo_completo.py`
  - `codigo/analisis_criticalidad_minimalista/kuramoto/` (KBlock, KConv2d, ModReLU)
  - `codigo/akorn/README.md`
  - `codigo/akorn/train_obj.py`, `eval_obj.py`, `data/download_clevrtex.sh`

Si alguna sección está incompleta o quieres que añada instrucciones concretas de debugging, tests automáticos o una checklist de PRs, dime qué priorizar y lo actualizo.
