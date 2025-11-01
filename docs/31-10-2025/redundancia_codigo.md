# Informe: redundancia de código
Fecha: 31-10-2025

Resumen ejecutivo
------------------
Este informe recoge las observaciones sobre código duplicado / solapado detectadas en el repositorio relacionadas con la implementación del modelo Kuramoto y el cálculo de \alpha_c. Está pensado para compartir con el equipo y priorizar refactorizaciones de bajo riesgo.

Hallazgos principales
---------------------

1) Módulo dinámico y utilidades de estado
- Archivo fuente principal: `codigo/analisis_criticalidad_minimalista/kuramoto/modelo.py` (contiene `KBlock`, `KConv2d`, `ModReLU`, utilidades `reshape`/`reshape_back`, `normalize`, `compute_energy`).
- Duplicaciones detectadas: instanciaciones y pequeñas variantes de `KBlock` se repiten en:
  - `encontrar_alpha_critico.py`, `encontrar_alpha_critico_clase3.py`
  - `analizar_imagen_individual.py`
  - `verificar_consistencia.py` y `verificar_scripts_alpha_c.py`
  - Notebook: `codigo/kuramoto_pruebas_basico.ipynb` contiene una versión inline de `KBlock` / `KConv2d` y helpers.

Impacto: si cambian las firmas o parámetros por defecto (p. ej. n, ch, T), es necesario actualizar muchas copias.

2) Métricas de orden y transformaciones temporales
- Implementación central: `codigo/analisis_criticalidad_minimalista/analisis/criticalidad.py` (clase `KuramotoMetrics` con `order_parameter` y otras métricas).
- Duplicaciones/solapamientos detectados en:
  - `PlanDeAccion2/akorn_criticality.py` (usa `KuramotoMetrics` pero también realiza transformaciones adicionales sobre series)
  - Notebooks y scripts con pequeñas variantes de `order_parameter` y cálculo de magnitudes/series.

Impacto: riesgo de resultados ligeramente inconsistentes por pequeñas diferencias numéricas (p. ej. uso de atan2 vs torch.angle, normalizaciones distintas).

3) I/O y checkpoints
- Problema: múltiples scripts usan la misma lógica para guardar y cargar checkpoints y distribuciones con un patrón repetido:
  - `torch.save(obj, path)`
  - `try: torch.load(path, weights_only=False)` / except TypeError: torch.load(path)
- Archivos afectados: `analizar_incremental_clases_impares.py`, `graficar_distribucion_criticalidad.py`, `generar_graficas_checkpoints.py`, scripts en `codigo/akorn/` (train/eval), `PlanDeAccion2/akorn_criticality.py`.

Impacto: mezcla de APIs de PyTorch entre versiones; cada script re-implementa el mismo fallback.

4) Graficado y animaciones
- Múltiples utilidades para plot/animación implementadas en:
  - `graficar_distribucion_criticalidad.py`
  - `generar_graficas_checkpoints.py`
  - Celdas de notebooks (animaciones y visualizaciones de dinámicas)
- Comportamiento repetido: histograma + anotación de media/desvío, extracción de frames desde series `xs`, transformaciones para quiver/heatmap.

Impacto: mantenimiento y corrección de bugs (p. ej. IndexError en color de parche) deben aplicarse en varias ubicaciones.

5) Parámetros hard-coded
- Valores que aparecen repetidos en muchos scripts/notebooks: `ch`, `n`, `T`, `gamma`, `del_t`, `ksize`, `init_omg`. Esto dificulta experimentación reproducible.

Recomendaciones (priorizadas)
--------------------------------

1) Extraer utilidades I/O (alta prioridad, bajo riesgo)
- Crear un helper `utils/io.py` con `load_checkpoint(path, map_location='cpu')` y `save_checkpoint(obj, path)` que gestione el fallback `weights_only` y cree directorios padres. Reemplazar llamadas directas a `torch.load`/`torch.save` por este helper en los scripts de análisis y graficado.

2) Centralizar fábrica de `KBlock` (media prioridad)
- Crear `kuramoto/factory.py` o `kuramoto/config.py` con `make_kblock(**params)` que aplique los valores por defecto del proyecto. Sustituir instanciaciones directas en scripts y tests.

3) Consolidar métricas y transformaciones (media prioridad)
- Refactor: mover wrappers y funciones repetidas a `analisis/metrics.py` o ampliar `KuramotoMetrics` para exponer helpers comunes (magnitudes, medias, normalizaciones). Es importante elegir una única implementación numéricamente robusta (uso de torch.angle/complex cuando posible).

4) Unificar plotting/animaciones (baja prioridad técnica, alta en consistencia)
- Crear `analisis/plotting.py` con funciones `plot_distribution`, `plot_order_curve`, `animate_dynamics`. Dejar scripts CLI delgados que llamen a estas funciones.

5) Parametrizar y documentar defaults (alta prioridad para reproducibilidad)
- Añadir `analisis/defaults.py` o leer parámetros desde un único `defaults.json`/`yaml` en `codigo/analisis_criticalidad_minimalista/`.

Acciones realizadas (sesión actual)
----------------------------------
- Auditoría de código (busquedas) para localizar duplicaciones.
- Este informe generado en `docs/31-10-2025/redundancia_codigo.md` (esta misma ruta).

Propuesta de siguientes pasos (si quieres que lo haga)
---------------------------------------------------
Opción A — Cambios no invasivos (recomendado primer paso)
- Implementar `utils/io.py` y reemplazar llamadas a `torch.load(..., weights_only=False)` en scripts de análisis (p. ej. `graficar_distribucion_criticalidad.py`, `generar_graficas_checkpoints.py`, `analizar_incremental_clases_impares.py`).

Opción B — Refactorización moderada
- Añadir `kuramoto/factory.py` y actualizar 3–5 scripts/notebooks como prueba de concepto; añadir test que valide que `make_kblock` crea bloques equivalentes.

Opción C — Refactorización completa
- Crear módulos `analisis/plotting.py`, `analisis/metrics.py`, `kuramoto/factory.py`, mover lógica duplicada y añadir batería de tests. Esto requiere más tiempo y pruebas.

Si confirmas, procedo con la opción que elijas. En caso contrario, puedo generar un PR con el cambio no invasivo (Opción A) listo para tu revisión.

---
Archivo generado automáticamente por petición del equipo — 31-10-2025
