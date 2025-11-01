# Reporte: Última ejecución de scripts Python
Fecha de generación: 31-10-2025 (13:00)

## Resumen ejecutivo

Este reporte documenta el estado de ejecución de los scripts principales del proyecto, basado en logs, archivos de salida, procesos activos y timestamps de modificación.

---

## Scripts activos (en ejecución)

### analizar_incremental_clases_impares.py
- **Estado**: ✅ EN EJECUCIÓN (PID 42912)
- **Inicio**: 2025-10-31 07:28:09
- **Comando**: `python analizar_incremental_clases_impares.py --device mps --max_imgs_per_class 6000 --checkpoint_dir checkpoints_impares --output_dir distribuciones_impares`
- **Última modificación del script**: 2025-10-29 09:34:21
- **Log principal**: `run_impares.log` (204 KB, última actualización 31-oct 12:57)
- **Progreso actual**: 
  - Clase 1: 1000 imágenes procesadas (checkpoint más reciente: 31-oct 12:30)
  - Clase 3: 1000 imágenes procesadas (checkpoint más reciente: 31-oct 12:59)
  - Clase 5: 892 imágenes procesadas (checkpoint más reciente: 31-oct 11:00)
  - Clase 7: 900 imágenes procesadas, actualmente procesando batch 900-999 (6% completado)
  - Clase 9: 900 imágenes procesadas (checkpoint más reciente: 31-oct 12:01)
- **Outputs generados**:
  - Checkpoints: `checkpoints_impares/checkpoint_clase{1,3,5,7,9}_*.pt` (último: clase3_1000imgs.pt @ 31-oct 12:59)
  - Distribuciones acumulativas: `distribuciones_impares/distribucion_acumulativa_*.pt`
- **Observaciones**: Ejecución larga en progreso (>5 horas de CPU time), checkpointing cada 100 imágenes funcionando correctamente. Uso de MPS (Apple Metal) para aceleración.

---

## Scripts ejecutados recientemente (última semana)

### generar_graficas_checkpoints.py
- **Estado**: ✅ COMPLETADO
- **Última ejecución**: 2025-10-31 11:34:20
- **Última modificación del script**: 2025-10-31 11:34:20
- **Outputs generados**:
  - `temp/distribucion_clase1_900imgs.png` (167 KB, 31-oct 11:34)
  - `temp/distribucion_clase3_900imgs.png` (136 KB, 31-oct 11:34)
  - `temp/distribucion_clase5_892imgs.png` (141 KB, 31-oct 11:34)
  - `temp/distribucion_clase7_900imgs.png` (156 KB, 31-oct 11:34)
  - `temp/distribucion_clase9_800imgs.png` (142 KB, 31-oct 11:34)
- **Propósito**: Generar gráficas de distribución de α_c a partir de checkpoints más grandes por clase.
- **Observaciones**: Ejecución exitosa, generó 5 PNGs con histogramas, boxplots y series temporales anotadas con μ y σ.

### graficar_distribucion_criticalidad.py
- **Estado**: ✅ COMPLETADO
- **Última ejecución**: ~2025-10-31 10:40 (inferido por timestamp de modificación)
- **Última modificación del script**: 2025-10-31 10:40:49
- **Propósito**: Graficar distribuciones de criticalidad desde archivos `.pt`.
- **Observaciones**: Script parcheado para manejar PyTorch >=2.6 (`weights_only=False`) y evitar IndexError en matplotlib. Usado para inspeccionar distribuciones intermedias.

### run_kuramoto_TRAIN_MAC.py
- **Estado**: ⏸️ COMPLETADO (ejecución previa)
- **Última ejecución**: ~2025-10-21 04:42 (basado en PID file)
- **Última modificación del script**: 2025-10-22 09:27:53
- **Log principal**: `kuramoto_train_mac.log` (1.1 MB, última actualización 21-oct 04:59)
- **PID**: `kuramoto_train_mac.pid` (proceso ya finalizado)
- **Propósito**: Análisis de criticalidad sobre MNIST train set completo (60,000 imágenes) usando MPS.
- **Outputs esperados**: Resultados guardados (ubicación no documentada en este análisis).
- **Observaciones**: Ejecución completa. Log sugiere siguiente paso: ejecutar `analizar_distribuciones.py` y `analizar_estadisticas_full_dataset.py`.

### test_single_image_referencia.py
- **Estado**: ✅ COMPLETADO
- **Última ejecución**: ~2025-10-22 10:52:14
- **Última modificación del script**: 2025-10-22 10:52:14
- **Output**: `test_referencia_clase3_img0.json` (6.3 KB, 22-oct 10:54)
- **Propósito**: Test de referencia para validar cálculo de α_c en imagen individual.

### encontrar_alpha_critico_clase3.py
- **Estado**: ✅ COMPLETADO
- **Última ejecución**: ~2025-10-22 10:39:21
- **Última modificación del script**: 2025-10-22 10:39:21
- **Output**: `alpha_c_clase3_optimizado.json` (3.5 KB, 22-oct 10:03)
- **Propósito**: Búsqueda optimizada de α_c para clase 3.

### generar_graficas_por_clase.py
- **Estado**: ✅ COMPLETADO
- **Última ejecución**: ~2025-10-21 08:16 (basado en log)
- **Última modificación del script**: 2025-10-22 09:27:53
- **Log principal**: `graficas_por_clase.log` (47 KB, 21-oct 08:16)
- **Propósito**: Generar gráficas de distribución por clase MNIST.

---

## Scripts sin evidencia de ejecución reciente

Los siguientes scripts no muestran evidencia de ejecución en las últimas 2 semanas (basado en logs, outputs o timestamps):

### Análisis y visualización
- `analizar_distribuciones.py`
- `analizar_distribuciones_completo.py`
- `analizar_estadisticas_full_dataset.py`
- `analizar_imagen_individual.py`
- `analizar_metricas_por_clase.py`
- `analizar_metricas_por_clase_mac.py`
- `analizar_test_100imgs_mac.py`
- `visualizar_alpha_critico.py`
- `visualizar_clase_unica.py`
- `visualizar_puntos_criticos.py`
- `graficar_distribuciones_checkpoints_impares.py`
- `monitor_progreso.py`

### Scripts de demostración y testing
- `ejemplo_completo.py`
- `main.py`
- `test_instalacion.py`
- `test_kuramoto_MAC_100imgs.py`
- `verificar_consistencia.py`
- `verificar_scripts_alpha_c.py`

### Scripts de búsqueda de α_c
- `encontrar_alpha_critico.py` (versión genérica)
- `run_kuramoto_por_clase.py`
- `run_kuramoto_TRAIN_full.py`

### Scripts de generación de gráficas
- `generar_graficas_pdf.py`

**Nota**: La ausencia de evidencia no significa que estos scripts no funcionen, solo que no han sido ejecutados recientemente o no generaron logs/outputs en ubicaciones conocidas.

---

## Scripts del proyecto AKOrN (codigo/akorn/)

**Estado general**: No se detectaron ejecuciones recientes de scripts de entrenamiento o evaluación en `codigo/akorn/`.

### Scripts principales sin evidencia de ejecución reciente:
- `train.py` — Entrenamiento general de AKOrN
- `train_obj.py` — Entrenamiento para detección de objetos
- `train_classification.py` — Entrenamiento para clasificación
- `train_sudoku.py` — Entrenamiento en Sudoku
- `eval_obj.py` — Evaluación de objetos
- `eval_sudoku.py` — Evaluación de Sudoku
- `visualizacion/visualizar_sudoku.py`
- Scripts de descarga: `data/download_*.sh`

**Observaciones**: No se encontraron logs, archivos de salida recientes ni procesos activos relacionados con AKOrN. Posiblemente el trabajo se ha concentrado en el módulo de análisis de criticalidad.

---

## Scripts en PlanDeAccion2/

No se detectaron logs ni outputs recientes en `codigo/PlanDeAccion2/`. Este directorio parece contener código experimental o de planificación no ejecutado recientemente.

---

## Resumen de estado por categoría

| Categoría | Total scripts | En ejecución | Ejecutados (última semana) | Sin evidencia reciente |
|-----------|---------------|--------------|----------------------------|------------------------|
| Análisis criticalidad | ~30 | 1 | 4 | ~25 |
| AKOrN | ~10 principales | 0 | 0 | ~10 |
| PlanDeAccion2 | ~5 | 0 | 0 | ~5 |

---

## Recomendaciones

1. **Monitoreo del proceso activo**: 
   - El script `analizar_incremental_clases_impares.py` lleva >5 horas ejecutándose. Recomendado: monitorizar regularmente con `tail -f run_impares.log` o el script `monitor_progreso.py`.

2. **Scripts sin uso reciente**:
   - Considerar archivar o documentar scripts que no se han ejecutado en >2 semanas si ya no son necesarios.
   - Alternativamente, ejecutar `test_instalacion.py` para validar que el entorno sigue funcional.

3. **Proyecto AKOrN**:
   - Si se planea retomar trabajo en AKOrN, verificar que datasets estén descargados (`data/download_*.sh`) y ejecutar un entrenamiento de prueba.

4. **Centralización de logs**:
   - Considerar crear un directorio `logs/` para centralizar todos los `.log` y `.pid` (actualmente están dispersos en el directorio raíz del módulo).

5. **Siguiente paso sugerido**:
   - Una vez que `analizar_incremental_clases_impares.py` complete, ejecutar los scripts de análisis sugeridos en `kuramoto_train_mac.log`:
     - `analizar_distribuciones.py --dataset train_mac`
     - `analizar_estadisticas_full_dataset.py --dataset train_mac`

---

**Fin del reporte.**  
Generado automáticamente el 31-10-2025.
