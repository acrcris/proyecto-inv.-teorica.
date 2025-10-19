# Objetivos


## Resolver sudokus con redes neuronales.

## Entender el funcionamiento codigo.

## Caracterizar la calidad de los dataset de entrenamiento para sudoku.

Dataset characteristics
### Diversidad y tamaño
A dataset is characterized by its size and diversity. Size indicates the number of examples. Diversity indicates the range those examples cover. Good datasets are both large and highly diverse. 

A dataset can also be characterized by the number of its features. For example, some weather datasets might contain hundreds of features, ranging from satellite imagery to cloud coverage values. Other datasets might contain only three or four features, like humidity, atmospheric pressure, and temperature. Datasets with more features can help a model discover additional patterns and make better predictions. However, datasets with more features don't always produce models that make better predictions because some features might have no causal relationship to the label.

(https://developers.google.com/machine-learning/intro-to-ml/supervised?utm_source=chatgpt.com)

### Etiquetado

Los datos etiquetados de alta calidad mejoran la precisión del modelo, ya que proporcionan señales de aprendizaje claras y coherentes. 

El etiquetado de datos también sirve para mitigar los sesgos, ya que garantiza la representación y el equilibrio de los conjuntos de datos para evitar que los modelos hereden los sesgos. Además, los datos etiquetados facilitan el procesamiento y el análisis de datos de forma automática, lo que permite a las máquinas gestionar y extraer información valiosa de grandes cantidades de datos de forma eficiente, lo que ahorra tiempo y esfuerzo en comparación con los métodos manuales.

(https://cloud.google.com/use-cases/data-labeling?utm_source=chatgpt.com)

### Como se diseñaron los datasets?

https://arxiv.org/abs/1803.09010?utm_source=chatgpt.com
https://arxiv.org/html/2406.19614v1?utm_source=chatgpt.com

## Compara el desempeño de la redes AKOrN con otros modelos.

[Solving Sudoku Using Oscillatory Neural Networks](2508.02250v1.pdf)

[Sudoku-Bench: Evaluating creative reasoning with Sudoku variants](2505.16135v1.pdf)

[ARTIFICIAL KURAMOTO OSCILLATORY NEURONS](Kuramoto.pdf)

## Proyecto: Evaluar y explicar AKOrN en Sudoku (12 semanas)

### Objetivo general
- Evaluar y explicar la capacidad de las redes AKOrN para resolver Sudoku, comparándolas con ItrSA y Transformers, analizando su dinámica tipo Kuramoto y su generalización ID/OOD.

### Definiciones (ID/OOD)
- ID (in-distribution): pruebas del mismo generador/distribución que el entrenamiento (SATNet test; `--data=id` según `alternativa1/codigo/akorn/scripts/sudoku.md`).
- OOD (out-of-distribution): pruebas de un generador/distribución distinta (RRN hard; `--data=ood`).

### Preguntas guía
- ¿Qué factores de AKOrN (T, K, L, ch, J, omega) impulsan la generalización OOD?
- ¿Cómo se relaciona la energía/sincronía de la dinámica con la satisfacción de restricciones del Sudoku?
- ¿Qué diferencias de diseño entre datasets ID (SATNet) y OOD (RRN) explican brechas de desempeño?

### Objetivos específicos (SMART)
- Reproducibilidad: Entrenar y evaluar AKOrN en ID/OOD siguiendo `alternativa1/codigo/akorn/scripts/sudoku.md`, alcanzando en 5 semanas ≥99% ID y ≥70% OOD con test-time extension (T=128) y energy voting (K≈100). Evidencia: tabla de resultados con medias/DE y seeds fijos.
- Comparativa: Entrenar ItrSA y Transformer con los comandos del mismo script y reportar la misma tabla comparativa. Plazo: semana 6.
- Ablaciones: Medir impacto de T∈{32,64,128}, K∈{0,32,100,512}, L∈{1,2}, ch∈{128,256,512}, y J∈{attn,conv} sobre solved-rate, coste y robustez OOD. Plazo: semana 9.
- Dinámica/energía: Usar trayectorias/energías del modelo (ver `alternativa1/SudokuAKOrN.md`) y el visor en `alternativa1/codigo/akorn/visualizacion/README.md` para correlacionar E(t), sincronía y constraint violations. Plazo: semana 8.
- Dataset: Caracterizar ID (SATNet) y OOD (RRN) en tamaño, diversidad y dificultad (pistas dadas, distribución de clues), y relacionarlo con desempeño. Plazo: semana 10.
- Informe y paquete reproducible: Reporte final con figuras, scripts de corrida y guía README para replicar tablas y figuras. Plazo: semana 12.

### Metodología y alcance
- Código base: Arquitectura en `alternativa1/SudokuAKOrN.md`. Entrenamiento/evaluación y datasets en `alternativa1/codigo/akorn/scripts/sudoku.md`. Visualizaciones en `alternativa1/codigo/akorn/visualizacion/README.md`.
- Datasets: Descargar SATNet (ID) y RRN (OOD) según el script; documentar tamaño y splits. Con cómputo limitado, reducir `epochs`, `ch` y `L`, y compensar con `T`/`K` en inferencia.
- Comparativas: Usar exactamente los flags de cada baseline; registrar seeds, tiempos e hiperparámetros.
- Teoría: Resumir vínculo Kuramoto–energía–convergencia y cómo `KLayer` implementa actualizaciones iterativas; conectar hallazgos empíricos con teoría.

### Cronograma (12 semanas)
- Semana 1: Setup, lectura base (docs y PDFs), descarga de datos; dry-run de entrenamiento/eval.
- Semanas 2–3: Entrenar AKOrN (config base), evaluar ID/OOD, primera tabla de resultados.
- Semanas 4–5: Refinar AKOrN para metas SMART de reproducibilidad; documentar configuración final.
- Semana 6: Entrenar/evaluar ItrSA y Transformer; completar tabla comparativa.
- Semanas 7–8: Visualización de dinámica y análisis de energía; ablaciones de T y K.
- Semana 9: Ablaciones de arquitectura (L, ch, J, omega/nl); curvas performance–costo.
- Semana 10: Caracterización de datasets y análisis dificultad→desempeño.
- Semana 11: Integración de resultados, figuras finales, verificación de reproducibilidad.
- Semana 12: Redacción final e informe + anexos de scripts.

### Métricas y criterios
- Solved rate por puzzle (%), en ID y OOD.
- Violaciones de restricciones promedio por puzzle y cell-accuracy.
- Coste: tiempo por puzzle y memoria; sensibilidad a T/K.
- Reproducibilidad: varianza entre seeds; configuración exacta en logs.
- Éxito del proyecto: tabla comparativa completa; ablation plots; correlaciones E(t)↔restricciones; guía reproducible.

### Baselines y controles
- Modelos: AKOrN, ItrSA, Transformer (scripts incluidos en `alternativa1/codigo/akorn/scripts/sudoku.md`).
- Solver simbólico opcional: backtracking/CP como control de exactitud/tiempo.
- Control de cómputo: si no hay GPU, usar `ch=128`, `L=1`, `epochs` reducidos, y mayor `T/K` en test-time.

### Riesgos y mitigación
- Cómputo insuficiente: bajar `ch/L/epochs`; usar `T/K` en inferencia; muestrear subsets equilibrados.
- Variabilidad OOD: fijar 3–5 seeds; reportar medias±DE.
- Datos/red: documentar mirrors y checksums; cachear descargas.

### Entregables
- Informe (10–15 págs) con tabla comparativa, ablation plots y análisis de dinámica.
- Carpeta de scripts para reproducir: train/eval/plots, con README y seeds.
- Visualizaciones de trayectorias de osciladores y ejemplos resueltos/fallidos.

