# Cálculo de α_c — Explicación detallada

Te explico paso a paso y con los detalles del código cómo estamos calculando α_c en el repositorio (resumen + matemáticas + implementación concreta y recomendaciones).

## Resumen breve

- Para cada imagen (o configuración) se hace un barrido (grid) de valores de α.
- Para cada α se escala la matriz de acoplamientos c → α·c y se ejecuta la dinámica Kuramoto desde una condición inicial aleatoria durante T pasos.
- Se calcula el parámetro de orden R(t) a cada paso a partir de las fases (extraídas de dos canales).
- Para cada α se toma la media de los últimos W pasos de R(t) → se obtiene una curva order_curve(α).
- Se estima α_c como el α donde la pendiente dR/dα es máxima (índice del máximo gradiente).
- Repetimos por imagen, y luego promediamos α_c por clase (media, σ).

## Matemática (lo que calcula el código)

- Fases: para cada tiempo t se calcula la fase por oscilador
  φ_j(t) = atan2(b, a)
  usando dos canales a,b (ver `KuramotoMetrics.order_parameter`).

- Parámetro de orden de Kuramoto (por tiempo):

  $$R(t) = \left|\frac{1}{N} \sum_j e^{i \phi_j(t)}\right|\,, $$

  que el código implementa como la magnitud del promedio complejo de exp(i·fase).

- Para cada α del grid se obtiene R(t; α). Luego:

  R_tail(α) = mean( R(t; α) over last W timesteps ).

  order_curve(α) := R_tail(α).

- α_c se define por el código como:

  α_c = α_{idx} donde idx = argmax( d order_curve / dα ).

  Es decir, se calcula `gradient = np.gradient(order_curve, alphas)` y se toma el α con gradiente máximo.

## Implementación exacta (archivo y funciones relevantes)

- Archivo principal: `encontrar_alpha_critico.py`.
  - Grid de α: función `_generate_alphas(start,end,step)`.
  - Para cada α: `_estimate_alpha_curve(...)`:
    - Escala `c_base` → `c_scaled = c_base * α`.
    - `x0` = ruido normal (condición inicial aleatoria).
    - Ejecuta `kblock(x0, c_scaled, T=timesteps, gamma=..., del_t=...)` y recoge `xs` (lista de estados temporales).
    - Calcula `r_series = KuramotoMetrics.order_parameter(xs)` (ver abajo).
    - Toma `tail = últimos window pasos` (o toda la serie si es más corta) y guarda `mean(tail)`.
  - Cálculo de α_c: `_critical_alpha(order_curve, alphas)` calcula `gradient = np.gradient(order_curve, alphas)` y retorna α en el `argmax` del gradiente.
  - Agrega α_c por imagen en `per_class_alphas[label]`. Al final devuelve por clase: lista de α_c, media, desviación estándar y curvas medias de R(α).

- Extracción de fases y R(t): `analisis/criticalidad.py` → `KuramotoMetrics.order_parameter(xs)`:
  - Para cada `x_t` en `xs` extrae canales `a,b`:
    `phase = arctan2(b, a)`
    `z = exp( i * phase )`
    `R(t) = |mean(z)|`

## Parámetros que influyen y dónde están

- `--alpha-start`, `--alpha-end`, `--alpha-step` (grid de α).
- `--timesteps` T (duración de la simulación por α).
- `--window` W (número de pasos finales de R(t) promediados para cada α).
- `gamma`, `delta_t` (parámetros dinámicos del bloque `KBlock`).
- `max_images_per_class` (cuántas imágenes por clase se usan para promediar).

## Intuición y por qué se usa el gradiente

- En una transición sincrónica, R(α) pasa de valores bajos a valores altos alrededor del punto crítico. La subida más pronunciada suele estar cerca de la transición, por eso se toma el α con mayor pendiente (máximo gradiente) como estimador del punto crítico.
- Ventajas: simple y robusto si R(α) es monótono y la transición es pronunciada.
- Limitaciones: ruido en R(α) (por condiciones iniciales y finite-size) puede desplazar el máximo. También si R(α) sube muy suavemente o tiene plateaus, el máximo gradiente puede ser ruidoso.

## Prácticas usadas en el código para mitigar ruido

- Para cada α se promedian los últimos W pasos de R(t) (reduce fluctuaciones temporales).
- Para cada clase se calcula α_c para múltiples imágenes y se reporta media y σ.
- Inicializaciones aleatorias `x0` por α (muestreo de variaciones).

## Posibles mejoras / alternativas (si quieres más precisión)

- Interpolación: interpolar `order_curve(α)` (p. ej. spline) y localizar el máximo de la derivada continua (mejor resolución que el paso de la grilla).
- Ajuste sigmoide: ajustar `R(α)` con una función sigmoide y tomar el punto de inflexión (más robusto frente a ruido).
- Criterio por umbral: definir α_c como el primer α donde `R_tail(α) > R_threshold` (ej. 0.1 o un porcentaje del rango).
- Bootstrap: recalcular α_c con remuestreo para estimar incertidumbre robusta.
- Aumentar `timesteps` y/o `window` para mayor convergencia de R(t) por α.
- Promediar resultados con varias condiciones iniciales `x0` por mismo α (en lugar de una sola) para reducir sensibilidad a initial conditions.

## Dónde ver los resultados/visualizaciones ya hechas

- Resultados por clase se guardan en `results` dentro del JSON que produce `encontrar_alpha_critico.py` si pasas `--output`.
- Visualización agrupada: `visualizar_alpha_critico.py` (hace bar plots de medias, boxplots y histograma global).
- Visualización de puntos críticos individuales y su R(α): `visualizar_puntos_criticos.py` (usa `np.interp` para encontrar R en `α_c`).

## Demostración opcional

Si quieres que lo demuestre, puedo ejecutar una corrida corta (ej.: `alpha_start=0`, `alpha_end=0.02`, `alpha_step=0.0005`, `timesteps=100`, `window=10`) sobre 1 o 2 imágenes para generar una gráfica explicativa de `order_curve(α)` y señalar `α_c`.

---

*Archivo generado automáticamente: `alpha_c_explicacion.md` (31-10-2025).*