# Plan de acción para el análisis de complejidad y criticidad en AKOrN

Este plan detalla los pasos necesarios para trasladar el análisis espectral y de correlaciones a largo plazo descrito por Myrov et al. a tu modelo **Artificial Kuramoto Oscillatory Neurons (AKOrN)**. La finalidad es identificar un posible régimen crítico sin asumir valores fijos de referencia, sino buscando **picos en métricas físicas** como la correlación temporal a largo plazo y la susceptibilidad. Los puntos siguientes combinan explicaciones conceptuales, referencias al artículo original y sugerencias prácticas para trabajar con el código de AKOrN.

## 1  Conceptos clave del artículo de referencia

El marco de **criticalidad cerebral** propone que el cerebro opera cerca de la transición de fase entre un estado subcrítico (desorden) y uno supercrítico (orden). En esa región emergen correlaciones espacio‑temporales sin escala, observables como **correlaciones temporales de largo alcance (LRTCs)**, avalanchas de actividad de ley de potencias y una maximización de la capacidad de procesamiento【654020391997339†L423-L430】. Myrov et al. utilizan el **análisis de fluctuación sin tendencia** (DFA) para cuantificar las LRTCs; definen la zona crítica como aquella donde el exponente DFA (\(\alpha\)) alcanza un **pico** y supera un umbral (\(\alpha>0{.}65\))【654020391997339†L904-L907】.  Las medidas de sincronía global, como el parámetro de orden \(R\), sirven para seguir la transición de fase pero **no** definen por sí solas la criticalidad【654020391997339†L423-L430】.

## 2  Preparación de las series temporales en AKOrN

### 2.1  Cargar el modelo y generar las salidas intermedias

1. **Instalar dependencias.** Asegúrate de tener un entorno Python con PyTorch y las librerías de AKOrN. Clona o descomprime el repositorio y instala los requisitos.

2. **Instanciar AKOrN.** Importa la clase y configúrala con los parámetros deseados. Por ejemplo, una sola capa de Kuramoto con ocho iteraciones:

```python
from source.models.objs.knet import AKOrN
model = AKOrN(n=4, ch=256, L=1, T=8, J='attn', gamma=1.0)
model.eval()
```

3. **Preparar entradas.** Utiliza datos reales (imágenes) para que la dinámica refleje las tareas del modelo. Para pruebas rápidas puedes emplear un tensor aleatorio de forma `[1, 3, 128, 128]`.

4. **Obtener activaciones y energías.** Llama a `model.forward` con los indicadores `return_xs=True` y `return_es=True`. Esto devuelve, además de la salida, una lista anidada `xs` con las activaciones de cada capa y una lista `es` con la energía por iteración:

```python
with torch.no_grad():
    output, xs, es = model(x_in, return_xs=True, return_es=True)
# xs[1] es la lista de activaciones de la KLayer; es[1] la energía acumulada.
```

### 2.2  Construir la serie \(R(t)\)

Cada elemento `x_t` de `xs[1]` es un tensor `[B, C, H, W]` donde los canales se agrupan en parejas que representan la parte real e imaginaria del oscilador. Para cada iteración \(t\):

1. **Reagrupar canales.** Divide los canales en grupos de tamaño `n` (por ejemplo, 4) para separar las componentes reales e imaginarias:

```python
B, C, H, W = x_t.shape
n = model.n
x_flat = x_t.view(B, C//n, n, H, W)
real = x_flat[..., 0, :, :]  # primera componente
imag = x_flat[..., 1, :, :]  # segunda componente
```

2. **Calcular las fases.** La fase de cada oscilador es `theta = torch.atan2(imag, real)`. Convierte estas fases en vectores unitarios complejos mediante `exp(i theta)`.

3. **Promediar y extraer la magnitud.** Haz la media de estos vectores sobre todos los osciladores (todas las posiciones y grupos), luego toma el valor absoluto. Ese valor es el parámetro de orden \(R_t\):

```python
complex_vecs = torch.exp(1j * theta)
R_t = complex_vecs.mean().abs().item()
```

Repite este proceso para cada iteración de la KLayer y obtendrás la serie temporal \(R(t)\). Si prefieres trabajar con energía \(E(t)\), puedes utilizar la lista `es[1]` que se construye directamente en `klayer.py` como suma de \(-\text{sim}\) en cada paso.

## 3  Análisis espectral y de DFA

### 3.1  Cálculo del espectro de potencias

1. **Transformada rápida de Fourier.** Usa la FFT o el método de Welch sobre la serie \(R(t)\) o \(E(t)\) para obtener su densidad espectral de potencia \(S(f)\).

2. **Pendiente en escala log–log.** Representa `log10(S(f))` frente a `log10(f)` y ajusta una recta en la banda de frecuencias bajas (donde la gráfica es aproximadamente lineal). La pendiente \(-\beta\) indica el balance entre fluctuaciones lentas y rápidas. Cerca de la criticalidad se espera una pendiente próxima a \(-1\) (ruido rosa), aunque en modelos como AKOrN puede variar: pendientes más negativas (por ejemplo, \(-3\) o \(-4\)) indican dinámica más rígida; pendientes cercanas a cero corresponden a ruido blanco. **No busques un valor “correcto”;** interesa comprobar si la pendiente cambia o la varianza de \(R(t)\) se maximiza al variar los parámetros.

### 3.2  Detrended Fluctuation Analysis (DFA)

1. **Segmentación y detrending.** Divide la serie \(R(t)\) en segmentos de longitud \(s\) (con \(s\) desde 4 hasta 1/10 del tamaño de la serie). Para cada segmento, elimina la tendencia local (por ejemplo, ajustando y restando una recta).

2. **Fluctuación residual.** Calcula la desviación estándar residual \(F(s)\) de cada segmento y promedia sobre todos los segmentos de tamaño \(s\).

3. **Estimación del exponente.** Repite el proceso para varios valores de \(s\). Representa `log(F(s))` frente a `log(s)` y ajusta una recta: la pendiente es el exponente \(\alpha\). En la práctica:

   - \(\alpha \approx 0{.}5\): ruido blanco (no hay LRTCs).
   - \(\alpha \approx 1\): ruido rosa (correlaciones de largo alcance). En el artículo, los picos de \(\alpha\) rondan 1 y señalan el régimen crítico【654020391997339†L904-L907】【654020391997339†L543-L544】.
   - \(\alpha > 1\): ruido marrón; dinámica muy suavizada.

4. **Interpretación física.** No establezcas un objetivo fijo (por ejemplo, \(\alpha=1\)); lo relevante es **identificar un pico** de \(\alpha\) cuando varies el acoplamiento u otros parámetros. Ese pico indica la transición entre dinamismo caótico y sincronía rígida, tal como se observa en el estudio de Myrov et al.【654020391997339†L904-L907】

## 4  Barrido de parámetros y contrastes

1. **Variar el acoplamiento y la arquitectura.** En AKOrN, ajusta el parámetro de acoplamiento `gamma`, el número de capas `L` y la topología de la conectividad (`J='conv'` o `J='attn'`). Para cada combinación, repite los pasos 2 y 3 y registra las series \(R(t)\), su PSD y el exponente DFA.

2. **Buscar picos en \(\alpha\) y en la varianza.** La **región crítica** será aquella donde \(\alpha\) o la varianza de \(R(t)\) alcanzan un máximo. Myrov et al. definen el régimen crítico como el conjunto de parámetros donde más del 10 % de los nodos tienen \(\alpha > 0{.}65\)【654020391997339†L904-L907】; puedes adaptar un criterio similar contando cuántas capas o instancias de AKOrN presentan picos significativos.

3. **Correlación cruzada entre nodos.** Además del DFA, calcula la **correlación de amplitudes** o de fases entre nodos (salidas de distintas capas o canales). En el modelo jerárquico, la correlación cruzada se maximiza en el mismo entorno de parámetros donde emergen las LRTCs【654020391997339†L914-L930】. Comparar el comportamiento de \(\alpha\) y de la correlación cruzada te dará mayor certeza de estar cerca de la criticalidad.

## 5  Interpretación y recomendaciones

* **Régimen crítico como región, no valor único.** El objetivo no es alcanzar un valor específico de \(R\), pendiente del PSD o exponente DFA, sino detectar **picos** en estas métricas al barrer parámetros. Estos picos reflejan la transición de la red entre desorden y orden y señalan un punto de máxima susceptibilidad y complejidad【654020391997339†L423-L430】.

* **Evitar umbrales rígidos.** Aunque en el artículo utilizan \(\alpha>0{.}65\) para definir la región crítica, eso responde a la dinámica específica de su modelo. En AKOrN los valores pueden ser diferentes debido a la arquitectura, el tipo de entrada o la cantidad de iteraciones. Busca siempre el máximo relativo más que un número absoluto.

* **Complementar con otras métricas.** La pendencia del espectro, el DFA y la correlación cruzada son **proxies** de la criticalidad. Considera también otras medidas de red (agrupamiento, longitud de camino) y dinámicas (distribución de avalanchas) para tener una visión completa.

## 6  Resumen operativo

1. **Preparar el entorno** con PyTorch y cargar AKOrN.
2. **Ejecutar el modelo** sobre datos reales y recoger las series \(R(t)\) y \(E(t)\) usando `return_xs` y `return_es`.
3. **Calcular el PSD** de esas series y estimar la pendiente en log–log; registrar el valor.
4. **Aplicar DFA** sobre las mismas series para obtener el exponente \(\alpha\).
5. **Repetir** para distintas configuraciones de acoplamiento (`gamma`), número de capas (`L`) y tipos de conectividad (`conv`, `attn`).
6. **Identificar picos** en \(\alpha\) y en la varianza de \(R(t)\); cruzarlos con la correlación entre nodos.
7. **Extraer conclusiones**: la región donde coinciden los picos de \(\alpha\), de varianza y de correlación cruzada indica un comportamiento crítico; compara este comportamiento con el desempeño del modelo para evaluar si operar cerca del borde del caos mejora la tarea.

---

Este plan sintetiza los pasos necesarios para reproducir y aplicar el análisis de Myrov et al. a tu red AKOrN. Al seguirlo, podrás generar series temporales de tus capas, calcular sus exponenetes de fluctuación y espectros, y localizar posibles regímenes críticos de forma físicamente fundamentada.
