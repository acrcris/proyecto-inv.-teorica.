# kenet.py


# knet.py — Notas legibles y trazables

Estas notas explican el archivo **knet.py** (clase `SudokuAKOrN`) conectándolo con el artículo de AKOrN. Incluye ejemplos, código y fórmulas en $\LaTeX$, además de citas simples para poder verificar cada afirmación en el PDF adjunto.

---

## 1) ¿Qué son los *embeddings*?

**Idea intuitiva.** Un *embedding* es un diccionario entrenable que asigna cada símbolo a un vector denso. Si el símbolo es el dígito `7`, el *embedding* devuelve un vector en $\mathbb{R}^{ch}$ que captura relaciones útiles (p. ej., que 7 y 8 pueden ser “cercanos” en cierto contexto).

**Definición precisa en RNAs.** Un *embedding* es una **tabla de parámetros** $E\in \mathbb{R}^{V\times ch}$ que implementa una funciónx=
$$
E:\{0,1,\dots,V-1\}\to \mathbb{R}^{ch},\quad i\mapsto E(i).
$$
En PyTorch se implementa con `nn.Embedding(V, ch)` y se entrena por gradiente como cualquier otra capa.

**¿Qué es $V$?** $V$ es el **tamaño del vocabulario** (número de símbolos distintos) que la tabla de *embeddings* puede indexar. Formalmente, la capa aprende una matriz $E\in\mathbb{R}^{V\times ch}$ cuya fila $i$ es el vector del símbolo $i$ (índices válidos $0,\dots,V-1$).

- **En Sudoku:** $V=10$ porque usamos dígitos $\{0,\dots,9\}$ (convención: 0 = casilla en blanco).
- **En PLN:** $V$ es el número de *tokens* del vocabulario (palabras/subpalabras), más posibles *tokens* especiales (p. ej., `<pad>`, `<unk>`).

**Forma esperada:** si `inp` es un tensor entero de forma `[B, H, W]` con valores en `[0, V-1]`, entonces `embedding(inp)` produce `[B, H, W, ch]` (y luego puedes permutar a `[B, ch, H, W]` para tratarlo como mapa de características).

**Contexto AKOrN (artículo).** El bloque AKOrN parte de una observación y construye un **estímulo condicional inicial** $C$; una forma práctica de hacerlo es usar *embeddings* de los símbolos de entrada. El paper describe claramente: “primero transformamos la observación con una función relativamente simple para crear la **C** inicial; **X** se inicializa con C, un *embedding* fijo aprendido, vectores aleatorios, o una mezcla de estos; luego el bloque aplica un **Kuramoto layer** y un **readout** para procesar el par $\{X,C\}$.” fileciteturn3file0L99-L106 _(Miyato et al., “Artificial Kuramoto Oscillatory Neurons (AKOrN)”, Sec. 4, p. 3)_

En el **caso Sudoku** del artículo, **C** se inicializa **con *embeddings* de los dígitos 0–9** (usando 0 para casillas en blanco); además, si una casilla tiene dígito dado, $x_i$ toma $c_i$; si está en blanco, $x_i$ se inicializa al azar sobre la esfera. Durante el entrenamiento usan $T{=}16$ pasos de Kuramoto. fileciteturn3file2L43-L53 _(Sec. 6.2 “Solving Sudoku”, p. 8; Miyato et al., “Artificial Kuramoto Oscillatory Neurons (AKOrN)”)_

**Ejemplo mínimo (código):**
```python
import torch.nn as nn
ch = 64
embedding = nn.Embedding(10, ch)  # 0..9 -> R^ch
# Ejemplo: id 7 -> vector en R^ch
vec7 = embedding(torch.tensor(7))  # shape: [ch]
```

---

## 2) ¿Qué rol juegan los *embeddings* en `knet.py` y qué instrucciones los tocan?

En tu clase:
```python
self.embedding = nn.Embedding(10, ch)
```
Define la tabla **0–9 $\to$ $\mathbb{R}^{ch}$**.

En `feature(...)` (pseudofragmento fiel al flujo real):
```python
inp = convert_onehot_to_int(inp)          # [B,9,9,9] one-hot -> enteros 0..9
c = self.embedding(inp).permute(0,3,1,2)  # C: [B, ch, 9, 9]
```
Con esto construyes **C**, el estímulo condicional que alimenta a **KLayer**. Esto está en línea con el bloque AKOrN del paper: **Kuramoto layer** actualiza $X$ usando $C$ y el **readout** extrae rasgos de $X$ para crear una nueva $C$ del siguiente bloque. fileciteturn3file0L103-L106 _(Miyato et al., “Artificial Kuramoto Oscillatory Neurons (AKOrN)”, Sec. 4, p. 3)_

En Sudoku (artículo), **C** se inicializa exactamente con los *embeddings* 0–9 y **X** usa $c_i$ en celdas dadas (o aleatorio en las vacías) antes de ejecutar $T$ pasos. fileciteturn3file2L50-L53 _(Sec. 6.2 “Solving Sudoku”, p. 8; Miyato et al., “Artificial Kuramoto Oscillatory Neurons (AKOrN)”)_

---

## 3) ¿Para qué sirve `nn.ModuleList()` aquí?

`nn.ModuleList()` es una **lista de submódulos** registrada por PyTorch. A diferencia de una lista de Python cruda, sus elementos:
- quedan **rastreables** (sus parámetros se entrenan y guardan en el `state_dict`),
- se **serializan** correctamente,
- permiten **apilar** arquitecturas repetitivas.

En tu código:
```python
self.layers = nn.ModuleList()
for l in range(self.L):
    self.layers.append(nn.ModuleList([
        KLayer(...),
        nn.Sequential(ReadOutConv(...), ..., ...)
    ]))
```
Con esto construyes **L bloques** $[\texttt{KLayer} \Rightarrow \texttt{ReadOut}]$**, exactamente el patrón del bloque AKOrN (Kuramoto + readout). fileciteturn3file0L93-L106 _(Fig. 2, p. 3; Miyato et al., “Artificial Kuramoto Oscillatory Neurons (AKOrN)”)_

---

## 4) Flujo de la clase `SudokuAKOrN` y relación con el artículo

### 4.1 Estructura de alto nivel

Cada bloque procesa $\{X^{(l)}, C^{(l)}\}$:
- **Kuramoto layer** aplica **T** pasos de actualización a $X$ usando $C$.
- **Readout** toma $X^{(l,T)}$ y produce la nueva $C^{(l+1)}$.

Tras **L** bloques, **$C^{(L)}$** se usa para la **predicción final**. Esto se ilustra en la Fig. 2 del paper (“**C(L) is used to make the final prediction**”). fileciteturn3file0L93-L95 _(Fig. 2, p. 3; Miyato et al., “Artificial Kuramoto Oscillatory Neurons (AKOrN)”)_

En código (esquema):
```python
x, c = x0, c0  # x0: estado inicial, c0: de embeddings
for (klayer, readout) in self.layers:
    xs, es = klayer(x, c, self.T, self.gamma)  # T pasos de Kuramoto
    x = xs[-1]                                 # X^{(l,T)}
    c = readout(x)                             # C^{(l+1)}
logits = self.out(c)                            # 9 logits por celda
logits = logits.permute(0,2,3,1)                # [B,9,9,9]
```

### 4.2 Actualizaciones de Kuramoto (fórmulas)

El paper implementa la dinámica con la discretización (4)–(5):
$$
\Delta x^{(l,t)}_i = \Omega_i\, x^{(l,t)}_i + \operatorname{Proj}_{x^{(l,t)}_i}\Big( c^{(l)}_i + \sum_j J_{ij}\, x^{(l,t)}_j \Big),\tag{4}
$$
$$
 x^{(l,t+1)}_i = \Pi\big[ x^{(l,t)}_i + \gamma\, \Delta x^{(l,t)}_i \big],\qquad \Pi(x)=\tfrac{x}{\lVert x\rVert_2},\tag{5}
$$
con **$\gamma>0$** aprendible (tamaño de paso). Se ejecutan **T** pasos y se pasa $X^{(l,T)}$ al siguiente bloque. fileciteturn3file0L108-L151 _(Eqs. (4)–(5), Sec. 4, pp. 3–4; Miyato et al., “Artificial Kuramoto Oscillatory Neurons (AKOrN)”)_

### 4.3 Caso Sudoku del artículo

- **Inicialización de C** con *embeddings* de dígitos **0–9**; 0 representa blanco. 
- **Inicialización de X**: $x_i=c_i$ si hay dígito dado; si no, $x_i$ es aleatorio uniforme sobre la esfera.
- **T** durante entrenamiento: $T=16$ pasos de Kuramoto. 
Todo esto se indica explícitamente en la sección 6.2 del paper. fileciteturn3file2L43-L53 _(Sec. 6.2 “Solving Sudoku”, p. 8; Miyato et al., “Artificial Kuramoto Oscillatory Neurons (AKOrN)”)_

### 4.4 Cómo se refleja en tu implementación

- **Embeddings**:
```python
self.embedding = nn.Embedding(10, ch)
# en feature(...)
c = self.embedding(inp).permute(0,3,1,2)  # [B, ch, 9, 9]
```
- **Bloques AKOrN (L veces)**:
```python
self.layers = nn.ModuleList([ ... ])  # pares (KLayer, ReadOut)
```
- **Dinámica Kuramoto + Readout** (por bloque):
```python
xs, es = klayer(x, c, self.T, self.gamma)  # ecuaciones (4)-(5)
x = xs[-1]
c = readout(x)
```
- **Salida final** (9 clases por celda 9×9):
```python
logits = self.out(c)              # Conv1x1 -> 9 canales
logits = logits.permute(0,2,3,1)  # [B,9,9,9]
```

> **Recordatorio de trazabilidad:** La estructura “Kuramoto + readout, usar C(L) para predecir” se muestra en la Fig. 2 y el texto adyacente del artículo. fileciteturn3file0L93-L106 _(Fig. 2, p. 3; Miyato et al., “Artificial Kuramoto Oscillatory Neurons (AKOrN)”)_

---

## 5) Resumen operativo para leer el código

1. **Construye C** desde los *embeddings* de los dígitos (0–9). fileciteturn3file2L50-L53 _(Sec. 6.2 “Solving Sudoku”, p. 8; Miyato et al., “Artificial Kuramoto Oscillatory Neurons (AKOrN)”)_
2. **Inicializa X** (dado: $x_i=c_i$; blanco: aleatorio en la esfera). fileciteturn3file2L50-L53 _(Sec. 6.2 “Solving Sudoku”, p. 8; Miyato et al., “Artificial Kuramoto Oscillatory Neurons (AKOrN)”)_
3. **Itera L bloques**: en cada uno, aplica $T$ **pasos Kuramoto** y luego un **readout** para obtener la nueva C. fileciteturn3file0L108-L154 _(Sec. 4, pp. 3–4; Miyato et al., “Artificial Kuramoto Oscillatory Neurons (AKOrN)”)_
4. **Predice** con **C(L)** (cabezal conv 1×1 → 9 logits por celda). fileciteturn3file0L93-L95 _(Fig. 2, p. 3; Miyato et al., “Artificial Kuramoto Oscillatory Neurons (AKOrN)”)_

---

### Referencias del PDF
- Bloque AKOrN (Fig. 2), uso de C(L) y descripción de los módulos: fileciteturn3file0L93-L106.
- Discretización y normalización (ecuaciones 4–5), $\gamma$ aprendible y pasos T: fileciteturn3file0L108-L151.
- Sudoku: inicialización con *embeddings* 0–9, política para $x_i$ y $T=16$: fileciteturn3file2L43-L53.
- **Artículo:** *Artificial Kuramoto Oscillatory Neurons (AKOrN)* — Takeru Miyato, Sindy Löwe, Andreas Geiger, Max Welling.
# 
# ---
# ## Código para el reemplazo al final de la celda (Sage)
#
# Pega esto **al final de la celda** que imprime los conjuntos generadores minimales de $D_4$, para que en la salida se muestre **`f`** en lugar de **`s`**:
#
# ```python
# # helper para imprimir usando 'f' (reflexión) en vez de 's'
# def as_rf(x): 
#     return str(x).replace('s', 'f')
#
# print("Cantidad:", len(minimal_sets))
# for T in minimal_sets:
#     print("{ " + ", ".join(as_rf(x) for x in T) + " }")
# ```