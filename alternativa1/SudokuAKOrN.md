**Resumen**
- Propósito: modelo para resolver Sudoku que combina embeddings de dígitos con bloques de dinámica tipo Kuramoto (`KLayer`) y proyecciones `ReadOutConv` para refinar el estado.
- Entrada: tableros 9x9 en one‑hot (`[B,9,9,9]`) y máscara `is_input` (`[B,9,9,1]`).
- Salida: logits por celda para dígitos 1–9 (`[B,9,9,9]`).
- Relación directa: usa `KLayer`, `ReadOutConv`, `ResBlock(FF)`, `BNReLUConv2d`.

**Ubicación**
- `akorn/source/models/sudoku/knet.py:18`

**Constructor**
- `n`: tamaño del problema (para Sudoku, 9). Guía particiones en `KLayer` y canales de `ReadOutConv`.
- `ch`: ancho de embedding interno (canales de características).
- `L`: número de bloques apilados `[KLayer → ReadOut]`.
- `T`: pasos de iteración por `KLayer`.
- `gamma`: paso/ganancia de integración (parámetro entrenable `nn.Parameter([gamma])`).
- `J`: tipo de conectividad en `KLayer` (p.ej., `"attn"`).
- `use_omega`, `global_omg`, `init_omg`, `learn_omg`: controlan `OmegaLayer` dentro de `KLayer`.
- `nl`: activa bloque `ResBlock(FF)` y `BNReLUConv2d` en el readout.
- `heads`: número de cabezas si `J="attn"`.

**Arquitectura**
- Embedding de dígitos: `nn.Embedding(10, ch)` mapea 0–9 a $\mathbb R^{ch}$ y se reordena a `[B,C,H,W]`.
- Bloques apilados (`L` veces):
  - `KLayer(n,ch,J,...)`: evoluciona `x` durante `T` pasos condicionado por `c` y `gamma`.
  - `readout = nn.Sequential(ReadOutConv, ResBlock(FF)?, BNReLUConv2d?)`: actualiza `c` del siguiente bloque.
- Cabezal final: `nn.ReLU() → nn.Conv2d(ch, 9, 1)` y `permute` a `[B,9,9,9]`.

**Flujo de datos**
- `feature(inp,is_input)`: convierte one‑hot a enteros, embebe a `c`, inicializa `x` con ruido en celdas vacías, itera los pares `(KLayer, readout)` acumulando trayectorias `xs`, energías `es` y actualiza `c`.
- `forward(c,is_input,...)`: llama `feature`, aplica `self.out`, y expone opcionalmente `xs` y `es`.

**Formas/tensores**
- `inp`: `[B,9,9,9]` one‑hot; `is_input`: `[B,9,9,1]`.
- `c,x`: `[B,ch,9,9]` durante el cómputo; salida logits `[B,9,9,9]`.

**Convenciones de dimensiones**
- `[B,C,H,W]` (NCHW en PyTorch):
  - `B`: tamaño de batch (número de tableros procesados en paralelo).
  - `C`: canales (dimensión de características o embedding).
  - `H`/`W`: alto y ancho espaciales (para Sudoku, 9×9).
- En este modelo se usan dos convenciones según el paso:
  - Entrada/salida de Sudoku: canal al final `[B,H,W,9]` para comodidad con one‑hot de dígitos.
  - Cómputo interno de la red: canal primero `[B,C,H,W]` (PyTorch por defecto). Se convierte con `permute` entre ambas.

**Codificación De Entrada (one‑hot e is_input)**
- One‑hot `[B,9,9,9]`:
  - El último eje de tamaño 9 representa las clases de dígitos 1–9.
  - En una celda dada con dígito d, el vector es todo ceros salvo un 1 en el índice `d-1`.
  - En celdas vacías, el vector es todo ceros (no hay 1s).
  - Se convierte a enteros con `convert_onehot_to_int`: produce un tablero `[B,9,9]` con valores `0` para vacías y `1..9` para celdas dadas. Esto permite usar `nn.Embedding(10, ch)` (incluye índice 0 para “vacío”).
- Máscara `is_input` `[B,9,9,1]`:
  - Indicador binario (0/1) de si la celda venía pre‑rellenada en el puzzle.
  - Se obtiene como la suma a lo largo del eje de clases del one‑hot (presencia de algún 1 ⇒ 1, si no ⇒ 0).
  - Uso en la inicialización del estado: `x0 = is_input * c + (1 - is_input) * n`, donde `c` es el embedding del dígito y `n` es ruido gaussiano. Así, las celdas dadas empiezan ancladas a su embedding y las no dadas empiezan desde ruido.
  - Esta máscara puede reutilizarse en evaluación o pérdidas si se desea tratar de forma distinta celdas dadas vs. por resolver (aunque el forward del modelo la usa solo para inicialización de `x`).


**Dependencias**
- `KLayer`: dinámica iterativa condicionada por `c` (Kuramoto). `akorn/source/layers/klayer.py:61`
- `ReadOutConv`: proyección de norma por grupo para actualizar `c`. `akorn/source/layers/common_layers.py:64`
- `ResBlock`: envoltura residual para `FF`. `akorn/source/layers/common_layers.py:37`
- `FF`: bloque feed‑forward 1x1 (2 capas con BNReLUConv2d). `akorn/source/layers/common_layers.py:132`
- `BNReLUConv2d`: normaliza→activa→conv 1x1. `akorn/source/layers/common_layers.py:92`

**Detalles clave**
- `gamma` es `nn.Parameter` y afecta la estabilidad/velocidad de convergencia de `KLayer`.
- `fixed_noise` permite reproducibilidad en la inicialización de `x`.
- `nl=False` elimina no‑linealidades/normalización en el readout para una ruta casi lineal.

**Uso mínimo**
- Ejemplo: inicializa con `n=9, ch=64, L=2, T=16`, pasa `X` y `is_input` (one‑hot/máscara) y recibe logits `[B,9,9,9]`.
