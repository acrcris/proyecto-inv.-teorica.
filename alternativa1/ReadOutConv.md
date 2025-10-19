**Resumen**
- Propósito: proyectar un mapa de características `x` a un nuevo sesgo `c` por grupo, midiendo la norma por cada uno de los `ro_N` componentes y sumando un sesgo entrenable por canal.
- Entrada/Salida: entrada `[B,inch,H,W]`, salida `[B,outch,H,W]`.

**Ubicación**
- `akorn/source/layers/common_layers.py:64`

**Constructor**
- `inch`: canales de entrada.
- `outch`: canales de salida (número de grupos tras la proyección).
- `ro_N`: factor de “expansión” por grupo antes de tomar la norma (divide el canal expandido en `outch × ro_N`).
- `kernel_size`, `stride`, `padding`: parámetros de la conv interna (`invconv`).

**Forward**
- `invconv(x)` produce `[B, outch*ro_N, H, W]`.
- `unflatten(1,(outch,-1))` agrupa en `[B,outch,ro_N,H,W]` y aplica `torch.linalg.norm` en dim `ro_N`.
- Suma `bias` por canal: `+ bias[None,:,None,None]`.

**Uso típico**
- Dentro de `nn.Sequential` en un bloque readout para actualizar el `c` que condiciona al siguiente `KLayer`.

