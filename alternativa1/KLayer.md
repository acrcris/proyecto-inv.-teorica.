**Resumen**
- Propósito: capa de dinámica tipo Kuramoto que itera un estado `x` condicionado por un sesgo `c` y una conectividad `J` (conv o atención), manteniendo cada grupo de `n` canales sobre la esfera mediante normalización.
- Entrada: `x,c` con forma `[B,C,H,W]` o tokens `[B,T,C]` (según modo).
- Salida: lista `xs` de estados por paso y `es` de energías/similaridad negativa.

**Ubicación**
- `akorn/source/layers/klayer.py:61`

**Constructor**
- `n`: tamaño de grupo (debe dividir a `ch`).
- `ch`: canales totales.
- `J`: `"conv"` o `"attn"`; define `self.connectivity`.
- `c_norm`: `"gn"`, `"sandb"` o `None`; normaliza el sesgo `c`.
- `use_omega`, `init_omg`, `global_omg`, `learn_omg`: controlan `OmegaLayer` (rotación en el espacio de fase).
- `ksize`: kernel si `J="conv"`.
- `gta`, `hw`, `heads`: parámetros para atención con transformaciones geométricas (`Attention`).
- `apply_proj`: si proyecta la actualización al espacio tangente.

**Componentes internos**
- `connectivity`: `nn.Conv2d(ch,ch,ksize,...)` o `Attention(ch, heads, ...)`.
- `c_norm`: `GroupNorm`, `ScaleAndBias` o identidad. `akorn/source/layers/common_layers.py:193`
- `omg`: `OmegaLayer` o identidad (rotación). `akorn/source/layers/klayer.py:14`

**Algoritmo**
- `kupdate(x,c)`: calcula `y = J(x) + c`, aplica `OmegaLayer(x)` si procede, reagrupa en `n`, proyecta al tangente (si `apply_proj`), retorna `dxdt` y `sim` (similitud con el estado).
- `forward(x,c,T,gamma)`: normaliza `c` y `x`, itera `T` veces `x ← normalize(x + gamma*dxdt)`, acumula `xs` y energía `es = -sim` agregada.

**Formas/tensores**
- Si imagen: `[B,C,H,W]`, con `C % n == 0`.
- Reagrupación usa `kutils.reshape/reshape_back`. `akorn/source/layers/kutils.py:10`

**Consideraciones**
- `gamma` controla estabilidad; valores altos pueden oscilar/divergir si `use_omega` está activo.
- `apply_proj=False` desactiva la proyección, dejando la dinámica menos restringida.
- `gta/hw` deben ajustarse a la resolución espacial efectiva.

