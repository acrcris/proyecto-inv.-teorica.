**Resumen**
- Propósito: bloque secuencial Normalización → Activación → Conv2d configurable, útil como pieza básica en readouts y MLPs espaciales.
- Entrada/Salida: `[B,inch,H,W]` → `[B,outch,H,W]`.

**Ubicación**
- `akorn/source/layers/common_layers.py:92`

**Constructor**
- `inch`, `outch`: canales de entrada/salida.
- `kernel_size`, `stride`, `padding`: parámetros de la convolución.
- `norm`: `"gn"`, `"bn"` o `None` (identidad).
- `act`: activación (por defecto `nn.ReLU()`).

**Forward**
- Aplica `norm(inch)` → `act` → `Conv2d(inch→outch)` sobre el tensor de entrada.

**Notas**
- Si `norm=None`, se usa identidad (solo `act` y `conv`).
- Usado tanto dentro de `FF` como al final del readout de Sudoku cuando `nl=True`.

