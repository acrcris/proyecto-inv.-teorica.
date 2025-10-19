**Resumen**
- Propósito: bloque feed‑forward de dos capas 1x1 con normalización/activación intermedias (`BNReLUConv2d`), similar al MLP de Transformers pero en dominio espacial.
- Entrada/Salida: entrada `[B,inch,H,W]`, salida `[B,outch,H,W]`.

**Ubicación**
- `akorn/source/layers/common_layers.py:132`

**Constructor**
- `inch`, `outch`: canales de entrada/salida.
- `hidch`: canales intermedios (por defecto `4*inch`).
- `kernel_size`, `stride`, `padding`: se propagan a los 2 bloques `BNReLUConv2d`.
- `norm`: tipo de normalización pasada a `BNReLUConv2d` (e.g., `"bn"`, `"gn"`, `None`).
- `act`: activación usada por `BNReLUConv2d` (por defecto `ReLU`).

**Estructura**
- `BNReLUConv2d(inch,hidch,...) → BNReLUConv2d(hidch,outch,...)`.

**Uso en SudokuAKOrN**
- Envuelto por `ResBlock(FF(...))` en el readout para refinar `c` con no linealidad.

