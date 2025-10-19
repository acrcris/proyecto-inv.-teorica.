**Resumen**
- Propósito: bloque residual genérico que envuelve una función `fn` arbitraria y aplica `x + fn(x)`.
- Entrada/Salida: misma forma que `x`.

**Ubicación**
- `akorn/source/layers/common_layers.py:37`

**Constructor**
- `fn`: módulo a aplicar dentro del salto residual. En Sudoku se usa con `FF`.

**Forward**
- Retorna `x + fn(x)`, útil para estabilizar el entrenamiento y permitir refinamiento incremental.

**Uso en SudokuAKOrN**
- Activa si `nl=True` en el readout: `ResBlock(FF(...))` seguido de `BNReLUConv2d`.

