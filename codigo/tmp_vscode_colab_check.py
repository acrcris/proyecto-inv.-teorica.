#!/usr/bin/env python3
# TEMPORAL: borrar tras probar
# Este script solo verifica que vscode_colab esté importable y muestra su API pública.
# No inicia servidores ni descarga nada.

import inspect
import sys

try:
    import vscode_colab
except Exception as e:
    print("Fallo import vscode_colab:", type(e).__name__, e)
    sys.exit(1)

print("OK: import vscode_colab")
print("Ruta módulo:", getattr(vscode_colab, "__file__", "N/A"))
print("Versión:", getattr(vscode_colab, "__version__", "desconocida"))

public = [a for a in dir(vscode_colab) if not a.startswith("_")]
print("Atributos públicos:", public)

for name in public:
    obj = getattr(vscode_colab, name)
    if inspect.isfunction(obj):
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "(...)"
        print("func", name, sig)
    elif inspect.isclass(obj):
        print("class", name)

