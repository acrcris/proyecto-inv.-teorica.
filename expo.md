**Breve Contexto**
- Las redes neuronales tradicionales en aprendizaje automático aprenden una función que mapea entradas a salidas ajustando pesos para minimizar una pérdida. Son muy eficaces en visión, lenguaje y tareas supervisadas, pero cuando el problema exige respetar reglas estrictas o generalizar a distribuciones diferentes, pueden apoyarse en correlaciones espurias y perder robustez.
- Por esa razón, investigaciones recientes han propuesto redes basadas en osciladores acoplados. AKOrN es un ejemplo: modela unidades oscilatorias que interactúan y buscan configuraciones coherentes según un principio de energía, lo que encaja de forma natural con problemas de satisfacción de restricciones como el Sudoku.

**Cómo Funcionan las Neuronas en una Red Normal**
- Cada neurona calcula una combinación lineal de sus entradas seguida de una no linealidad: `y = σ(w·x + b)`. Varias capas de estas neuronas componen una función compleja; el entrenamiento por gradiente ajusta los pesos para minimizar la pérdida sobre datos etiquetados.
- Ventaja: inferencia rápida en un solo “paso” hacia adelante y gran compatibilidad con hardware actual. Desventaja: la salida es un mapeo estático; las reglas del problema no se imponen explícitamente y la generalización puede deteriorarse fuera de la distribución vista en entrenamiento.

**AKOrN: Diferencias Clave y Funcionamiento**
- Las “neuronas” son osciladores que evolucionan en el tiempo y se acoplan mediante una conectividad que les permite sincronizarse o desincronizarse según las restricciones del problema (modelo de Kuramoto).
- La dinámica itera durante varios pasos y luego un módulo de lectura proyecta el estado final a la predicción deseada. En evaluación se puede extender el número de pasos para favorecer la convergencia y realizar múltiples inicializaciones, seleccionando la de menor energía (energy voting).

**Beneficios y Falencias**
- Modelos tradicionales
  - Beneficios: entrenamiento estandarizado, ecosistema maduro, inferencia muy rápida, excelente desempeño dentro de la distribución de entrenamiento.
  - Falencias: pueden depender de patrones superficiales; el cumplimiento explícito de reglas no está garantizado y la robustez fuera de distribución puede ser limitada.
- AKOrN
  - Beneficios: incorpora un principio de energía y una dinámica recurrente que favorece soluciones consistentes con las restricciones; ofrece señales interpretables (energía, sincronización) y puede mejorar la generalización al ajustar la inferencia.
  - Falencias: mayor coste de cómputo en inferencia al aumentar pasos e intentos; sensibilidad a hiperparámetros y necesidad de una configuración cuidadosa.
