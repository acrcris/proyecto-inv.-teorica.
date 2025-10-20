# Criticalidad en sistemas de osciladores acoplados

Excelente pregunta, Laura 🌿.
La *criticalidad* en un sistema de *osciladores acoplados* (por ejemplo, redes de Kuramoto, modelos de Hopf o redes neuronales oscilatorias) se refiere al punto de *transición entre orden y desorden*, o entre *fase incoherente y sincronizada*. Medirla implica identificar señales o métricas que indiquen que el sistema está cerca de ese punto crítico.

Aquí tienes las *formas más comunes y rigurosas* de medir o detectar la criticalidad 👇

---

### 1. Medir la sincronización global (orden de Kuramoto)

La medida más clásica es el *parámetro de orden de Kuramoto*, definido como

```math
R(t) e^{i \psi(t)} = \frac{1}{N} \sum_{j=1}^{N} e^{i \theta_j(t)}
```

Donde:

- \(\theta_j\) son las fases de los osciladores.
- \(R(t) \in [0, 1]\) mide el grado de sincronía global.
- \(\psi(t)\) es la fase promedio del grupo.

**Interpretación**

- \(R \approx 0\): desorden → fase incoherente.
- \(R \approx 1\): sincronización total.
- *El punto crítico* es donde \(R\) pasa de \(\approx 0\) a \(> 0\), lo que ocurre al variar el acoplamiento \(K\) en el modelo de Kuramoto.

**Cómo medirlo**

- Simula para distintos valores de \(K\).
- Calcula el promedio temporal \(\langle R \rangle\) en estado estacionario.
- Busca el valor \(K_c\) donde \(\langle R \rangle\) crece abruptamente (transición de fase).

---

### 2. Distribución de avalanchas o eventos

Si tu sistema tiene eventos intermitentes (picos de sincronía, oscilaciones locales, etc.), puedes medir *avalanche statistics*:

- Calcula los tamaños \(S\) y duraciones \(T\) de las avalanchas (por ejemplo, grupos de osciladores que se activan juntos).
- Si la distribución sigue una *ley de potencia*:

```math
P(S) \sim S^{-\alpha}, \quad P(T) \sim T^{-\tau}
```

Entonces el sistema muestra *comportamiento crítico* (sin escala característica).

**Evidencia de criticalidad**

- Exponentes estables \(\alpha, \tau\) (típicamente entre 1.5 y 2.5).
- Escalamiento entre \(S\) y \(T\): \(S \sim T^{\gamma}\).

---

### 3. Correlaciones espacio-temporales largas

En el punto crítico:

- Las *correlaciones* entre osciladores decaen como una ley de potencia, no exponencialmente.
- Puedes medir:

```math
C_{ij} = \langle \cos(\theta_i - \theta_j) \rangle_t
```

- Analiza cómo \(C(r)\) (correlación media a distancia \(r\)) decae.

Un decaimiento tipo \(C(r) \sim r^{-\eta}\) indica criticalidad. Si es exponencial \(C(r) \sim e^{-r/\xi}\), estás lejos del punto crítico.

---

### 4. Fluctuaciones máximas de sincronía

Cerca del punto crítico, las fluctuaciones de \(R(t)\) son *máximas*. Puedes medir:

```math
\chi = N\left(\langle R^2\rangle - \langle R\rangle^2\right)
```

El máximo de \(\chi(K)\) indica la *transición crítica* (análoga a la susceptibilidad en física estadística).

---

### 5. Dimensión fractal o entropía de permutación

Para sistemas con dinámicas no estacionarias, puedes usar:

- *Entropía de permutación*, que mide complejidad temporal.
- *Dimensión fractal del atractor* (por ejemplo, con correlación de Grassberger-Procaccia).

El punto crítico suele maximizar la *complejidad*: ni orden total ni caos completo.

---

### Ejemplo resumido en práctica (modelo de Kuramoto)

```python
# ejemplo básico
import numpy as np


def kuramoto(theta, K, omega, dt):
    N = len(theta)
    dtheta = omega + (K / N) * np.sum(np.sin(theta[:, None] - theta), axis=1)
    return theta + dtheta * dt


N = 100
omega = np.random.normal(0, 1, N)
theta = np.random.uniform(0, 2 * np.pi, N)
dt = 0.01

for K in np.linspace(0, 5, 50):
    for t in range(10000):
        theta = kuramoto(theta, K, omega, dt)
    R = np.abs(np.mean(np.exp(1j * theta)))
    print(K, R)
```

Luego grafica \(R(K)\) para visualizar la transición y detectar el punto crítico.

---

¿Podrías contarme qué tipo de *osciladores acoplados* estás usando (Kuramoto, Hopf, red neuronal, mecánicos...)? Así puedo indicarte exactamente *cómo medir la criticalidad* en tu caso (hay diferencias importantes entre modelos).
