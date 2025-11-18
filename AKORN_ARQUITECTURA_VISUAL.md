# Arquitectura Visual de AKOrN para MNIST

## 🏗️ Estructura por Capas

```
┌────────────────────────────────────────────────────────────────────────┐
│                        AKOrN Network para MNIST                         │
│                                                                         │
│  Entrada: 1×28×28  (escala gris MNIST)                                │
│  Salida: 10  (classes 0-9)                                             │
└────────────────────────────────────────────────────────────────────────┘

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ETAPA 0: NORMALIZACIÓN                                                ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Entrada:    (batch, 1, 28, 28)                                       ┃
┃ Operación:  ToTensor() normalización RGB (identity para MNIST)       ┃
┃ Salida:     (batch, 1, 28, 28)  valores en [0, 1]                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                              ↓
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ETAPA 1: CONVOLUCIÓN INICIAL                                          ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Kernel:     1×3×3 → 64×3×3                                           ┃
┃ Entrada:    (batch, 1, 28, 28)                                       ┃
┃ Salida:     (batch, 64, 28, 28)  [Conv RGB]                         ┃
┃                                                                       ┃
┃ Parámetros: 1×(3×3+1)×64 = 640 pesos                                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                              ↓
╔═══════════════════════════════════════════════════════════════════════╗
║ BLOQUE KURAMOTO 1 (LAYER 1)                                           ║
║ ═════════════════════════════════════════════════════════════════════ ║
║                                                                       ║
║  KLayer:  64 osciladores × 28×28 posiciones = 50,176 osciladores   ║
║  Timesteps: T=3 (por defecto)                                        ║
║  Conectividad: Convoluciones locales (J="conv")                      ║
║                                                                       ║
║  ┌─────────────────────────────────────────────────────────────────┐║
║  │ TIMESTEP 0: Inicialización                                      ││
║  ├─────────────────────────────────────────────────────────────────┤║
║  │ x_0 = f_conv(entrada) = Conv(64→64, k=9)                        ││
║  │ Phase θ_0 = atan2(imag, real)                                   ││
║  │ Order R_0 = |mean(exp(i·θ_0))| ∈ [0,1]                          ││
║  │ Output:  (batch, 64, 28, 28)                                    ││
║  └─────────────────────────────────────────────────────────────────┘║
║                              ↓                                       ║
║  ┌─────────────────────────────────────────────────────────────────┐║
║  │ TIMESTEP 1: Evolución Kuramoto                                  ││
║  ├─────────────────────────────────────────────────────────────────┤║
║  │ C_1 = Conv(64→64, k=9) @ x_0      [Coupling]                   ││
║  │ Δθ_1 = atan2(imag(C_1)) - atan2(imag(x_0))                      ││
║  │ x_1 = x_0 + γ·sin(Δθ_1)            [Kuramoto update]           ││
║  │ R_1 = |mean(exp(i·θ_1))|  [Sincronización aumenta]             ││
║  │ Output:  (batch, 64, 28, 28)                                    ││
║  └─────────────────────────────────────────────────────────────────┘║
║                              ↓                                       ║
║  ┌─────────────────────────────────────────────────────────────────┐║
║  │ TIMESTEP 2: Convergencia                                        ││
║  ├─────────────────────────────────────────────────────────────────┤║
║  │ C_2 = Conv(64→64, k=9) @ x_1                                    ││
║  │ Δθ_2 = atan2(imag(C_2)) - atan2(imag(x_1))                      ││
║  │ x_2 = x_1 + γ·sin(Δθ_2)                                         ││
║  │ R_2 = |mean(exp(i·θ_2))|  [Patrón emerge]                       ││
║  │ Output:  (batch, 64, 28, 28)  ← SALIDA FINAL CAPA 1            ││
║  └─────────────────────────────────────────────────────────────────┘║
║                              ↓                                       ║
║  PARÁMETRO DE ORDEN (Sincronización):                               ║
║  ┌──────────────────────────────────────────────────────────────────┐║
║  │ Ejemplo: dígito "5" complejo                                    ││
║  │  R_0 ≈ 0.3   (inicio: bordes desorganizados)                    ││
║  │  R_1 ≈ 0.6   (estructura local emerge)                          ││
║  │  R_2 ≈ 0.8   (patrón "5" sincronizado)                          ││
║  │                                                                  ││
║  │ Ejemplo: dígito "1" simple                                      ││
║  │  R_0 ≈ 0.4   (rasgos lineales detectados)                       ││
║  │  R_1 ≈ 0.7   (rápida convergencia)                              ││
║  │  R_2 ≈ 0.85  (patrón "1" claro)                                 ││
║  └──────────────────────────────────────────────────────────────────┘║
╚═══════════════════════════════════════════════════════════════════════╝
                              ↓
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ETAPA READOUT 1: EXTRACCIÓN DE FEATURES                              ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Entrada:    (batch, 64, 28, 28)                                     ┃
┃ Operación:  FF Block + ResBlock                                     ┃
┃ FF:         64→128 conv channels                                    ┃
┃ ResBlock:   Conexión residual                                       ┃
┃ Pooling:    MaxPool 28×28 → 14×14                                   ┃
┃ Salida:     (batch, 128, 14, 14)                                    ┃
┃                                                                      ┃
┃ Parámetros: ~100k (aprox)                                          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                              ↓
╔═══════════════════════════════════════════════════════════════════════╗
║ BLOQUE KURAMOTO 2 (LAYER 2)                                           ║
║ ═════════════════════════════════════════════════════════════════════ ║
║                                                                       ║
║  KLayer:  128 osciladores × 14×14 posiciones = 25,088 osciladores  ║
║  Timesteps: T=3                                                      ║
║  Entrada:   (batch, 128, 14, 14)  [Features de capa 1]             ║
║                                                                       ║
║  Evolución similar a KLayer 1:                                       ║
║  t=0→1→2: Osciladores "ven" features de nivel 1                     ║
║           y detectan patrones de SEGUNDO orden                       ║
║                                                                       ║
║  Ejemplo: Dígito "5"                                                 ║
║  - Capa 1: Detectó bordes horizontales/verticales                    ║
║  - Capa 2: Los acopla → "ángulo de esquina", "forma de S"           ║
║                                                                       ║
║  R_layer2 típicamente > R_layer1  (más abstracto, mejor organizado)║
║  Salida:   (batch, 128, 14, 14)                                     ║
╚═══════════════════════════════════════════════════════════════════════╝
                              ↓
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ETAPA READOUT 2: ABSTRACCIÓN                                         ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Entrada:    (batch, 128, 14, 14)                                    ┃
┃ FF + ResBlock:  128→256 channels                                    ┃
┃ Pooling:    14×14 → 7×7                                             ┃
┃ Salida:     (batch, 256, 7, 7)                                      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                              ↓
╔═══════════════════════════════════════════════════════════════════════╗
║ BLOQUE KURAMOTO 3 (LAYER 3)  [OPCIONAL]                              ║
║ ═════════════════════════════════════════════════════════════════════ ║
║                                                                       ║
║  KLayer:  256 osciladores × 7×7 posiciones = 12,544 osciladores   ║
║  Timesteps: T=3                                                      ║
║  Entrada:   (batch, 256, 7, 7)  [Features abstractas]              ║
║                                                                       ║
║  Función: Detectar patrones GLOBALES                                 ║
║  - "¿Es un dígito con curva?"                                        ║
║  - "¿Tiene simetría?"                                                ║
║  - "¿Coincide con patrón 5, 2, 8?"                                   ║
║                                                                       ║
║  R_layer3 generalmente más alto: patrón muy definido                 ║
║  Salida:   (batch, 256, 7, 7)                                        ║
╚═══════════════════════════════════════════════════════════════════════╝
                              ↓
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ POOLING GLOBAL + CLASIFICADOR                                        ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Entrada:    (batch, 256, 7, 7)                                      ┃
┃ Op. 1:      AdaptiveAvgPool2d(1)  → (batch, 256, 1, 1)              ┃
┃ Op. 2:      Flatten → (batch, 256)                                  ┃
┃ Op. 3:      Linear(256, 10)  [Clasificador]                         ┃
┃                                                                      ┃
┃ Logits: [s_0, s_1, s_2, ..., s_9]  scores para cada clase           ┃
┃ Salida: Softmax → probabilidades [p_0, ..., p_9]                    ┃
┃ Predicción: argmax(logits) ∈ {0,1,2,...,9}                         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                              ↓
                         SALIDA: 10
                    (Clase predicha 0-9)
```

---

## 📋 Tabla de Capas y Parámetros

| Capa | Tipo | Entrada | Salida | Canales | Parámetros | Notas |
|------|------|---------|--------|---------|------------|-------|
| 0 | RGB Norm | 1×28×28 | 1×28×28 | 1 | 0 | Identity para MNIST |
| 1 | Conv | 1×28×28 | 64×28×28 | 1→64 | 640 | Kernel inicial 3×3 |
| 2 | KLayer | 64×28×28 | 64×28×28 | 64 | ~50k | T=3, 50,176 osciladores |
| 3 | Readout | 64×28×28 | 128×14×14 | 64→128 | ~100k | FF + ResBlock + Pool |
| 4 | KLayer | 128×14×14 | 128×14×14 | 128 | ~25k | T=3, 25,088 osciladores |
| 5 | Readout | 128×14×14 | 256×7×7 | 128→256 | ~250k | FF + ResBlock + Pool |
| 6 | KLayer | 256×7×7 | 256×7×7 | 256 | ~12k | T=3, 12,544 osciladores |
| 7 | Pool + FC | 256×7×7 | 10 | 256→10 | 2,570 | Clasificador final |

**Total Parámetros**: ~440k (modelo mediano)

---

## 🔄 Flujo de Datos: Ejemplo Imagen "5"

```
ENTRADA: Imagen 28×28 = 
┌──────────────────────┐
│ ▓▓░░░░░░░░░░░░░░░░░░│
│ ▓▓░░░░░░░░░░░░░░░░░░│
│ ░░░░░░░░░░░░░░░░░░░░│  ← Píxeles: 0=blanco, 1=negro
│ ░░▓▓▓▓░░░░░░░░░░░░░░│
│ ░░▓▓▓▓░░░░░░░░░░░░░░│
│ ░░░░░░░░░░░░░░░░░░░░│
│ ░░░░░░░░░░▓▓░░░░░░░░│
│ ░░░░░░░░░░▓▓░░░░░░░░│
│ ░░▓▓▓▓▓▓▓▓░░░░░░░░░░│
└──────────────────────┘

DESPUÉS CAPA 1 (Conv inicial):
64 canales activados:
├─ Canal_0: Detecta bordes verticales
├─ Canal_1: Detecta bordes horizontales
├─ Canal_2: Detecta esquinas
├─ Canal_3-63: Otros filtros
│   (cada uno extrae feature diferente)
└─ Output: (batch, 64, 28, 28)

DURANTE KLayer 1 (T=3):
t=0: Osciladores en desorden (R ≈ 0.2)
     "Bordes aislados detectados"
t=1: Empiezan a acoplarse (R ≈ 0.5)
     "Bordes se conectan"
t=2: Sincronización (R ≈ 0.7)
     "Forma de 'S' emergente"

READOUT 1: Extrae features sincronizadas (128 canales)
           Pooling reduce a 14×14

KLayer 2 (T=3):
t=0: "Ve" la forma 'S' emergente (R ≈ 0.4)
t=1: Estructura global (R ≈ 0.7)
t=2: Patrón coherente "definitivamente un 5" (R ≈ 0.85)

READOUT 2: Abstrae a features de alto nivel (256 canales)
           Pooling reduce a 7×7

KLayer 3 (T=3):
t=0: Patrón global bien definido (R ≈ 0.6)
t=1: Discriminación fina (R ≈ 0.8)
t=2: Decisión final (R ≈ 0.9)
     "Es definitivamente un 5, no confundir con 2 u 8"

CLASIFICADOR FINAL:
Logits ≈ [-2.1, -1.5, 0.5, -0.8, -1.2, 4.2, -1.0, -0.3, 0.1, -2.5]
         └─ clase 5 tiene score máximo (4.2)

PREDICCIÓN: 5 ✓ (con confianza = softmax(4.2) ≈ 0.98)
```

---

## 🔬 Parámetros de Kuramoto Explicados

### En cada KLayer:

```python
def kuramoto_update(x_t, C_t, gamma=1.0):
    """
    Actualización de Kuramoto en espacio complejo
    
    Interpretación:
    - x_t: estado (velocidad angular) de osciladores en tiempo t
    - C_t: input acoplado desde convoluciones
    - gamma: fuerza de acoplamiento (learning rate temporal)
    """
    
    # Extraer fases
    theta_x = torch.atan2(x_t.imag, x_t.real)      # fase de estado
    theta_c = torch.atan2(C_t.imag, C_t.real)      # fase de acoplamiento
    
    # Diferencia de fases
    delta_theta = theta_c - theta_x
    
    # Ecuación de Kuramoto clásica
    x_new = x_t + gamma * torch.sin(delta_theta)
    
    # Parámetro de orden (sincronización)
    # R = |⟨exp(i*theta)⟩| mide coherencia de fase
    phase = torch.atan2(x_t.imag, x_t.real)
    exp_i_theta = torch.exp(1j * phase)
    R = torch.abs(torch.mean(exp_i_theta))  # ∈ [0, 1]
    
    return x_new, R
```

### Comparación con Tu Análisis Kuramoto

**Tu setup**:
```python
# En tu análisis
T = 100  # Muy alto
alpha ∈ [0, 0.1]  # Alpha varía
R(alpha) → busca transición de fase
alpha_c = donde R(alpha) salta
```

**AKOrN setup**:
```python
# En AKOrN
T = 3  # Bajo (no espera convergencia completa)
alpha ≈ implícito en convoluciones (aprendible)
R(t) → evoluciona con timesteps
características ≠ punto crítico (objetivo diferente)
```

---

## 🧮 Fórmulas Clave

### 1. Dinámica de Kuramoto por Timestep
$$x_{t+1} = x_t + \gamma \sin(\theta_c - \theta_x)$$

Donde:
- $x_t$ = estado del oscilador en tiempo t
- $\gamma$ = parámetro de paso (default 1.0)
- $\theta_x = \arg(x_t)$ = fase del estado
- $\theta_c = \arg(C_t)$ = fase del acoplamiento

### 2. Parámetro de Orden
$$R(t) = \left| \frac{1}{N} \sum_{i=1}^N e^{i\theta_i(t)} \right|$$

Interpretación:
- $R = 0$: Completamente desincronizados
- $R = 1$: Perfectamente sincronizados
- $0 < R < 1$: Sincronización parcial

### 3. Conectividad Convolucional
$$C_t = \text{Conv}(x_t; \mathbf{w})$$

Donde $\mathbf{w}$ son pesos de convolución (aprendibles durante entrenamiento)

---

## 🎯 Resumen de Funcionamiento

```
PRINCIPIO: 
Osciladores localmente acoplados mediante convoluciones
→ Sincronización de fase emerge según estructura visual
→ Diferentes patrones → diferentes R(t)
→ Features se construyen a partir de patrones sincrónicos

RESULTADO:
Red profunda que entiende visión a través de dinámicas de fase
Más interpretable que CNNs tradicionales (R(t) muestra sincronización)
Potencialmente sensible a estructuras críticas (como tu α_c)

PARA MNIST:
- Capas bajas: Sincronización local de bordes
- Capas medias: Sincronización de estructura
- Capas altas: Sincronización global de forma
- Output: Predicción 0-9 basada en patrones sincrónicos

COMPARACIÓN CON TU ANÁLISIS:
Tu análisis busca: α_c donde R(α) cambia (transición de fase)
AKOrN busca: R(t) que predice clase (emergencia de features)

CONEXIÓN POSIBLE:
Imágenes con α_c bajo (R(α) suave) pueden tener
R(t) convergente rápido en AKOrN (fáciles de clasificar)

Imágenes con α_c alto (R(α) abrupto) pueden tener
R(t) complejo en AKOrN (difíciles de clasificar)
```

---

**Arquitectura lista para implementación**  
**Documentación compatible con ambos análisis (Kuramoto + AKOrN)**

