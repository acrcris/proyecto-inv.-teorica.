# 🧪 Sección de Test de Consistencia - Documentación

## 🎯 Objetivo

La **Sección 13** de ambos notebooks implementa un sistema completo de validación que permite:

1. ✅ **Verificar** que el cálculo de C_crítico es **consistente** y **reproducible**
2. 📊 **Visualizar** la curva completa R_final vs C para ejemplos de cada clase
3. 🔍 **Identificar** la transición de fase de sincronización en osciladores de Kuramoto
4. ⚠️ **Detectar** posibles inconsistencias o errores en el procesamiento

---

## 📋 Estructura de la Sección de Test

### Celda 1: Configuración
```python
NUM_EJEMPLOS_POR_CLASE = 3  # Imágenes por clase a testear
CLASES_TEST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Todas las clases
```

**Personalización:**
- Cambia `NUM_EJEMPLOS_POR_CLASE = 1` para test rápido
- Usa `CLASES_TEST = [0, 1, 2]` para testear solo algunas clases
- Usa `NUM_EJEMPLOS_POR_CLASE = 10` para test exhaustivo

### Celda 2: Selección de Ejemplos
- Lee la base de datos SQLite
- Selecciona **aleatoriamente** N imágenes de cada clase
- Usa semilla fija (`SEED=1`) para reproducibilidad

**Output esperado:**
```
✅ Ejemplos seleccionados:
  Clase 0: 3 imágenes
  Clase 1: 3 imágenes
  ...
  Clase 9: 3 imágenes
```

### Celda 3: Recálculo de Curvas
- Recalcula la curva **completa** R_final vs C para cada ejemplo
- Guarda tanto el valor de C_crítico guardado como el recalculado
- Incluye toda la información: R_final, c_range, índice del máximo

**Propósito:** Verificar que el cálculo sea determinístico y reproducible.

### Celda 4: Gráficas por Clase
**Visualización:** Grid de 2×5 (10 subplots, una por clase)

**Cada subplot muestra:**
- 📈 Curvas R_final vs C de los 3 ejemplos seleccionados
- 📍 Línea vertical en C_crítico de cada ejemplo
- 🔴 Línea roja sólida: Media de C_crítico de toda la clase
- 🎯 Puntos: Valor de R_final en C_crítico

**Interpretación:**
```
┌─────────────────────────┐
│ Clase 0                 │
│                         │
│ R │     ╱─────          │ ← Transición clara
│   │    ╱                │
│   │___╱                 │
│   └────────────── C     │
│      ↑                  │
│   C_crítico             │
└─────────────────────────┘
```

**Lo que buscamos:**
- ✅ Transición **suave** de R bajo → R alto
- ✅ C_crítico **agrupado** (poca variabilidad)
- ✅ Media de clase **cerca** de los ejemplos individuales
- ⚠️ Si hay gran dispersión → revisar parámetros

### Celda 5: Gráficas Detalladas con Derivada
**Visualización:** Grid de 2×5 con **doble eje Y**

**Cada subplot muestra:**
- **Eje izquierdo (azul):** R_final vs C
- **Eje derecho (naranja):** dR/dC (derivada)
- 🔴 Línea vertical roja: C_crítico (donde dR/dC es máximo)
- 🔵 Punto azul: Valor de R en C_crítico
- 🟠 Cuadrado naranja: Valor máximo de dR/dC

**Interpretación:**
```
R_final │         ╱───── (azul)
        │        ╱
        │    ___╱
        │___╱
        └────────────── C
           ↑
dR/dC   │  ╱╲  (naranja, punteada)
        │ ╱  ╲
        │╱____╲___
        └────────────── C
           ↑
        C_crítico = argmax(dR/dC)
```

**Lo que buscamos:**
- ✅ Pico **claro** y **único** en dR/dC
- ✅ Pico coincide con la **región de transición** en R_final
- ⚠️ Si hay múltiples picos → imagen ruidosa o parámetros subóptimos
- ⚠️ Si no hay pico → C_crítico mal calculado

### Celda 6: Verificación Numérica de Consistencia
**Output esperado:**
```
VERIFICACIÓN DE CONSISTENCIA
============================================================

📊 Clase 0:
  ✅ Img  1234: Guardado=0.1500, Recalculado=0.1500, Δ=0.0000
  ✅ Img  5678: Guardado=0.1450, Recalculado=0.1450, Δ=0.0000
  ✅ Img  9012: Guardado=0.1600, Recalculado=0.1600, Δ=0.0000

📊 Clase 1:
  ✅ Img  3456: Guardado=0.1300, Recalculado=0.1300, Δ=0.0000
  ...

✅ Todas las mediciones son consistentes (tolerancia: ±0.02)
```

**Tolerancia:** ±0.02 (2× el paso de c_range que es 0.01)

**Posibles resultados:**
- ✅ **Δ = 0.0000**: Perfecto, cálculo 100% reproducible
- ✅ **Δ ≤ 0.01**: Excelente, dentro de la resolución de c_range
- ⚠️ **Δ > 0.02**: Advertencia, revisar por qué hay diferencia

**Causas de inconsistencias:**
1. Cambio en `SEED` entre ejecuciones
2. Cambio en `x_init` (no es el mismo estado inicial)
3. Modelo `kblock` recargado con pesos diferentes
4. Error numérico acumulado (muy raro)

### Celda 7: Visualización Combinada
**Layout:** Imagen MNIST + Curva R_final + Derivada dR/dC

```
┌────────┬─────────────────────┬─────────────────────┐
│ Imagen │   R_final vs C      │    dR/dC vs C       │
│  MNIST │                     │                     │
│   (0)  │      ╱────          │      ╱╲             │
│        │     ╱               │     ╱  ╲            │
│        │  __╱                │  __╱    ╲__         │
├────────┼─────────────────────┼─────────────────────┤
│   (1)  │   ...               │   ...               │
└────────┴─────────────────────┴─────────────────────┘
```

**Propósito:** Ver la **imagen original** junto con su comportamiento dinámico.

**Observaciones esperadas:**
- Dígitos **simples** (1, 0) → C_crítico **bajo** (se sincronizan fácil)
- Dígitos **complejos** (8, 9) → C_crítico **alto** (necesitan más acoplamiento)
- Relación entre **estructura visual** y **dinámica de sincronización**

---

## 🔍 Cómo Interpretar los Resultados

### 1. Curva R_final vs C Típica

**Esperada:**
```python
R_final
  1.0 │         ╱──────────  ← Sincronización completa
      │        ╱
  0.5 │       ╱              ← Transición (C_crítico aquí)
      │      ╱
  0.0 │_____╱                ← Sin sincronización
      └──────────────────── C
      0.0   C_c    2.0
```

**Características:**
- R ≈ 0 para C pequeño (osciladores desacoplados)
- **Transición abrupta** cerca de C_crítico
- R → alto para C > C_crítico (sincronización)
- Forma sigmoide o escalón

### 2. Derivada dR/dC

**Esperada:**
```python
dR/dC
      │     ╱╲              ← Pico único y claro
      │    ╱  ╲
      │   ╱    ╲
      │__╱      ╲___
      └──────────────────── C
            ↑
         C_crítico
```

**Criterio de calidad:**
- ✅ **Pico único**: Transición bien definida
- ✅ **Pico alto**: Transición abrupta (mejor señal)
- ✅ **Pico estrecho**: C_crítico preciso
- ⚠️ **Pico ancho**: Transición gradual (menos crítica)
- ❌ **Sin pico**: No hay transición clara

### 3. Consistencia Entre Imágenes de la Misma Clase

**Esperado:**
```
Clase 0: C_c = 0.15 ± 0.03  ← Baja variabilidad ✅
Clase 8: C_c = 0.18 ± 0.08  ← Alta variabilidad ⚠️
```

**Interpretación:**
- **Baja std**: Imágenes de la clase son **similares** dinámicamente
- **Alta std**: Imágenes de la clase son **heterogéneas**
  - Puede ser normal (ej: dígito 8 tiene muchas variantes)
  - Puede indicar outliers o errores

### 4. Comparación Entre Clases

**Hipótesis a verificar:**
1. **Clases simples (0, 1) → C_crítico bajo**
   - Menos complejidad visual → se sincronizan fácil
2. **Clases complejas (8, 9) → C_crítico alto**
   - Más complejidad visual → requieren más acoplamiento
3. **Variabilidad intra-clase**
   - Clases con escritura uniforme → baja variabilidad
   - Clases con variantes (ej: 7 con/sin barra) → alta variabilidad

---

## 🧪 Casos de Uso del Test

### Caso 1: Test Rápido Después del Procesamiento

**Configuración:**
```python
NUM_EJEMPLOS_POR_CLASE = 1
CLASES_TEST = [0, 1, 2]  # Solo 3 clases
```

**Tiempo:** ~1-2 minutos
**Propósito:** Verificación rápida de que todo funciona

### Caso 2: Validación Completa

**Configuración:**
```python
NUM_EJEMPLOS_POR_CLASE = 5
CLASES_TEST = range(10)  # Todas las clases
```

**Tiempo:** ~10-15 minutos
**Propósito:** Test exhaustivo antes de publicar resultados

### Caso 3: Debug de Clase Específica

**Configuración:**
```python
NUM_EJEMPLOS_POR_CLASE = 10
CLASES_TEST = [8]  # Solo clase con problemas
```

**Propósito:** Investigar por qué una clase tiene resultados raros

### Caso 4: Comparación de Parámetros

**Flujo:**
1. Procesa dataset con parámetros A (ej: T=30)
2. Ejecuta test → guarda gráficos
3. Procesa dataset con parámetros B (ej: T=50)
4. Ejecuta test → compara gráficos

**Propósito:** Ver cómo cambian las curvas con distintos parámetros

---

## 🎨 Gráficos Generados

### 1. `test_consistencia_R_vs_C.png`
- Grid 2×5 con curvas de múltiples ejemplos por clase
- Incluye media de clase
- **Uso:** Vista general de consistencia

### 2. `test_detallado_R_derivada.png`
- Grid 2×5 con R_final y dR/dC en doble eje
- **Uso:** Verificar que C_crítico es el máximo de dR/dC

### 3. `test_imagenes_curvas.png`
- Imagen MNIST + R_final + dR/dC en una fila
- **Uso:** Relacionar apariencia visual con dinámica

**Ubicación:** Todos en Google Drive (`DRIVE_PATH/`)

---

## 🚨 Problemas Comunes y Soluciones

### Problema 1: "No hay ejemplos para mostrar"

**Causa:** Clase no tiene imágenes procesadas en la BD

**Solución:**
```python
# Verificar qué clases tienen datos
import sqlite3
conn = sqlite3.connect(DB_PATH)
for clase in range(10):
    cursor = conn.cursor()
    cursor.execute(f'SELECT COUNT(*) FROM clase_{clase}')
    print(f"Clase {clase}: {cursor.fetchone()[0]} imágenes")
conn.close()
```

### Problema 2: Inconsistencias detectadas (Δ > 0.02)

**Causas posibles:**
1. **Seed diferente**: Asegúrate de ejecutar `set_seed(SEED)` antes del test
2. **x_init diferente**: Usa el mismo x_init que en el procesamiento
3. **Modelo modificado**: Verifica que `kblock` no haya cambiado

**Debug:**
```python
# Verificar que x_init es el mismo
print(f"x_init hash: {hash(x_init.cpu().numpy().tobytes())}")
# Si es diferente, regenerar:
set_seed(SEED)
x_init = torch.randn(1, PARAMS['ch'], PARAMS['h'], PARAMS['w'])
```

### Problema 3: Curvas no tienen transición clara

**Causas posibles:**
1. **T muy pequeño**: Aumenta T (ej: de 30 a 50)
2. **gamma muy bajo**: Aumenta gamma (ej: de 0.7 a 0.9)
3. **Imagen ruidosa**: Normal en algunos casos, no es error

**Verificación:**
```python
# Ver si R_final varía con C
for dato in datos_test[0]:  # Clase 0
    R_vals = np.array(dato['R_final'])
    print(f"Rango de R: {R_vals.min():.3f} - {R_vals.max():.3f}")
    # Esperado: 0.1 - 0.8 (buena variación)
    # Problemático: 0.4 - 0.45 (poca variación)
```

### Problema 4: Múltiples picos en dR/dC

**Causa:** Transición no es suave (puede ser normal)

**Interpretación:**
- Si picos son pequeños → ruido numérico, ignorar
- Si picos son grandes → imagen compleja, múltiples escalas

**Suavizar derivada:**
```python
from scipy.ndimage import gaussian_filter1d
dR_dc_suave = gaussian_filter1d(dR_dc, sigma=2)
```

---

## 📊 Análisis Avanzado

### Comparar Media de Test vs Media de Clase Completa

```python
for clase in CLASES_TEST:
    if clase not in datos_test:
        continue
    
    # Media de ejemplos del test
    c_crits_test = [d['c_critical_recalculado'] for d in datos_test[clase]]
    media_test = np.mean(c_crits_test)
    
    # Media de toda la clase (de resumen)
    media_clase = resultados_resumen[clase]['mean']
    
    diferencia = abs(media_test - media_clase)
    print(f"Clase {clase}: Test={media_test:.4f}, "
          f"Clase={media_clase:.4f}, Δ={diferencia:.4f}")
```

**Esperado:** Δ < 0.05 (test representa bien la clase)

### Detectar Outliers

```python
for clase in CLASES_TEST:
    if clase not in datos_test:
        continue
    
    c_crits = [d['c_critical_recalculado'] for d in datos_test[clase]]
    media = np.mean(c_crits)
    std = np.std(c_crits)
    
    for dato in datos_test[clase]:
        c = dato['c_critical_recalculado']
        z_score = abs(c - media) / std
        if z_score > 2:  # Más de 2 desviaciones estándar
            print(f"⚠️  Outlier: Clase {clase}, "
                  f"Img {dato['image_idx']}, C={c:.4f} (z={z_score:.2f})")
```

---

## 🎯 Checklist de Validación

Antes de considerar los resultados como válidos:

- [ ] ✅ Test ejecutado sin errores
- [ ] ✅ Todas las clases tienen ejemplos procesados
- [ ] ✅ Consistencia: todas las Δ < 0.02
- [ ] ✅ Curvas R_final muestran transición clara
- [ ] ✅ Derivadas tienen pico único en C_crítico
- [ ] ✅ Media de test ≈ media de clase completa
- [ ] ✅ No hay outliers inexplicables
- [ ] ✅ Gráficos guardados en Drive
- [ ] ✅ Resultados físicamente sensibles (0 < C_c < 2)

---

## 📝 Resumen

La sección de test permite:

1. **Verificar consistencia** del cálculo de C_crítico
2. **Visualizar** la transición de sincronización
3. **Detectar errores** antes de publicar resultados
4. **Entender** la relación entre imagen y dinámica
5. **Comparar** comportamiento entre clases

**Tiempo de ejecución:**
- Test rápido (1 img/clase): ~1 min
- Test normal (3 img/clase): ~5 min
- Test exhaustivo (10 img/clase): ~15 min

**Salidas:**
- 3 gráficos PNG en Google Drive
- Reporte de consistencia en terminal
- Detección automática de inconsistencias

¡Ahora tienes una validación completa de tus resultados! 🎉
