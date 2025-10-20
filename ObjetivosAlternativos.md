# Objetivos del proyecto AKoRN (con citas breves)

## ✅ OBJETIVOS ALCANZABLES CON EL CÓDIGO AKORN ACTUAL

### 1) **✅ Detectar evidencia de estado crítico** 
   - **QUÉ HACER:** Analizar las series temporales de los osciladores Kuramoto (fases, magnitudes, energía) para verificar leyes de potencia en avalanchas de sincronización y distribuciones de tamaño finito.
   - **CÓDIGO DISPONIBLE:**
     - `KLayer` en `source/layers/klayer.py`: implementa dinámicas tipo Kuramoto con acoplamiento configurable.
     - `return_xs=True, return_es=True` en `KBlock`: devuelve series temporales de estados y energía.
     - Ya tienes en `kuramoto_pruebas_basico.ipynb`: cálculo de parámetro de orden R(t), DFA, PSD.
   - **IMPLEMENTACIÓN:** Medir distribución de duraciones/tamaños de eventos de sincronización, calcular exponentes críticos (DFA α ≈ 1, pendiente PSD ≈ -1).
   - **Citas:** *Bullmore & Sporns 2009* (PDF p. 4: Box 3, redes "scale-free" y "small-world").

### 2) **✅ Construir y analizar la red funcional (correlaciones/PLV)**
   - **QUÉ HACER:** Extraer series temporales de cada píxel/oscilador, calcular correlaciones de Pearson o PLV (phase locking value) entre pares, construir matriz de adyacencia.
   - **CÓDIGO DISPONIBLE:**
     - Series temporales `xs` de shape `[T, B, ch, H, W]` ya disponibles.
     - Tu notebook tiene: `np.corrcoef(serie_channel_mean.T)` para correlaciones.
     - En `analisis_criticalidad_minimalista/analisis/criticalidad.py`: clases `Correlacion` y `MutualInformation`.
   - **IMPLEMENTACIÓN:** Calcular matriz de correlación/PLV espacial (píxeles como nodos), umbralizar o usar pesos, visualizar con networkx.
   - **Citas:** *Bullmore & Sporns 2009* (PDF p. 2: pipeline); *Rubinov & Sporns 2010* (PDF p. 3: redes nulas).

### 3) **✅ Segregación (clusters/modularidad)**
   - **QUÉ HACER:** Sobre la red funcional, detectar comunidades (Louvain, spectral clustering) y medir coeficiente de clustering local/global.
   - **CÓDIGO DISPONIBLE:**
     - `eval_obj.py` ya usa clustering (K-Means, agglomerative) sobre embeddings espaciales.
     - `segmentacion/clustering.py`: clase `SegmentadorKMeans`.
   - **IMPLEMENTACIÓN:** Aplicar clustering sobre matriz de correlación, calcular modularidad Q usando networkx o BCT (Brain Connectivity Toolbox en Python).
   - **Citas:** *Rubinov & Sporns 2010* (PDF p. 3: segregación y clustering; modularidad).

### 4) **✅ Integración (caminos/eficiencia)**
   - **QUÉ HACER:** Calcular longitud de camino promedio y eficiencia global sobre la red funcional.
   - **CÓDIGO DISPONIBLE:**
     - Matriz de adyacencia construible desde correlaciones (objetivo 2).
   - **IMPLEMENTACIÓN:** Usar networkx: `nx.average_shortest_path_length()`, `nx.global_efficiency()`.
   - **Citas:** *Bullmore & Sporns 2009* (PDF p. 3: "Path length and efficiency"); *Rubinov & Sporns 2010* (PDF p. 7: eficiencia global).

### 5) **✅ Small-world**
   - **QUÉ HACER:** Verificar alta clustering + caminos cortos; comparar con redes aleatorias de Erdős-Rényi.
   - **CÓDIGO DISPONIBLE:**
     - Clustering y caminos (objetivos 3 y 4).
   - **IMPLEMENTACIÓN:** Calcular C (clustering) y L (path length), comparar con C_rand y L_rand, reportar σ = (C/C_rand) / (L/L_rand).
   - **Citas:** *Bullmore & Sporns 2009* (PDF p. 4: small-worldness estandarizada).

### 6) **⚠️ Hubs y roles (provinciales vs. conectores)** - PARCIAL
   - **QUÉ HACER:** Calcular centralidades y caracterizar hubs por su rol dentro/entre módulos.
   - **CÓDIGO DISPONIBLE:**
     - Matriz de adyacencia y comunidades (objetivos 2 y 3).
   - **IMPLEMENTACIÓN:** Usar networkx para centralidades (degree, betweenness, closeness); calcular z-score intra-módulo y participation coefficient (fórmulas en Rubinov & Sporns 2010).
   - **NOTA:** Requiere implementar métricas específicas de rol modular.
   - **Citas:** *Bullmore & Sporns 2009* (PDF p. 3: hubs y roles); *Rubinov & Sporns 2010* (PDF p. 9: z-score y participation).

### 7) **❌ Motifs y mesoescala** - NO ALCANZABLE DIRECTAMENTE
   - **QUÉ HACER:** Detectar patrones recurrentes de conectividad (triángulos, reciprocidad, etc.).
   - **CÓDIGO DISPONIBLE:** No hay implementación de motif detection en AKOrN.
   - **IMPLEMENTACIÓN:** Requiere usar librerías externas (networkx, graph-tool) o implementar conteo de motifs de 3-nodos.
   - **PRIORIDAD:** Baja, requiere trabajo adicional significativo.
   - **Citas:** *Bullmore & Sporns 2009* (PDF p. 3: motivos locales).

### 8) **✅ Robustez y "lesiones" virtuales**
   - **QUÉ HACER:** Simular remoción de nodos/enlaces y medir impacto en eficiencia/sincronización.
   - **CÓDIGO DISPONIBLE:**
     - Red funcional construible (objetivo 2).
     - Dinámica Kuramoto en `KLayer` permite simular lesiones eliminando conexiones en `connectivity`.
   - **IMPLEMENTACIÓN:** Remover nodos de alta centralidad, recalcular eficiencia/R(t), comparar con remoción aleatoria.
   - **Citas:** *Rubinov & Sporns 2010* (PDF p. 7: resiliencia y lesiones virtuales).

### 9) **✅ Buenas prácticas de validación**
   - **QUÉ HACER:** Comparar métricas contra redes nulas (aleatorias preservando grado) y reportar estabilidad a umbrales.
   - **CÓDIGO DISPONIBLE:**
     - Todas las métricas anteriores.
   - **IMPLEMENTACIÓN:** Generar redes aleatorias con networkx (`nx.random_reference()`), repetir análisis, reportar distribuciones.
   - **Citas:** *Rubinov & Sporns 2010* (PDF p. 3: redes nulas y densidad).

---

## 📊 RESUMEN DE VIABILIDAD

| Objetivo | Alcanzable | Esfuerzo | Prioridad |
|----------|-----------|----------|-----------|
| 1. Estado crítico | ✅ SÍ | Bajo | ⭐⭐⭐ |
| 2. Red funcional | ✅ SÍ | Bajo | ⭐⭐⭐ |
| 3. Segregación | ✅ SÍ | Medio | ⭐⭐⭐ |
| 4. Integración | ✅ SÍ | Bajo | ⭐⭐⭐ |
| 5. Small-world | ✅ SÍ | Bajo | ⭐⭐⭐ |
| 6. Hubs/roles | ⚠️ PARCIAL | Medio | ⭐⭐ |
| 7. Motifs | ❌ NO | Alto | ⭐ |
| 8. Robustez | ✅ SÍ | Medio | ⭐⭐ |
| 9. Validación | ✅ SÍ | Medio | ⭐⭐⭐ |

---

## 🎯 PLAN DE IMPLEMENTACIÓN RECOMENDADO

### Fase 1: Fundamentos (2-3 semanas)
1. **Estado crítico:** DFA, PSD, distribuciones de avalanchas de sincronización.
2. **Red funcional:** Matriz de correlación píxel-a-píxel, umbralización adaptativa.
3. **Validación:** Redes nulas y estabilidad a umbrales.

### Fase 2: Topología (2-3 semanas)
4. **Segregación:** Clustering, detección de comunidades (Louvain).
5. **Integración:** Eficiencia global, longitud de camino.
6. **Small-world:** Comparación con aleatorias, cálculo de σ.

### Fase 3: Análisis avanzado (2-4 semanas)
7. **Hubs/roles:** Centralidades, z-score intra-módulo, participation coefficient.
8. **Robustez:** Lesiones virtuales, curvas de resiliencia.

### Fase 4 (opcional): Motifs
9. **Motifs:** Implementar o usar graph-tool para conteo de motifs 3-nodos.