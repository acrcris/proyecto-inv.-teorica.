# Experimentos de Variación de C_scale en AKOrN

**Fecha:** 18 de noviembre de 2025  
**Objetivo:** Entrenar modelos AKOrN con diferentes valores de acoplamiento externo (C_scale) para estudiar el comportamiento en torno al punto de transición de fase.

---

## 📋 Contexto

Este documento describe los experimentos en curso para estudiar cómo varía el comportamiento del modelo AKOrN al modificar el parámetro de acoplamiento externo `c_scale`. Los valores de C_scale fueron derivados del análisis de la distribución de C_crítico a través de las clases de MNIST.

### Análisis Previo
- Se analizó la distribución de C_crítico para todas las clases de MNIST
- Se identificaron 26 centros de bins representativos en el rango [0.0062, 0.3162]
- Estos valores representan diferentes regímenes de sincronización en osciladores de Kuramoto

---

## 🎯 Estrategia de Experimentación

### Modificación del Modelo
Se añadió el parámetro `c_scale` al modelo AKOrN para controlar el acoplamiento externo:

**Archivos modificados:**
- `source/models/classification/knet.py`: 
  - Agregado `c_scale` como parámetro no entrenable
  - Modificado forward para escalar el acoplamiento: `c_scaled = c * self.c_scale`
  
- `train_classification.py`:
  - Agregado argumento `--c_scale` (default=1.0)
  - Integrado en la instanciación del modelo

### Valores de C_scale Seleccionados

**Estrategia:** Muestreo intercalado (uno sí, uno no) de los 26 valores originales

**14 valores finales:**
```
0.0062, 0.031, 0.0558, 0.0806, 0.1054, 0.1302, 0.155, 
0.1798, 0.2046, 0.2294, 0.2542, 0.279, 0.3038, 0.3162
```

**Justificación:**
- Incluye el mínimo (0.0062) y máximo (0.3162)
- Cobertura uniforme del rango de interés
- Reduce tiempo de experimento de 122h a ~77h

---

## ⚙️ Configuración de Entrenamiento

### Parámetros del Modelo
- **n:** 2 (dimensión del oscilador, compatible con OmegaLayer)
- **ch:** 64 (canales)
- **T:** 10 (pasos temporales, reducido de 30)
- **L:** 2 (capas, reducido de 3)
- **ksizes:** [7, 5]
- **gamma:** 0.7
- **Parámetros totales:** ~1.6M

### Hiperparámetros de Entrenamiento
- **Dataset:** MNIST
- **Épocas:** 15 (reducido de 50)
- **Batch size:** 128 (optimizado para MPS)
- **Learning rate:** 0.0001
- **Checkpoints:** cada 5 épocas
- **Device:** MPS (Apple Metal Performance Shaders)

### Optimizaciones para Apple Silicon (M3)
1. **T reducido de 30 a 10:** Reduce cómputo por iteración ~3x
2. **L reducido de 3 a 2:** Menos parámetros (~1.6M vs ~5.4M)
3. **Batch size optimizado a 128:** Balance entre throughput y uso de GPU
4. **Épocas reducidas a 15:** Suficiente para observar comportamiento

---

## 🔬 Pruebas de Rendimiento

### Hardware
- **Chip:** Apple M3 (8 cores)
- **RAM:** 16 GB
- **GPU:** MPS integrada

### Resultados de Benchmarking

#### Test 1: Proceso Individual
- **Configuración:** T=10, L=2, batch_size=128
- **Rendimiento:** ~2.36s por iteración
- **Tiempo por época:** ~18 minutos
- **Tiempo por experimento (15 épocas):** ~5.5 horas

#### Test 2: 3 Procesos en Paralelo
- **Rendimiento:** ~6.9s por iteración (promedio)
- **Penalización:** 2.9x más lento por proceso
- **Conclusión:** ❌ NO viable (MPS se satura)
- **Tiempo total:** ~75h (similar a secuencial)

#### Test 3: Configuración T=30, L=3
- **Rendimiento:** ~220s por iteración
- **Penalización:** 93x más lento que T=10, L=2
- **Conclusión:** ❌ Inviable para sweep completo

### Decisión Final
**Ejecutar 1 proceso a la vez con T=10, L=2, batch_size=128**

---

## 📊 Plan de Ejecución

### Script: `train_c_critico_half.sh`

**Características:**
- 14 experimentos secuenciales
- 1 experimento a la vez (máximo rendimiento)
- Logs individuales en `runs/c_critico_sweep_half/logs_c_scale_*.txt`
- Checkpoints en `runs/c_critico_sweep_half/c_scale_*/checkpoint_*.pth`

**Tiempo estimado total:** ~77 horas (~3.2 días)

### Estructura de Salida
```
runs/c_critico_sweep_half/
├── sweep_master.log          # Log principal del sweep
├── sweep_master.pid           # PID del proceso maestro
├── logs_c_scale_0.0062.txt   # Logs individuales
├── logs_c_scale_0.031.txt
├── ...
└── c_scale_0.0062/            # Directorios de experimentos
    ├── checkpoint_5.pth
    ├── checkpoint_10.pth
    ├── checkpoint_15.pth
    └── events.out.tfevents.*  # TensorBoard
```

---

## 📈 Análisis Esperado

### Métricas a Evaluar
1. **Accuracy vs C_scale:** ¿Cómo varía el rendimiento con el acoplamiento?
2. **Dinámica de entrenamiento:** Convergencia en diferentes regímenes
3. **Transición de fase:** Identificación de C_crítico efectivo
4. **Robustez:** Variabilidad del modelo en torno a la transición

### Visualizaciones Planeadas
- Curvas de accuracy vs C_scale
- Evolución de loss por experimento
- Comparación de dinámicas de osciladores
- Análisis de parámetro de orden de Kuramoto

---

## 🚀 Estado Actual

### Progreso
- ✅ Modificación del código completada
- ✅ Tests de rendimiento ejecutados
- ✅ Configuración optimizada
- ✅ Script de sweep preparado
- 🔄 **Experimento 1/14 en curso** (C_scale = 0.0062)

### Próximos Pasos
1. Completar sweep de 14 experimentos (~77h)
2. Analizar resultados y métricas
3. Generar visualizaciones
4. Documentar hallazgos sobre transición de fase
5. Evaluar necesidad de experimentos adicionales

---

## 📝 Notas Técnicas

### Limitaciones Identificadas
- **MPS no escala bien con múltiples procesos:** Mejor 1 proceso a la vez
- **T=30 inviable:** Demasiado lento para sweep completo
- **Batch size crítico:** 128 óptimo, 256 causa degradación

### Lecciones Aprendidas
1. Apple Silicon MPS tiene limitaciones para paralelización
2. Reducción de T y L es crucial para viabilidad
3. Checkpoints frecuentes importantes (experimentos largos)
4. Monitoreo continuo necesario para detectar fallos temprano

### Archivos de Configuración
- `train_c_critico_half.sh`: Script principal de ejecución
- `monitor_sweep.sh`: Script de monitoreo de progreso
- `representantes_por_bin_centros.csv`: Fuente de valores C_scale

---

## 🔗 Referencias

- Análisis de C_crítico: `codigo/analisis_criticalidad_minimalista/`
- Notebook de pruebas: `kuramoto_pruebas_basico.ipynb`
- Resultados de distribución: `analisis_metricas/`
- Código AKOrN: `source/models/classification/knet.py`

---

**Última actualización:** 18 de noviembre de 2025, 15:30
