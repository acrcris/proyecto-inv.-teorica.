# INSTRUCCIONES: Optimización de Parámetros - Fase Exploratoria

## 📋 Resumen

Este script realiza una **búsqueda secuencial** de parámetros óptimos para acercar el modelo de Kuramoto al estado crítico.

## 🎯 Objetivo

Encontrar parámetros que minimicen:

```
score = |R_final - 0.5| + 0.8×|DFA_α - 1.0| + 0.8×|PSD_slope + 1.0|
```

Donde:
- **R_final ≈ 0.5**: Estado crítico de sincronización
- **DFA α ≈ 1.0**: Correlaciones de largo alcance críticas (ruido rosa)
- **PSD slope ≈ -1.0**: Espectro 1/f (ruido rosa)

## ⚙️ Estrategia

**Búsqueda Secuencial (Greedy)**:
1. Optimizar `T_steps` (fijando el resto)
2. Optimizar `gamma` (usando mejor T_steps)
3. Optimizar `del_t` (usando mejores T_steps y gamma)
4. Optimizar `init_omg`
5. Optimizar `ksize`

**Total**: 25 evaluaciones (5 valores × 5 parámetros)

Cada evaluación usa 100 imágenes (10 por clase).

## ⏱️ Tiempo Estimado

- **Con GPU**: ~1 hora
- **Sin GPU**: ~3-5 horas

## 🚀 Ejecución

### Paso 1: Asegurar entorno

```bash
cd /home/crperezp/proyectos/ProyectoInvTeorica/Proyecto-Inv.-teorica./codigo/analisis_criticalidad_minimalista
```

### Paso 2: Verificar dependencias

El script usa las librerías ya instaladas:
- NumPy ✅
- SciPy ✅
- PyTorch (opcional, pero RECOMENDADO para GPU)

### Paso 3: Ejecutar

```bash
python3 optimizar_parametros_secuencial.py
```

## 📊 Resultados Generados

El script creará el directorio `resultados_optimizacion_secuencial/` con:

### 1. `resultados_optimizacion_secuencial.csv`

Tabla con todas las evaluaciones:

| T_steps | gamma | del_t | init_omg | ksize | R_final | DFA_α | PSD_slope | score | param_name | param_value |
|---------|-------|-------|----------|-------|---------|-------|-----------|-------|------------|-------------|
| 50      | 0.7   | 0.9   | 0.1      | 7     | 0.523   | 1.12  | -1.32     | 0.234 | T_steps    | 50          |
| ...     | ...   | ...   | ...      | ...   | ...     | ...   | ...       | ...   | ...        | ...         |

### 2. `mejor_configuracion.txt`

Archivo de texto con la mejor configuración encontrada:

```
MEJOR CONFIGURACIÓN ENCONTRADA
==================================================

Parámetros:
  T_steps: 75
  gamma: 0.5
  del_t: 0.7
  init_omg: 0.15
  ksize: 5

Métricas (validación con 500 imágenes):
  R_final: 0.4985
  DFA α: 1.0234
  PSD slope: -1.1245
  Score: 0.1845
```

## 📈 Interpretación de Resultados

### Score Actual (baseline):
```
Parámetros: T_steps=100, gamma=0.7, del_t=0.9, init_omg=0.1, ksize=7
Score: ~3.264
```

### Qué buscar:

- **Score < 1.0**: Mejora significativa 🎉
- **Score 1.0-2.0**: Mejora moderada 👍
- **Score > 2.5**: Mejora marginal ⚠️
- **Score > 3.0**: No hay mejora ❌

### Criterios de éxito:

| Métrica | Objetivo | Rango Aceptable |
|---------|----------|-----------------|
| R_final | 0.5000   | 0.40 - 0.60     |
| DFA α   | 1.0000   | 0.85 - 1.15     |
| PSD slope | -1.0000 | -1.30 a -0.70  |

## 🔄 Siguientes Pasos

### Si hay mejora (score < 2.0):

1. **Instalar PyTorch con CUDA** (si aún no está)
2. **Ejecutar Grid Search Reducido** (243 combinaciones)
3. **Validar con full test set** (10,002 imágenes)

### Si NO hay mejora (score > 2.5):

Posibles causas:
1. **Limitación arquitectónica**: El modelo de Kuramoto con 4 canales puede no ser suficiente
2. **Parámetros fuera de rango**: Expandir espacio de búsqueda
3. **Problemas fundamentales**: Considerar cambios en la arquitectura

## 🐛 Troubleshooting

### Error: "CUDA out of memory"

Reducir imágenes por evaluación:
```python
N_IMAGES_PER_CONFIG = 50  # En lugar de 100
```

### Error: "ModuleNotFoundError: No module named 'torch'"

El script funcionará en CPU si PyTorch no está instalado, pero será más lento.

### Proceso muy lento (> 5 horas)

Verificar:
1. ¿Está usando GPU? → Debería decir "GPU detectada"
2. Si no hay GPU, considerar:
   - Reducir imágenes: `N_IMAGES_PER_CONFIG = 50`
   - Reducir valores por parámetro: 3 en lugar de 5

## 📞 Monitoreo

Durante la ejecución verás:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║               OPTIMIZACIÓN SECUENCIAL DE PARÁMETROS                          ║
║                    Fase Exploratoria - Grid Search                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

✅ GPU detectada: NVIDIA Quadro P1000
   Memoria disponible: 4.0 GB

================================================================================
📊 Optimizando: T_steps
================================================================================

   Evaluando: T_steps=50, gamma=0.70, del_t=0.90, init_omg=0.10, ksize=7
      Procesando: 100%|██████████| 100/100 [00:52<00:00,  1.91it/s]
      Score: 2.145 | R=0.587 | DFA=1.42 | PSD=-2.87

   Evaluando: T_steps=75, gamma=0.70, del_t=0.90, init_omg=0.10, ksize=7
      ...
```

## ⚠️ Importante

- **NO interrumpir** durante una evaluación (esperar que termine el subset de 100 imágenes)
- Los resultados se van guardando en CSV después de cada parámetro optimizado
- Si se interrumpe, puedes reiniciar (no hay checkpointing automático en esta versión)

## 📚 Referencias

Ver documentación completa en:
- `ANALISIS_COMPLETO_TODAS_METRICAS.md`: Análisis de estado actual
- `PLAN_DE_ACCION.md`: Plan general del proyecto

---

**Fecha de creación**: Octubre 21, 2025  
**Versión**: 1.0 - Fase Exploratoria
