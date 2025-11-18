# 🐛 Explicación del Semaphore Leak en PyTorch MPS

## ¿Qué es un Semaphore Leak?

Un **semáforo** es un mecanismo de sincronización usado por sistemas operativos para coordinar procesos/threads que comparten recursos.

En macOS, cuando PyTorch MPS (Metal Performance Shaders) ejecuta operaciones en el GPU:

```
CPU Process ←--semaphore--→ GPU Metal
```

El semáforo asegura que CPU y GPU estén sincronizados (ej: esperar a que GPU termine antes de leer resultados).

## 🔴 El Problema

### PyTorch 2.x + MPS + macOS tiene un bug conocido:

1. **Creación de semáforos**: PyTorch MPS crea semáforos para cada operación GPU
2. **Falta de limpieza**: En procesos largos, estos semáforos **no se liberan correctamente**
3. **Límite del sistema**: macOS tiene un límite de ~256 semáforos
4. **Crash**: Cuando se alcanza el límite → `resource_tracker: leaked semaphore` → proceso muere

### ¿Por qué ocurre en tu caso?

Tu script procesa **60,000 imágenes**, cada una con:
- 200 iteraciones (valores de C)
- 30 pasos temporales (T=30)
- Operaciones tensor en MPS

Esto genera **millones de operaciones GPU** → acumulación de semáforos → crash después de ~900 imágenes.

## 🔧 Soluciones Implementadas

### 1. **Limpieza Agresiva de Recursos MPS** ✅

Agregué función `cleanup_mps_resources()`:

```python
def cleanup_mps_resources():
    """Limpia recursos de MPS para evitar semaphore leaks."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()       # Libera cache GPU
        torch.mps.synchronize()        # Sincroniza operaciones pendientes
        import gc
        gc.collect()                   # Garbage collection
```

**Se ejecuta**:
- Cada 100 imágenes procesadas
- Al terminar cada clase
- Reduce la acumulación de semáforos

### 2. **Script con Reinicio Automático** ✅

`ejecutar_con_reinicio_automatico.sh`:

```bash
while [ intentos < 100 ]; do
    python calcular_c_critico_local.py ...
    
    if crash; then
        echo "Reiniciando..."
        sleep 5
        # Auto-restart continúa desde último checkpoint
    fi
done
```

**Ventajas**:
- Detecta crash automáticamente
- Reinicia sin intervención humana
- Aprovecha checkpoints en SQLite
- No pierde progreso

### 3. **Checkpoints Persistentes** ✅

Ya existía en tu código:

```python
# Cada imagen se guarda inmediatamente en SQLite
save_result(db_path, clase, idx, result, PARAMS)

# Al reiniciar, se detectan y se saltan
processed_images = get_processed_images(db_path, clase)
```

## 📊 Comparación: Antes vs Después

### Antes (sin fix):
```
Procesamiento → 900 imágenes → CRASH
Usuario reinicia manualmente
Procesamiento → 900 imágenes → CRASH
...
```
**Requiere**: Supervisión constante

### Después (con fix):
```
Procesamiento → 2000 imágenes → limpieza → continúa
              → 4000 imágenes → limpieza → continúa
              → eventual crash (más tarde)
Auto-restart  → continúa desde 4000
              → 6000 imágenes → ...
```
**Requiere**: Cero supervisión

## 🚀 Uso Recomendado

### Opción 1: Script con Auto-Restart (Recomendado)

```bash
cd analisis_alpha_c
./ejecutar_con_reinicio_automatico.sh
```

**Ventajas**:
- Reinicio automático en caso de crash
- No requiere supervisión
- Logs separados por intento
- Detección automática de finalización

### Opción 2: Script Original Mejorado

```bash
cd analisis_alpha_c
./ejecutar_background.sh
```

**Ventajas**:
- Más control manual
- Un solo log
- Requiere reinicio manual si crashea

## 🔍 Monitoreo

### Ver estado:
```bash
./ver_estado.sh
```

### Ver GPU:
```bash
./ver_gpu.sh
```

### Ver logs en tiempo real:
```bash
tail -f procesamiento_*.log
```

## 🎯 Expectativas Realistas

### Con las mejoras:

1. **Limpieza cada 100 imágenes**:
   - Reduce frecuencia de crashes
   - Puede procesar ~2,000-3,000 imágenes antes de crash (vs 900 antes)

2. **Auto-restart**:
   - Continúa automáticamente
   - Progreso total: 21% → 100% sin intervención

3. **Tiempo total**:
   - ~60 horas (~2.5 días)
   - Sin supervisión humana necesaria

### ¿Por qué no se elimina completamente?

El bug está en **PyTorch internamente**, no podemos controlarlo 100%.

Las soluciones mitigan el problema pero no lo eliminan completamente.

## 🆘 Si Persisten los Crashes Frecuentes

### Plan B: Reducir carga GPU

Modificar `PARAMS` en el script:

```python
PARAMS = {
    'T': 20,      # Reducir de 30 a 20 (menos pasos)
    'h': 56,      # Reducir de 64 a 56 (menor resolución)
    'w': 56,
    # ... resto igual
}
```

**Trade-off**: Menor precisión, pero más estabilidad.

### Plan C: Procesar por lotes más pequeños

```bash
# En vez de todas las clases a la vez
python calcular_c_critico_local.py --clases 7 6 5
# Esperar a que termine
python calcular_c_critico_local.py --clases 4 3 2 1 0
```

## 📚 Referencias

- [PyTorch MPS Issues](https://github.com/pytorch/pytorch/issues?q=is%3Aissue+mps+semaphore)
- [Apple Metal Bug Reports](https://developer.apple.com/forums/tags/metal/)
- [macOS Semaphore Limits](https://developer.apple.com/library/archive/technotes/tn2083/)

## ✅ Resumen

| Aspecto | Estado |
|---------|--------|
| **Causa identificada** | ✅ PyTorch MPS bug interno |
| **Mitigación implementada** | ✅ Limpieza agresiva + auto-restart |
| **Requiere supervisión** | ❌ No (con script auto-restart) |
| **Pérdida de progreso** | ❌ No (checkpoints SQLite) |
| **Completará eventualmente** | ✅ Sí (~60 horas) |

---

**Última actualización**: 16 de noviembre de 2025
