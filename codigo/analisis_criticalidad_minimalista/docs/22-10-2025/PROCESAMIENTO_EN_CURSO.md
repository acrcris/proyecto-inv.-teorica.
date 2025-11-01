# 🚀 PROCESAMIENTO EN CURSO - 60K IMÁGENES

**Fecha inicio:** 20 octubre 2025, 15:32:51  
**PID:** 49519  
**Estado:** ✅ EJECUTÁNDOSE

---

## 📊 Información del Proceso

- **Dataset:** MNIST Training Set (60,000 imágenes)
- **Velocidad actual:** 6.69 img/s (mejor que estimado: 3.5 img/s)
- **Tiempo estimado:** ~2.5 horas (mejorado de 4.8h)
- **Progreso inicial:** 60/60,000 (0.1%)
- **Checkpoints:** Cada 100 imágenes (600 checkpoints totales)
- **Directorio:** `resultados_kuramoto_TRAIN_MAC_60k/`

---

## 🔍 Comandos de Monitoreo

### 1. Ver progreso en tiempo real
```bash
tail -f kuramoto_train_mac.log
```

### 2. Ver progreso resumido
```bash
./ver_progreso_TRAIN_MAC.sh
```

### 3. Verificar que el proceso está corriendo
```bash
ps -p $(cat kuramoto_train_mac.pid)
```

### 4. Ver últimas 50 líneas del log
```bash
tail -50 kuramoto_train_mac.log
```

### 5. Contar checkpoints completados
```bash
ls resultados_kuramoto_TRAIN_MAC_60k/checkpoints/ | wc -l
```

---

## 📈 Métricas Calculadas

Por cada imagen se calculan:

1. ✅ **R_stationary** - Parámetro de orden estacionario
2. ✅ **mag_channel_mean_final** - Magnitud media por canal (4 valores)
3. ✅ **PSD_slope** - Pendiente del espectro de potencia
4. ✅ **DFA_alpha** - Exponente DFA
5. ✅ **shannon_entropy_by_channel** - Entropía Shannon por canal (4 valores)
6. ✅ **mutual_info** - Información mutua (CORREGIDO)

---

## ⏱️ Estimación de Tiempo

| Progreso | Tiempo | Checkpoint |
|----------|--------|------------|
| 10% (6,000) | ~15 min | checkpoint 60 |
| 25% (15,000) | ~37 min | checkpoint 150 |
| 50% (30,000) | ~1.2 h | checkpoint 300 |
| 75% (45,000) | ~1.9 h | checkpoint 450 |
| 100% (60,000) | ~2.5 h | checkpoint 600 |

---

## 🛑 Detener el Proceso (si es necesario)

```bash
# Leer el PID
PID=$(cat kuramoto_train_mac.pid)

# Detener el proceso
kill $PID

# O forzar detención
kill -9 $PID
```

**Nota:** Los checkpoints se guardan cada 100 imágenes, así que puedes reanudar desde el último checkpoint.

---

## ✅ Al Finalizar

Cuando el proceso termine (2.5 horas aproximadamente):

1. **Verificar finalización:**
   ```bash
   tail -100 kuramoto_train_mac.log
   ```

2. **Ejecutar análisis por clase:**
   ```bash
   python3 analizar_metricas_por_clase_mac.py
   ```

3. **Generar visualizaciones:**
   ```bash
   # El script de análisis generará automáticamente distribuciones
   ```

---

## 📂 Archivos Generados

- **Log:** `kuramoto_train_mac.log`
- **PID:** `kuramoto_train_mac.pid`
- **Checkpoints:** `resultados_kuramoto_TRAIN_MAC_60k/checkpoints/checkpoint_*.pt`
- **Resultado final:** `resultados_kuramoto_TRAIN_MAC_60k/metricas_train_60k_MAC.pt`

---

## 🔥 Tips

- No cierres la terminal (usa `screen` o `tmux` si necesitas desconectar)
- Revisa el progreso cada ~30 minutos con `./ver_progreso_TRAIN_MAC.sh`
- Si hay un error, revisa el log completo: `less kuramoto_train_mac.log`
- El proceso usa GPU Apple M3 (MPS) automáticamente

---

**Estado actual:** ✅ Corriendo sin errores
**Siguiente revisión sugerida:** En 30 minutos (aprox. 12,000 imágenes procesadas)
