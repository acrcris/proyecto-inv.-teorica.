# 🚀 QUICK START - EJECUTAR EN OTRA MÁQUINA

## ⚡ INICIO RÁPIDO

```bash
# 1. Clonar repositorio
git clone https://github.com/ACRCris/Proyecto-Inv.-teorica..git
cd Proyecto-Inv.-teorica./codigo/analisis_criticalidad_minimalista

# 2. Crear entorno virtual e instalar dependencias
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio numpy scipy matplotlib tqdm pandas seaborn

# 3. Dar permisos a scripts
chmod +x ver_progreso_TRAIN.sh ver_progreso.sh

# 4. Ejecutar análisis Training Set (60k imágenes)
nohup python3 run_kuramoto_TRAIN_full.py > kuramoto_train.log 2>&1 &
echo $! > kuramoto_train.pid

# 5. Monitorear progreso
./ver_progreso_TRAIN.sh
```

## 📊 OPCIONES DISPONIBLES

### Opción 1: Test Set (10k imágenes) - Rápido
```bash
nohup python3 run_kuramoto_full_dataset.py > kuramoto_test.log 2>&1 &
echo $! > kuramoto_test.pid
./ver_progreso.sh
```
- **Tiempo**: ~40 minutos
- **Espacio**: ~500 MB
- **Por clase**: ~1,000 imágenes

### Opción 2: Training Set (60k imágenes) - Completo [RECOMENDADO]
```bash
nohup python3 run_kuramoto_TRAIN_full.py > kuramoto_train.log 2>&1 &
echo $! > kuramoto_train.pid
./ver_progreso_TRAIN.sh
```
- **Tiempo**: ~3.7 horas
- **Espacio**: ~3 GB
- **Por clase**: ~6,000 imágenes
- **Robustez**: Máxima (poder estadístico >99%)

## 📚 DOCUMENTACIÓN COMPLETA

Lee estos archivos para información detallada:

1. **`SETUP_OTRA_MAQUINA.md`** ← EMPIEZA AQUÍ
   - Prerequisitos detallados
   - Instalación paso a paso
   - Troubleshooting completo

2. **`INSTRUCCIONES_TRAIN_60k.md`**
   - Guía completa de ejecución
   - Análisis post-procesamiento
   - Valor científico

3. **`RESUMEN_OPCION_C.md`**
   - Resumen ejecutivo
   - Comparación test vs train
   - Cronograma

## ✅ PREREQUISITOS MÍNIMOS

- Linux (Ubuntu 20.04+)
- Python 3.8+
- GPU NVIDIA con CUDA (recomendado)
- 16 GB RAM
- 15 GB espacio en disco

## 🆘 AYUDA

Si tienes problemas:
1. Lee `SETUP_OTRA_MAQUINA.md` sección troubleshooting
2. Revisa logs: `tail -100 kuramoto_train.log`
3. Verifica GPU: `nvidia-smi`

## 📈 RESULTADOS ESPERADOS

Al finalizar tendrás:
- ✅ 60,000 imágenes procesadas
- ✅ ~6,000 por clase
- ✅ 6,060,000 observaciones temporales
- ✅ Análisis estadístico robusto
- ✅ Visualizaciones de alta calidad
- ✅ Resultados listos para publicación

---

**Última actualización**: Octubre 20, 2025  
**Commits**: `8bf6bfc1`, `2f1be83b`  
**Repositorio**: https://github.com/ACRCris/Proyecto-Inv.-teorica.
