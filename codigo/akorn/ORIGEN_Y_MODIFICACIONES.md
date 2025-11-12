# AKOrN - Artificial Kuramoto Oscillatory Neurons

## 📍 Origen del Código

Este directorio contiene código del paper **AKOrN** (ICLR 2025):
- **Paper**: https://arxiv.org/abs/2410.13821
- **Repositorio original**: [Agregar URL del repo original de AKOrN aquí]
- **Autores**: [Autores del paper original]
- **Licencia**: [Licencia del repo original - verificar]

## 🔧 Modificaciones Propias

Este fork/copia incluye las siguientes modificaciones para nuestro proyecto de investigación:

### 1. Soporte para Apple Silicon (MPS)
**Archivo**: `train_classification.py`
- **Líneas**: 26-47
- **Cambio**: Función `_resolve_device()` para auto-detectar MPS/CUDA/CPU
- **Razón**: Entrenar en Mac M1/M2 sin CUDA

### 2. Soporte para MNIST
**Archivo**: `train_classification.py`
- **Líneas**: 219-235
- **Cambio**: Carga de dataset MNIST con data augmentation
- **Razón**: Clasificar MNIST para correlacionar con análisis de criticalidad α_c

### 3. Scripts de Setup
**Archivos nuevos**:
- `setup_mnist.py` - Script para descargar MNIST
- `train_mnist.sh` - Script bash para entrenar con parámetros óptimos
- `train_mnist_colab.ipynb` - Notebook para entrenar en Google Colab
- `GUIA_MNIST.md` - Documentación de uso
- `GUIA_COLAB_GPU.md` - Guía para vscode-colab
- `INICIO_RAPIDO_VSCODE_COLAB.md` - Quick start

### 4. Documentación Adicional
**Archivos nuevos** (fuera de este directorio):
- `/ANALISIS_AKORN_MNIST.md` - Análisis técnico completo
- `/AKORN_RESUMEN_EJECUTIVO.md` - Resumen ejecutivo
- `/AKORN_ARQUITECTURA_VISUAL.md` - Arquitectura visual

## 🎓 Cómo Citar

Si usas este código en una publicación, cita el paper original:

```bibtex
@inproceedings{akorn2025,
  title={Artificial Kuramoto Oscillatory Neurons},
  author={[Autores]},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

Y referencia este proyecto:

```bibtex
@misc{proyecto-inv-teorica-2025,
  author={ACRCris},
  title={Análisis de Criticalidad en MNIST usando Dinámica de Kuramoto},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/ACRCris/Proyecto-Inv.-teorica.}}
}
```

## 📝 Licencia

**Código original**: [Licencia del repo original - MIT/Apache/etc.]  
**Modificaciones**: [Tu licencia - mantener compatible con original]

---

## 🔄 Actualizar desde Original

Si el repo original de AKOrN se actualiza:

1. Descargar último código original
2. Comparar con este directorio: `diff -r akorn_original/ akorn/`
3. Aplicar cambios manualmente preservando nuestras modificaciones
4. Documentar cambios en este README

---

## 📞 Contacto

- **Proyecto original AKOrN**: [Contacto autores originales]
- **Este fork/adaptación**: ACRCris - [Tu email/contacto]

---

**Última actualización**: Noviembre 2025  
**Versión AKOrN base**: [Verificar commit/tag del original]
