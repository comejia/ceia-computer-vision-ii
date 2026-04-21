# Detección de Anomalías en Audio usando Transfer Learning (CV2)

## Objetivo del Proyecto
Este proyecto desarrolla un prototipo funcional para detectar anomalías en sonidos de maquinaria industrial. Convierte archivos de audio en espectrogramas de Mel (representaciones visuales) y aplica **Transfer Learning** utilizando un modelo pre-entrenado de visión por computadora (ResNet18) para clasificar el estado de la máquina en dos categorías: `normal` o `anomalia`.

## Dataset
Se utiliza una versión simplificada del **MIMII Dataset** (Malfunctioning Industrial Machine Investigation and Inspection).
* **Enlace de descarga :** [MIMII Dataset - -6_dB_fan](https://zenodo.org/record/3384388/files/-6_dB_fan.zip).

## Instalación y Reproducibilidad
1. Clonar este repositorio.
   ```bash
   git clone https://github.com/carlosrivasa/indufan.git
   cd indufan
   ```

2. Crear un entorno virtual, activarlo e instalar las dependencias:
   - With uv (recommended):
   ```bash
   uv venv
   source .venv/bin/activate # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```
   - With pip:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Uso
ejecutar el notebook clasif_ventiladores.ipynb