# Usa una imagen base oficial de Python
FROM python:3.11-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo requirements.txt al contenedor
COPY requirements.txt .
COPY models /app/models

# Instala las dependencias del sistema, incluidas las necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instala las dependencias de Python especificadas en el archivo requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código fuente de la aplicación al contenedor
COPY src/ /app/src/

# Expone el puerto que usará la aplicación (Railway usa el puerto 8080)
EXPOSE 8080
# Comando para ejecutar la aplicación usando gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "src.app:app"]
