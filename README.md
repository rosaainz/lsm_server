# LSM Server


## 📃 Descripción

LSM Server es una aplicación diseñada para el procesamiento de imágenes enviadas desde una aplicación móvil. El objetivo principal es detectar y reconocer señas de la Lengua de Señas Mexicana (LSM) utilizando modelos de aprendizaje automático.

## 📚 Características

- Recepción de imágenes desde la aplicación móvil.
- Procesamiento de imágenes para la detección de señas.
- Uso de modelos de clasificación entrenados para interpretar las señas.
- Respuesta con los resultados de la detección a la aplicación móvil.

## 🖥️ Tecnologías Utilizadas

- **Python**: Lenguaje principal para la lógica del servidor.
- **Flask**: Framework web ligero para manejar solicitudes HTTP.
- **OpenCV**: Librería para el procesamiento de imágenes.
- **scikit-learn**: Utilizado para cargar y aplicar los modelos de clasificación.
- **Mediapipe**: Utilizado para el preprocesamiento de imágenes y detección de características.

## 🧬 Estructura del Proyecto

```plaintext
.
├── models                 # Carpeta que contiene los modelos entrenados (.pkl)
├── src                    # Código fuente de la aplicación
│   └── app.py             # Archivo principal de la aplicación Flask   
├── uploads                # Carpeta para almacenar imágenes recibidas
└── README.md              # Documentación del proyecto
```
## ⚙️ Instalación
Para instalar y configurar el servidor, sigue estos pasos:
1. Clona este repositorio:
    ```sh
        git clone https://github.com/rosaainz/lsm_server.git
    ```
2. Navega al directorio del proyecto:
    ```sh
      cd lsm_server
    ```
3. Ejecuta la aplicación:
    ```sh
      python3 src/app.py
    ```

## 🤳 Uso
Para ejecutar el servidor, utiliza el siguiente comando:
    ```sh
        python src/app.py
    ```
El servidor estará disponible en http://localhost:4000.

##  🙌  Contribuir

¡Las contribuciones son bienvenidas! Para contribuir, sigue estos pasos:

1. Haz un fork del proyecto.
2. Crea una nueva rama (`git checkout -b feature/nueva-funcionalidad`).
3. Realiza los cambios necesarios y haz un commit (`git commit -am 'Añade nueva funcionalidad'`).
4. Haz push a la rama (`git push origin feature/nueva-funcionalidad`).
5. Abre un Pull Request.

## ⚖️ Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

## 🐚 Contacto

Si tienes alguna pregunta o sugerencia, por favor escribeme a través de www.linkedin.com/in/rosa-sainz-0b0b19212.







