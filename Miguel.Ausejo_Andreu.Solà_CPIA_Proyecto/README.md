# Proyecto CPIA: Detección de Transacciones Fraudulentas

**Autores:** Miguel Ausejo Gallego, Andreu Solà i Dagas  
**Fecha:** Diciembre 2025

## 1. Introducción

Bienvenidos a nuestro proyecto final para la asignatura de CPIA. El objetivo principal de este trabajo es diseñar, implementar y evaluar un sistema capaz de detectar transacciones bancarias fraudulentas utilizando técnicas de Machine Learning y Deep Learning.

Trabajamos con un *dataset* de transacciones anonimizadas (Credit Card Fraud Detection) que presenta un fuerte desbalance de clases: la gran mayoría de operaciones son legítimas, mientras que los fraudes representan una pequeña minoría. Nuestro reto ha sido crear modelos capaces de identificar estos casos anómalos maximizando el **Recall** (sensibilidad), para asegurar que detectamos la mayor cantidad de fraudes posible.

## 2. Estructura del Repositorio

En esta carpeta encontrarán todos los recursos necesarios para reproducir nuestros experimentos y entender el desarrollo del proyecto:

*   **`Miguel.Ausejo_Andreu.Solà_CPIA_Proyecto.pdf`**: La memoria escrita completa del proyecto. Recomendamos leerla para entender en profundidad el contexto teórico, la metodología y el análisis detallado de los resultados.
*   **Notebooks de Modelos (.ipynb)**:
    *   **`RF+MLP.ipynb`**: Contiene la implementación, entrenamiento y validación de los modelos **Random Forest** y **Multi-Layer Perceptron (MLP)**. Incluye la optimización de hiperparámetros con Optuna.
    *   **`Autoencoder.ipynb`**: Implementación del modelo **Autoencoder** (aprendizaje no supervisado/semi-supervisado) para la detección de anomalías basada en el error de reconstrucción.
    *   **`TabNet.ipynb`**: Implementación del modelo **TabNet**, una arquitectura de Deep Learning especializada en datos tabulares, que resultó ser nuestro mejor modelo.
*   **`hf_deploy.zip`**: Archivo comprimido que contiene todos los ficheros necesarios para el despliegue de la aplicación web en Hugging Face (Dockerfile, app.py, pesos del modelo, etc.).

## 3. Descripción de los Modelos

Hemos explorado cuatro enfoques distintos, desde algoritmos clásicos hasta redes neuronales profundas:

1.  **Random Forest (RF):** Un ensamblaje de árboles de decisión optimizado con Optuna. Logró un buen equilibrio y robustez gracias a su naturaleza de votación por mayoría.
    *   *Recall obtenido:* 0.853
2.  **Multi-Layer Perceptron (MLP):** Una red neuronal clásica implementada con `scikit-learn`. Aunque efectiva, fue superada por el Random Forest en este caso específico.
    *   *Recall obtenido:* 0.709
3.  **Autoencoder:** Un enfoque basado en la reconstrucción de datos. La hipótesis es que el modelo aprenderá a reconstruir transacciones normales y fallará con las fraudulentas. Sin embargo, debido a la sutileza de algunos fraudes, su rendimiento fue limitado.
    *   *Recall obtenido (con estrategia seleccionada):* 0.5226
4.  **TabNet:** Nuestro modelo estrella. Combina la interpretabilidad de los árboles de decisión con la potencia de aprendizaje de las redes neuronales profundas. Utiliza mecanismos de atención para seleccionar *features* relevantes en cada paso.
    *   *Recall obtenido:* **0.9431** (Threshold 0.3)

## 4. Despliegue (Deployment)

Como parte final del proyecto, hemos contenerizado nuestro mejor modelo (**TabNet**) y lo hemos desplegado como una aplicación web interactiva utilizando **Docker**, **Streamlit** y **Hugging Face Spaces**.

El archivo `hf_deploy.zip` contiene:
*   `app.py`: La aplicación de Streamlit que sirve la interfaz web.
*   `Dockerfile`: Configuración para construir la imagen del contenedor.
*   `requirements.txt`: Lista de dependencias necesarias.
*   `tabnet_model.zip`: Los pesos entrenados del modelo.
*   `scaler.pkl`: El objeto `RobustScaler` ajustado para normalizar las nuevas entradas igual que en el entrenamiento.
*   `model_columns.pkl`: Lista de columnas para asegurar la consistencia en la entrada de datos.

Pueden probar la aplicación en vivo aquí:
👉 **[Fraud Detector Specialist en Hugging Face](https://huggingface.co/spaces/aandreeeuu/Fraud-Detector-Specialist)**

## 5. Requisitos e Instalación

Para ejecutar los notebooks localmente, hemos utilizado **Python 3.13** y el entorno de **VS Code**. Las principales librerías que necesitan instalar son:

*   `pandas` y `numpy` (Manipulación de datos)
*   `scikit-learn` (Modelos clásicos y preprocesamiento)
*   `torch` (PyTorch, base para TabNet y Autoencoder)
*   `pytorch_tabnet` (Librería específica para TabNet)
*   `optuna` (Optimización bayesiana de hiperparámetros)
*   `imblearn` (Técnicas de balanceo como SMOTE)
*   `matplotlib` y `seaborn` (Visualización)

## 6. Ejecución

Recomendamos ejecutar los notebooks en el orden en que se presentan los modelos en la memoria si se desea seguir la narrativa del aprendizaje, aunque son independientes entre sí:
1.  `RF+MLP.ipynb`
2.  `Autoencoder.ipynb`
3.  `TabNet.ipynb`

¡Esperamos que encuentren interesante nuestro trabajo sobre la detección de fraude!
