[Español](README-es.md) | [English](README.md)

# BeatFinder 🎧: Clasificación de Géneros Musicales con Redes Neuronales
**BeatFinder** es el proyecto final del Bootcamp de Data Science & Machine Learning en 4Geeks Academy. Su objetivo principal es clasificar automáticamente grabaciones de audio en uno de sus 16 géneros principales utilizando las características acústicas de la señal y un modelo de Red Neuronal Profunda (DNN).

## 🎯 Objetivo del Proyecto
El objetivo principal de este proyecto es desarrollar un modelo de clasificación capaz de identificar el género musical de una pista basándose únicamente en sus propiedades acústicas.

La capacidad de realizar esta tarea de forma automática resulta valiosa en 
contextos como:
1. La **optimización** de sistemas de recomendación musical,
2. La **organización** eficiente de grandes bibliotecas digitales, 
3. El **análisis** de tendencias en la industria musical.


Para esto hemos realizado lo siguiente:
- **Extracción de Características (Feature Extraction)**: Utilizando Librosa, transformamos las señales de audio (representadas por el dataset [FMA](https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-small-medium) en características numéricas clave (MFCCs, Centroide Espectral, Tonalidad, etc.).

- **Preprocesamiento y Exploración (EDA)**: Limpiamos, analizamos y estandarizamos los datos, comparando el impacto de la detección y acotamiento de outliers.

- **Modelado**: Diseñamos y optimizamos una Red Neuronal Densa (DNN) para maximizar la precisión en la identificación de géneros tras explorar otros modelos así como Random Forest, SVM o Regresión Logísitca.

*Finalmente el modelo tiene 49,133 filas y 518 variables pudiendo predecir hasta 16 géneros distintos.*

## 🧠 Tecnología y Herramientas
| Categoría | Herramientas Clave |
| :--- | :--- |
| **Modelado & ML** | **TensorFlow/Keras** (Redes Neuronales), **Scikit-learn** (Random Forest, SVM), **Keras Tuner** (Optimización de Hiperparámetros). |
| **Análisis de Audio** | **Librosa** (Extracción de MFCCs, ZCR, Tonnetz, etc.) |
| **Lenguaje** | **Python, Streamlit** |
| **Gestión de Datos** | **Pandas, NumPy** |
| **Visualización** | **Seaborn, Matplotlib** |
| **Ingeniería de ML** | Guardado de **scalers** (`.pkl`) y **modelos**. |

## 📊 Exploración y Preprocesamiento de Datos (EDA)
El proyecto se basó en el dataset FMA (Free Music Archive), que contiene metadatos y características pre-extraídas.

Desafíos Superados:

- **Limpieza de Columnas**: Se manejó la compleja jerarquía de tres niveles en las columnas de características y se convirtió la información de género de strings a listas.

- **Selección del Dataset Ganador**: Se evaluaron 6 combinaciones de datos (Con/Sin Outliers, Crudos/Normalizados/MinMax) para determinar cuál ofrecía la mejor Accuracy inicial, encontrando que [Menciona el dataset ganador aquí, ej: X_train_sin_outliers_norm] fue el más efectivo para la clasificación.

- **Gestión del Desequilibrio**: El análisis univariante reveló un marcado desequilibrio de clases (predominio de Electronic, Rock, Experimental), un factor crítico a considerar en la evaluación del rendimiento. Por ello y por falta de datos decidimos mantener los géneros musicales más representativos.

- **Gestión de la RAM**: Para realizar nuestro trabajo nos hemos valido de Google Colab y GitHub Codespaces y pronto nos quedamos sin RAM debido a la gran cantidad de datos que el ordenador debía procesar. Para ello ajustamos los datatypes y eliminamos todas aquellas variables que dejamos de usar.

## 🏆 Resultados del Modelo
Se compararon cuatro modelos principales. El mejor rendimiento se logró tras la optimización de hiperparámetros de la Red Neuronal mediante Keras Tuner utilizando el mejor dataset seleccionado.

| Modelo | Optimización | Accuracy en Test Set |
| :--- | :--- | :--- |
| **Red Neuronal (DNN)** | Keras Tuner | 68'66% |
| Random Forest | RandomizedSearchCV | % |
| SVM (RBF/Lineal) | RandomizedSearchCV (Subset) | % |
| Regresión Logística | Base | % |


🥇 El modelo ganador fue la Red Neuronal con una precisión final entorno del 68%.

## ⚙️ Estructura del Repositorio
```
├── music_genre_identifier/
│   ├── data/
│   │   ├── processed/ (Contiene .parquet de X_train/test normalizados/minmaxed)
│   └── src/
│       ├── factorized_data/ (Contiene factorized_genre_top.json para decodificación)
│       └── models/ (Contiene el scaler y los modelos .pkl para despliegue en Streamlit)
├── models/
│   └── neural_network.pkl (El objeto del modelo ganador final)
└── README.md
```

## 🧑‍💻 Co-creadores
Este proyecto fue desarrollado en colaboración por:

[Daniel Páez](https://github.com/danielpaez-dev) | [Ivan Díaz](https://github.com/ivandla96) | [Tulio Giménez](https://github.com/TulioGimenez)
