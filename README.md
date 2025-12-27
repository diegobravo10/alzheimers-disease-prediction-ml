## Descripción del Problema
El diagnóstico clínico del Alzheimer suele ser complejo debido a la similitud de síntomas con el envejecimiento natural. Este proyecto utiliza Deep Learning para analizar patrones no lineales entre factores de riesgo (hipertensión, IMC, hábitos), métricas cognitivas (MMSE) y datos demográficos.

El flujo de trabajo incluye desde la limpieza de datos y reducción de dimensionalidad (PCA) hasta la optimización de hiperparámetros (Fine-Tuning) y el seguimiento de experimentos con MLOps.

## Stack Tecnológico

El proyecto fue desarrollado en Google Colab utilizando las siguientes tecnologías:

* **Procesamiento de Datos:** `Pandas`, `NumPy`.
* **Preprocesamiento:** `Scikit-learn` (StandardScaler, OneHotEncoder, PCA).
* **Modelado:** `TensorFlow`, `Keras`.
* **Optimización:** `Keras Tuner` (Hyperband Algorithm).
* **MLOps & Tracking:** `MLflow` (Registro de métricas, parámetros y modelos).
* **Visualización:** `Matplotlib`, `Seaborn`.

## Arquitectura y Fases

El desarrollo se dividió en 4 fases estratégicas:

1.  **Ingesta y EDA:** Análisis exploratorio, mapas de correlación y limpieza de identificadores.
2.  **Preprocesamiento:**
    * Transformación de variables (Estandarización y One-Hot Encoding).
    * **PCA (Principal Component Analysis):** Reducción de dimensionalidad conservando el 95% de la varianza.
3.  **Modelado y Optimización:**
    * *Modelo Base:* Arquitectura simple (RMSProp, Batch 16).
    * *Modelo Optimizado:* Arquitectura profunda (Adam, Batch 64, Dropout).
    * *Fine-Tuning:* Búsqueda automática de hiperparámetros.
4.  **Seguimiento con MLflow:**
    * Registro de experimentos para comparar *Loss* y *Accuracy*.
    * Comparativa de modelos (Base vs. Optimizado).
  
<img width="920" height="575" alt="Image" src="https://github.com/user-attachments/assets/16df07c6-487e-429d-8068-134d8402792f" />

<img width="886" height="603" alt="Image" src="https://github.com/user-attachments/assets/a7765e95-bbb7-4e7f-9533-26eae959bd56" />

<img width="886" height="688" alt="Image" src="https://github.com/user-attachments/assets/e194b535-a145-4e52-b8a3-8ba14add6093" />

## Resultados
Gracias a la optimización y el uso de **MLflow** para monitorear el desempeño, se lograron las siguientes mejoras:

| Métrica | Modelo Base | Modelo final |
| :--- | :---: | :---: |
| **Accuracy** | ~82% | **~79%** |
| **Recall (Sensibilidad)** | **~57%** | **~69%** |
| **F1-Score** | **~69%** | **~70%** |
