The clinical diagnosis of Alzheimer’s disease represents a significant challenge due to the overlap of its symptoms with the cognitive changes associated with normal aging. Manifestations such as memory loss, reduced functional capacity, and early cognitive impairments may go unnoticed or be attributed to non-pathological factors, leading to delays in timely diagnosis and early intervention.

The dataset used in this project, obtained from Kaggle, contains detailed information from 2,149 patients, including demographic variables, lifestyle factors, medical history, clinical measurements, cognitive and functional assessments, symptoms, and Alzheimer’s disease diagnosis. The complexity and richness of this dataset pose the challenge of how to effectively integrate and analyze these variables in order to build reliable predictive models capable of distinguishing between patients with and without Alzheimer’s disease.

Furthermore, there is a need to compare different modeling approaches, including classical machine learning models (Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, XGBoost, Support Vector Machines, and K-Nearest Neighbors), deep neural networks, Bayesian Networks, and unsupervised techniques such as K-means. This comparison aims to evaluate their predictive performance, stability, and behavior under dimensionality reduction and hyperparameter optimization.

Therefore, the central problem of this project is to determine which machine learning and deep learning techniques are most effective for the diagnosis of Alzheimer’s disease, considering the complexity of the dataset, dimensionality reduction through Principal Component Analysis (PCA), and hyperparameter fine-tuning. The ultimate objective is to support clinical decision-making and improve the early detection of the disease.

# Technology Stack

The project was developed in **Jupyter Notebook**, using the **Python** programming language and a set of tools focused on data analysis, machine learning, deep learning, and scientific experimentation. The technology stack used in this project is described below.

### Programming Language and Development Environment
* **Python**
* **Jupyter Notebook**

### Data Acquisition and Management
* **KaggleHub**: dataset download and management from Kaggle  
* **Pandas**: manipulation and analysis of structured data  
* **NumPy**: numerical operations and handling of multidimensional arrays  

### Data Preprocessing and Transformation
* **Scikit-learn**:
  * `StandardScaler`: standardization of numerical variables  
  * `OneHotEncoder`: encoding of categorical variables  
  * `ColumnTransformer`: combined application of transformations  
  * `Pipeline`: chaining preprocessing and modeling stages  
  * `PCA (Principal Component Analysis)`: dimensionality reduction  

### Class Imbalance Handling
* **imbalanced-learn**:
  * `SMOTE`: synthetic generation of samples for minority classes  
  * `RandomUnderSampler`: controlled reduction of the majority class  

These techniques were applied in specific notebooks to evaluate the impact of class balancing on the performance of classification models.

### Modeling and Machine Learning
* **Supervised Models (Scikit-learn)**:
  * Logistic Regression  
  * Decision Trees  
  * Random Forest  
  * Gradient Boosting  
  * XGBoost  
  * Support Vector Machines (SVM)  
  * K-Nearest Neighbors (KNN)  
  * Gaussian Naive Bayes (probabilistic Bayesian model)  

* **Unsupervised Clustering**:
  * **K-Means**: identification of patterns and groupings without using class labels  

### Deep Learning
* **TensorFlow**
* **Keras**
  * Dense neural networks (MLP) for classification  

### Model Optimization and Evaluation
* **Scikit-learn**:
  * `GridSearchCV` and `ParameterGrid`: hyperparameter fine-tuning  

* **Evaluation Metrics**:
  * Accuracy  
  * Precision  
  * Recall  
  * F1-score  
  * Confusion Matrix  

### Experiment Tracking
* **MLflow**:
  * Metrics logging  
  * Hyperparameter logging  
  * Model storage and comparison  

### Data Visualization
* **Matplotlib**
* **Seaborn**
* **Plotly** (interactive visualizations)

# Architecture

The project follows a modular Machine Learning architecture, organized into independent Jupyter Notebooks according to the modeling approach and the stage of the workflow. This structure enables experimentation, model comparison, and reproducibility.

The overall system architecture consists of the following stages:

1. **Data Ingestion**  
   - Dataset acquisition from Kaggle using KaggleHub.  
   - Loading and initial inspection of clinical, demographic, and cognitive patient data.

2. **Exploratory Data Analysis (EDA)**  
   - Statistical and visual analysis of the variables.  
   - Study of distributions, correlations, and class imbalance.  
   - Identification of initial patterns and potential data inconsistencies.

3. **Preprocessing and Transformation**  
   - Data cleaning and consistency checks.  
   - Encoding of categorical variables using OneHotEncoder.  
   - Scaling of numerical variables with StandardScaler.  
   - Dimensionality reduction using PCA when applicable.

4. **Modeling**  
   - Implementation of supervised, probabilistic, unsupervised, and deep learning models.  
   - Model training using training and validation datasets.

5. **Evaluation and Testing**  
   - Performance evaluation using classification metrics.  
   - Testing with new samples (unseen data) to analyze generalization capability.

# Phases

### Phase 0: Exploratory Data Analysis (EDA)
- Understanding the dataset and its clinical, demographic, and cognitive variables.  
- Analysis of distributions, correlations, and outliers.  
- Identification of class imbalance between patients with and without Alzheimer’s disease.

### Phase 1: Data Cleaning, Transformation, and Feature Treatment
- Data cleaning and quality control.  
- Encoding of categorical variables.  
- Scaling of numerical features.  
- Application of dimensionality reduction techniques (PCA).  
- Application of class balancing techniques (SMOTE and RandomUnderSampler) in specific notebooks.

### Phase 2: Model Training
- Training of classical machine learning models.  
- Training of probabilistic models (Gaussian Naive Bayes).  
- Training of deep learning models (Neural Networks).  
- Hyperparameter fine-tuning using GridSearchCV and ParameterGrid.

### Phase 3: Testing and Evaluation with New Samples
- Evaluation of model performance on the test dataset.  
- Testing with new samples to simulate real-world scenarios.  
- Analysis of metrics such as Accuracy, Precision, Recall, and F1-score.  
- Comparison of model behavior and generalization capability.


  

  
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
