# Alzheimer's Disease Diagnosis Using Machine Learning
The clinical diagnosis of Alzheimer’s disease represents a significant challenge due to the overlap of its symptoms with the cognitive changes associated with normal aging. Manifestations such as memory loss, reduced functional capacity, and early cognitive impairments may go unnoticed or be attributed to non-pathological factors, leading to delays in timely diagnosis and early intervention.

The dataset used in this project, obtained from Kaggle, contains detailed information from 2,149 patients, including demographic variables, lifestyle factors, medical history, clinical measurements, cognitive and functional assessments, symptoms, and Alzheimer’s disease diagnosis. The complexity and richness of this dataset pose the challenge of how to effectively integrate and analyze these variables in order to build reliable predictive models capable of distinguishing between patients with and without Alzheimer’s disease.

Furthermore, there is a need to compare different modeling approaches, including classical machine learning models (Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, XGBoost, Support Vector Machines, and K-Nearest Neighbors), deep neural networks, Bayesian Networks, and unsupervised techniques such as K-means. This comparison aims to evaluate their predictive performance, stability, and behavior under dimensionality reduction and hyperparameter optimization.

Therefore, the central problem of this project is to determine which machine learning and deep learning techniques are most effective for the diagnosis of Alzheimer’s disease, considering the complexity of the dataset, dimensionality reduction through Principal Component Analysis (PCA), and hyperparameter fine-tuning. The ultimate objective is to support clinical decision-making and improve the early detection of the disease.

## Technology Stack

The project was developed in Jupyter Notebook, using the Python programming language and a set of tools focused on data analysis, machine learning, deep learning, and scientific experimentation. The technology stack used in this project is described below.

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

## Architecture

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

## Phases

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

## What Was Achieved?

A set of machine learning and deep learning models was successfully developed and evaluated for the diagnosis of Alzheimer’s disease using a real clinical dataset that includes multiple demographic, cognitive, and lifestyle variables.

An exploratory data analysis (EDA) was conducted to understand the structure of the dataset, identify relevant patterns, and highlight the presence of class imbalance. Subsequently, data cleaning, transformation, and dimensionality reduction using PCA were applied, and their impact on model performance was evaluated.

Different modeling approaches were trained and compared, including classical machine learning models, probabilistic models (Naive Bayes), unsupervised techniques (K-Means), and neural networks, using evaluation metrics such as Accuracy, Precision, Recall, and F1-score. The results showed that tree-based and ensemble models (Random Forest, Gradient Boosting, and XGBoost) achieved the best performance, particularly when trained without applying PCA, while linear models exhibited more stable behavior when dimensionality reduction was applied.

Additionally, the generalization capability of the models was assessed through testing with unseen data (new samples), validating their applicability in scenarios closer to real clinical environments. Overall, the project identified the most effective techniques for this problem and provided a solid foundation to support the early detection of Alzheimer’s disease through computational methods.
## Screenshots

<img width="1690" height="1030" alt="Image" src="https://github.com/user-attachments/assets/7da27f6c-dca0-42ae-91eb-96ace87b33d8" />
Image showing the percentage distribution of the target variable, indicating the proportion of patients with and without an Alzheimer’s disease diagnosis in the dataset.

<img width="1147" height="1005" alt="Image" src="https://github.com/user-attachments/assets/af4a2133-2538-4dd5-9f8a-37f7f28e86fe" />
Correlation matrix 

<img width="1920" height="1200" alt="Image" src="https://github.com/user-attachments/assets/4ec2804c-e2d3-46f6-9013-8aca7182b4cd" />
This image shows the training of multiple models, where experiments were conducted using PCA with 95% explained variance, PCA reduced to two dimensions (2D), and without applying PCA.

<img width="1907" height="1006" alt="Image" src="https://github.com/user-attachments/assets/f4dde360-1adb-4615-93da-67242f06a8f7" />
Image showing the integration with MLflow and the comparison of three models after applying fine-tuning techniques.

<img width="1693" height="1027" alt="Image" src="https://github.com/user-attachments/assets/13803589-9f5f-4f94-87b1-017f50dcca96" />
Prediction probabilities obtained by both models when evaluating new patients, showing the confidence level associated with each diagnosis.

## Quantifiable Results

Based on the implementation and evaluation of various machine learning and deep learning models, the following quantifiable results were obtained. These results allow for an objective comparison of each model's performance using standard metrics such as Accuracy, Precision, Recall, and F1-score, both with and without the application of dimensionality reduction using PCA.

| Model               | PCA | Accuracy | Precision | Recall | F1-score |
|--------------------|-----|----------|-----------|--------|----------|
| Logistic Regression | Yes | 0.8093 | 0.7940 | 0.7824 | 0.7874 |
| Logistic Regression | No  | 0.8326 | 0.8183 | 0.8124 | 0.8151 |
| Decision Tree       | Yes | 0.7767 | 0.7558 | 0.7543 | 0.7550 |
| Decision Tree       | No  | 0.9326 | 0.9257 | 0.9270 | 0.9263 |
| Random Forest       | Yes | 0.8302 | 0.8365 | 0.7852 | 0.8009 |
| Random Forest       | No  | 0.9349 | 0.9369 | 0.9198 | 0.9274 |
| Gradient Boosting   | Yes | 0.8395 | 0.8389 | 0.8028 | 0.8156 |
| Gradient Boosting   | No  | 0.9442 | 0.9389 | 0.9389 | 0.9389 |
| XGBoost             | Yes | 0.8326 | 0.8268 | 0.7989 | 0.8094 |
| XGBoost             | No  | 0.9488 | 0.9480 | 0.9396 | 0.9435 |
| SVM                 | Yes | 0.8140 | 0.8000 | 0.7860 | 0.7919 |
| SVM                 | No  | 0.8209 | 0.8037 | 0.8063 | 0.8050 |
| KNN                 | Yes | 0.7372 | 0.7198 | 0.6775 | 0.6865 |
| KNN                 | No  | 0.7558 | 0.7463 | 0.6964 | 0.7073 |
| Neural Network      | Yes | 0.7883 | 0.7699 | 0.7602 | 0.7644 |
| Neural Network      | No  | 0.8488 | 0.7843 | 0.7894 | 0.7868 |
| K-Means Clustering  | Yes | 0.6465 | 0.4180 | 0.6465 | 0.5077 |
| Naive Bayes         | Yes | 0.8255 | 0.8233 | 0.7860 | 0.7987 |

<img width="4170" height="1766" alt="Image" src="https://github.com/user-attachments/assets/ca663dda-69ae-4e9c-8751-e08e57090b58" />

### Dataset
[Kaggle – Alzheimer’s Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)

