# 🩷 Breast Cancer Prediction System (SVM Advanced)

A machine learning project that analyses diagnostic tumour features, predicts malignancy (Benign / Malignant) using a custom-engineered Support Vector Machine (SVM) pipeline, and surfaces interactive visualizations and real-time predictions through a Streamlit dashboard.

---

## 📋 Overview

Given diagnostic records of breast tumour cell nuclei (radius, texture, perimeter, area, smoothness, compactness, concavity, etc.), this project:

- Preprocesses raw diagnostic data by mapping column headers, removing identifiers, and analysing feature distributions
- Implements custom Scikit-learn transformers (`LogTransformer` and `FeatureEngineer`) to generate domain-specific interaction features and ratios
- Trains an advanced Support Vector Machine (SVM) pipeline with an optimised decision threshold to accurately classify tumours as Malignant or Benign
- Packages the trained pipeline into an interactive, 4-tab Streamlit dashboard for data exploration, feature visualisation, patient-level prediction, and model evaluation

## 🗂️ Project Structure

```text
.
├── Breast_Cancer_Prediction.ipynb       # Data preprocessing, EDA & model training notebook
├── app.py                               # Streamlit interactive dashboard
├── requirements.txt                     # Python dependencies
├── breast_cancer.csv                    # Processed dataset
├── wdbc.data                            # Raw Wisconsin Breast Cancer dataset
├── best_svm_pipeline.pkl                # Saved Scikit-learn SVM pipeline artifact
├── best_threshold.pkl                   # Optimised classification decision threshold
└── feature_columns.pkl                  # Saved list of model input features
```

## 📓 Notebook Contents

The Jupyter Notebook focuses on data preparation and exploratory analysis:
1. **Data Preprocessing** — Loads raw `wdbc.data`, assigns standard column names (from `radius_mean` to `fractal_dimension_worst`), drops the redundant `id` column, and exports the clean data to `breast_cancer.csv`.
2. **Exploratory Data Analysis (EDA)** — Generates statistical summaries, examines the class distribution between Malignant (`M`) and Benign (`B`) tumours, and visualises numerical features using side-by-side histograms and boxplots.
3. **Custom Feature Engineering** — Builds custom Scikit-learn pipeline classes (`LogTransformer`, `FeatureEngineer`, `NumpyToArrayToDataFrame`) that engineer key ratio and interaction variables, including:
    - `area_perimeter_ratio` and `compactness_index`
    - `texture_se_x_worst` and `smoothness_se_x_area`
    -  `radius_worst_mean_ratio` and `concavity_worst_mean_ratio`

## 🖥️ Dashboard Features

The Streamlit dashboard (app.py) is organised into four interactive tabs:  

- 📊 **EDA** — Displays dataset dataframes, descriptive statistics, and a visual count plot of the diagnosis distribution (Malignant vs. Benign).  
- 📈 **Visualisations** — Allows users to dynamically select numeric features to view side-by-side KDE histograms and boxplots, alongside a complete correlation heatmap.  
- 🤖 **Prediction** — Provides interactive numerical inputs for all feature columns. Generates real-time predictions by comparing the model's malignant probability against the tuned threshold, displaying an alert banner and an annotated probability bar chart.  
- 📉 **Evaluation** — Showcases the model's performance on the dataset using ROC curves, Precision-Recall curves, a confusion matrix, a full classification report, and the ROC-AUC score.

## ⚙️ Setup & Usage

### 1. 📔 Run the notebook
Open the `.ipynb` file in Jupyter Notebook or Google Colab and execute the cells sequentially. This cleans the dataset, performs EDA, and ensures your model artifacts (`.pkl` files) are generated and saved.
### 2. 🚀 Run the dashboard
Ensure you have the required dependencies installed, then launch the Streamlit application:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🧰 Tech Stack

- **Language:** Python
- **Data & ML:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **Model persistence:** Joblib

## 🔗 Streamlit App

- https://breastcancersvm.streamlit.app/
