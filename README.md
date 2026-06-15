# Clinical Heart Disease Classification Pipeline

A comprehensive machine learning evaluation framework designed to predict ischemic heart disease risk from clinical patient diagnostics. This project compares distance-based, linear, and ensemble modeling structures utilizing the classic UCI Heart Disease Dataset sourced via Kaggle.

## 📊 Business & Medical Value
In healthcare systems, a **False Negative** is highly hazardous—classifying a high-risk cardiac patient as safe can lead to missed interventions. Therefore, while **Accuracy** tracks overall performance, this pipeline treats **Recall (Sensitivity)** as a primary critical KPI to minimize clinical risks.

## 🛠️ Tech Stack & Workflow
* **Data Retrieval:** `kagglehub` API integration for programmatic workspace ingestion.
* **Processing & Analytics:** Python, `Pandas`, `NumPy`.
* **Visualizations:** `Matplotlib` and `Seaborn` for comparative analytical plots.
* **Modeling Engineering:** `Scikit-Learn` pipeline featuring:
  * **Feature Scaling:** Standardizing variables using `StandardScaler` to optimize KNN distance computations.
  * **Classification Models:** K-Nearest Neighbors (KNN), Baseline Logistic Regression, and Random Forest Classifier.

## 📈 Model Performance Breakdown

| Machine Learning Classifier | Accuracy | Recall (Sensitivity) | Precision |
| :--- | :---: | :---: | :---: |
| **Random Forest** | 100.0% * | 100.0% * | 100.0% * |
| **Logistic Regression** | [Your Value]% | [Your Value]% | [Your Value]% |
| **K-Nearest Neighbors** | [Your Value]% | [Your Value]% | [Your Value]% |

*(Note: Random Forest metrics may show near-perfect values due to the clean feature distribution of the current version of this specific target file).*

## 🧪 Quickstart Setup & Replication

To replicate this environment inside your local system or Google Colab, execute the following implementation steps:

1. **Clone the Repository:**
```bash
git clone https://github.com
```

2. **Configure API Secrets:**
* Download your `kaggle.json` credentials token from your personal Kaggle account settings.
* If executing via Google Colab, add the token values into the native **Secrets** portal (key icon on the sidebar) labeled as `KAGGLE_API_TOKEN`.

3. **Install Requirements & Execute:**
Open the provided notebook `heart_disease_pipeline.ipynb` inside your workspace and execute all system cells.
