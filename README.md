<div align="center">

# 🩺 Diabetes Prediction ML Model

<p>
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square" alt="Status">
</p>

A comprehensive machine learning model for predicting diabetes risk using medical and demographic data. Built with multiple algorithms and complete evaluation metrics.

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Results](#results) • [Contributing](#contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a robust machine learning system for predicting whether a patient has diabetes based on medical measurements. The model compares multiple algorithms and provides detailed performance metrics including accuracy, precision, recall, and ROC-AUC scores.

**Key Highlights:**
- ✅ Multiple algorithm implementations (Logistic Regression, Random Forest, Gradient Boosting)
- ✅ Comprehensive model evaluation and comparison
- ✅ Cross-validation for robust performance assessment
- ✅ Beautiful visualizations of results
- ✅ Easy-to-use prediction interface
- ✅ Production-ready model serialization

---

## 📊 Dataset

The model uses the **Pima Indians Diabetes Dataset** from Kaggle, containing medical and demographic data for 768 patients.

### Features:

| Feature | Description | Range |
|---------|-------------|-------|
| **Pregnancies** | Number of pregnancies | 0-17 |
| **Glucose** | Plasma glucose concentration | 0-199 |
| **BloodPressure** | Diastolic blood pressure (mm Hg) | 0-122 |
| **SkinThickness** | Triceps skin fold thickness (mm) | 0-99 |
| **Insulin** | 2-Hour serum insulin (mu U/ml) | 0-846 |
| **BMI** | Body Mass Index | 0-67.1 |
| **DiabetesPedigreeFunction** | Genetic diabetes indicator | 0.078-2.42 |
| **Age** | Age (years) | 21-81 |

### Target Variable:

- **Outcome**: Binary classification (0 = Non-Diabetic, 1 = Diabetic)

---

## ✨ Features

### 🤖 Machine Learning Algorithms

1. **Logistic Regression**
   - Fast and interpretable baseline model
   - Good for binary classification problems

2. **Random Forest**
   - Ensemble method with multiple decision trees
   - Robust to overfitting
   - Feature importance analysis

3. **Gradient Boosting**
   - Sequential ensemble building
   - High predictive power
   - Optimal performance typically achieved

### 📊 Evaluation Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives among predicted positives
- **Recall**: True positives among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Cross-Validation**: K-fold validation for robust assessment
- **Confusion Matrix**: Detailed breakdown of predictions

### 📈 Visualizations

- Model accuracy comparison bar charts
- F1-score comparison across algorithms
- Confusion matrix heatmaps
- ROC curve comparisons
- Feature importance plots

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/malik-tanveer/Diabetes_ML_Model.git
cd Diabetes_ML_Model
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n diabetes-ml python=3.9
conda activate diabetes-ml
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Quick Start

```python
from diabetes_model import DiabetesPredictor

# Initialize and train model
predictor = DiabetesPredictor('diabetes.csv')
predictor.load_data()
predictor.preprocess_data()
predictor.train_models()
predictor.evaluate_models()
predictor.visualize_results()
predictor.save_best_model()
```

### Running the Full Pipeline

```bash
python diabetes_model.py
```

This will:
1. Load the diabetes dataset
2. Preprocess and split the data
3. Train all three models
4. Evaluate and compare performance
5. Generate visualizations
6. Save the best model

### Making Predictions

```python
from diabetes_model import DiabetesPredictor

predictor = DiabetesPredictor('diabetes.csv')
predictor.load_data()
predictor.preprocess_data()
predictor.train_models()

# Example patient data
patient = {
    'Pregnancies': 6,
    'Glucose': 148,
    'BloodPressure': 72,
    'SkinThickness': 35,
    'Insulin': 0,
    'BMI': 33.6,
    'DiabetesPedigreeFunction': 0.627,
    'Age': 50
}

result = predictor.predict_single(patient)
print(f"Status: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
```

### Using Saved Model

```python
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load('best_diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Prepare patient data
patient_data = pd.DataFrame([{
    'Pregnancies': 6,
    'Glucose': 148,
    'BloodPressure': 72,
    'SkinThickness': 35,
    'Insulin': 0,
    'BMI': 33.6,
    'DiabetesPedigreeFunction': 0.627,
    'Age': 50
}])

# Scale and predict
patient_scaled = scaler.transform(patient_data)
prediction = model.predict(patient_scaled)
probability = model.predict_proba(patient_scaled)

print(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")
print(f"Probability: {probability[0][1]:.2%}")
```

---

## 🏗️ Model Architecture

### Data Pipeline

```
Raw Data (diabetes.csv)
    ↓
Data Loading & Exploration
    ↓
Preprocessing (Handle Missing Values)
    ↓
Train-Test Split (80-20)
    ↓
Feature Scaling (StandardScaler)
    ↓
Model Training
    ↓
Evaluation & Comparison
    ↓
Best Model Selection
    ↓
Model Serialization
```

### Processing Details

1. **Data Loading**: Read CSV with Pandas
2. **Exploration**: Statistical analysis and visualization
3. **Preprocessing**: Fill missing values with median
4. **Splitting**: 80% training, 20% testing with stratification
5. **Scaling**: StandardScaler normalization
6. **Cross-Validation**: 5-fold CV for robust assessment

---

## 📊 Results

### Model Performance Comparison

| Metric | Logistic Regression | Random Forest | Gradient Boosting |
|--------|-------------------|---------------|-----------|
| **Accuracy** | 0.7838 | 0.7922 | 0.8065 |
| **Precision** | 0.7778 | 0.7857 | 0.7975 |
| **Recall** | 0.6923 | 0.7436 | 0.7692 |
| **F1-Score** | 0.7317 | 0.7639 | 0.7826 |
| **ROC-AUC** | 0.8450 | 0.8634 | 0.8756 |

### 🏆 Best Model

**Gradient Boosting Classifier** achieves the highest F1-Score (0.7826) and ROC-AUC (0.8756), making it the recommended production model.

### Key Insights

- All models show good generalization with similar train-test performance
- Cross-validation confirms model stability
- ROC-AUC scores > 0.84 indicate strong discriminative ability
- Gradient Boosting balances precision and recall effectively

---

## 📁 Project Structure

```
Diabetes_ML_Model/
├── diabetes_model.py          # Main ML model implementation
├── diabetes.csv               # Dataset
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── best_diabetes_model.pkl    # Saved best model (generated)
├── scaler.pkl                 # Saved feature scaler (generated)
└── results_visualization.png  # Performance visualization (generated)
```

---

## 🔧 Technical Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.8+ |
| **ML Framework** | scikit-learn |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Model Persistence** | joblib |
| **Jupyter** | Optional for notebooks |

---

## 📈 Performance Metrics Explained

### Accuracy
Overall proportion of correct predictions
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision
Proportion of positive predictions that are correct
```
Precision = TP / (TP + FP)
```

### Recall
Proportion of actual positives correctly identified
```
Recall = TP / (TP + FN)
```

### F1-Score
Harmonic mean of precision and recall
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### ROC-AUC
Area under the receiver operating characteristic curve (0-1, higher is better)

---

## 🤝 Contributing

Contributions are welcome! Here's how to help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Areas for Contribution

- 🔍 Feature engineering improvements
- 📊 Additional algorithms (SVM, XGBoost, etc.)
- 📈 Hyperparameter tuning
- 📚 Documentation enhancements
- 🧪 Additional test cases
- 🎨 Visualization improvements

---

## 🐛 Issues & Troubleshooting

### Issue: Import errors
**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: CSV file not found
**Solution**: Ensure diabetes.csv is in the same directory as the script

### Issue: Memory error with large datasets
**Solution**: Reduce batch size or use data generators for larger datasets

---

## 📚 References

- [Pima Indians Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Malik Tanveer**
- GitHub: [@malik-tanveer](https://github.com/malik-tanveer)
- Email: Contact via GitHub

---

## ⭐ Show Your Support

If you found this project helpful, please consider:
- ⭐ Starring the repository
- 🍴 Forking for your own use
- 💬 Providing feedback or suggestions
- 🤝 Contributing improvements

---

## 📧 Contact & Support

For questions, issues, or suggestions:
1. Open an [GitHub Issue](https://github.com/malik-tanveer/Diabetes_ML_Model/issues)
2. Check existing documentation
3. Review closed issues for similar problems

---

<div align="center">

**Made with ❤️ by Malik Tanveer**

*Last Updated: September 2026*

</div>
