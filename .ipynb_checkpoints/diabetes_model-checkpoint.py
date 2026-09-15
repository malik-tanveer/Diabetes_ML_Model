"""
Diabetes Prediction ML Model
A machine learning model for predicting diabetes risk using medical data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')


class DiabetesPredictor:
    """
    A comprehensive diabetes prediction model with multiple algorithms.
    """
    
    def __init__(self, data_path='diabetes.csv'):
        """
        Initialize the diabetes predictor.
        
        Args:
            data_path (str): Path to the diabetes CSV file
        """
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and display basic data information."""
        print("📊 Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(f"\nDataset Info:\n{self.df.info()}")
        print(f"\nFirst few rows:\n{self.df.head()}")
        print(f"\nBasic Statistics:\n{self.df.describe()}")
        return self
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Preprocess the data: handle missing values, separate features and target.
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
        """
        print("\n🔧 Preprocessing data...")
        
        # Assuming 'Outcome' is the target variable
        self.y = self.df['Outcome']
        self.X = self.df.drop('Outcome', axis=1)
        
        # Check for missing values
        if self.X.isnull().sum().sum() > 0:
            print("Missing values detected. Filling with median...")
            self.X = self.X.fillna(self.X.median())
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"✅ Data split: Train ({len(self.X_train)}), Test ({len(self.X_test)})")
        return self
    
    def train_models(self):
        """Train multiple ML models."""
        print("\n🤖 Training models...")
        
        models_config = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        for name, model in models_config.items():
            print(f"\n  Training {name}...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
            print(f"  ✅ {name} trained")
        
        return self
    
    def evaluate_models(self):
        """Evaluate all trained models."""
        print("\n📈 Evaluating models...")
        
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Model: {name}")
            print('='*50)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"ROC-AUC:   {roc_auc:.4f}")
            print(f"CV Score:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print(f"\nClassification Report:\n{classification_report(self.y_test, y_pred)}")
        
        return self
    
    def get_best_model(self):
        """Get the best performing model."""
        best_model_name = max(self.results, key=lambda x: self.results[x]['f1'])
        print(f"\n🏆 Best Model: {best_model_name} (F1-Score: {self.results[best_model_name]['f1']:.4f})")
        return best_model_name, self.models[best_model_name]
    
    def visualize_results(self, save_path='results_visualization.png'):
        """Create visualizations of model performance."""
        print(f"\n📊 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Diabetes Prediction Model - Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Model Comparison - Accuracy
        ax1 = axes[0, 0]
        models_names = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models_names]
        bars = ax1.bar(models_names, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_ylabel('Accuracy', fontsize=11)
        ax1.set_title('Model Accuracy Comparison', fontweight='bold')
        ax1.set_ylim([0, 1])
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Model Comparison - F1 Score
        ax2 = axes[0, 1]
        f1_scores = [self.results[m]['f1'] for m in models_names]
        bars = ax2.bar(models_names, f1_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_ylabel('F1-Score', fontsize=11)
        ax2.set_title('Model F1-Score Comparison', fontweight='bold')
        ax2.set_ylim([0, 1])
        for bar, f1 in zip(bars, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Confusion Matrix of Best Model
        best_model_name = max(self.results, key=lambda x: self.results[x]['f1'])
        cm = self.results[best_model_name]['confusion_matrix']
        ax3 = axes[1, 0]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=False)
        ax3.set_title(f'Confusion Matrix - {best_model_name}', fontweight='bold')
        ax3.set_ylabel('True Label')
        ax3.set_xlabel('Predicted Label')
        
        # 4. ROC Curve
        ax4 = axes[1, 1]
        for name in models_names:
            fpr, tpr, _ = roc_curve(self.y_test, self.results[name]['y_pred_proba'])
            ax4.plot(fpr, tpr, label=f"{name} (AUC: {self.results[name]['roc_auc']:.3f})", linewidth=2)
        ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title('ROC Curve Comparison', fontweight='bold')
        ax4.legend(loc='lower right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualizations saved to {save_path}")
        plt.show()
        
        return self
    
    def save_best_model(self, model_name=None):
        """Save the best model."""
        if model_name is None:
            model_name = max(self.results, key=lambda x: self.results[x]['f1'])
        
        model = self.models[model_name]
        filename = f'best_diabetes_model.pkl'
        joblib.dump(model, filename)
        print(f"\n💾 Best model saved: {filename}")
        
        # Also save the scaler
        scaler_filename = 'scaler.pkl'
        joblib.dump(self.scaler, scaler_filename)
        print(f"💾 Scaler saved: {scaler_filename}")
        
        return self
    
    def predict_single(self, features_dict):
        """
        Make prediction for a single patient.
        
        Args:
            features_dict (dict): Dictionary with feature names and values
            
        Returns:
            dict: Prediction result with probability
        """
        best_model_name = max(self.results, key=lambda x: self.results[x]['f1'])
        model = self.models[best_model_name]
        
        # Convert dict to DataFrame and scale
        features_df = pd.DataFrame([features_dict])
        features_scaled = self.scaler.transform(features_df)
        
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        return {
            'model': best_model_name,
            'prediction': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
            'probability': probability,
            'confidence': f"{probability*100:.2f}%"
        }


def main():
    """Main execution function."""
    print("="*60)
    print("🩺 DIABETES PREDICTION ML MODEL")
    print("="*60)
    
    # Create and train model
    predictor = DiabetesPredictor('diabetes.csv')
    predictor.load_data()
    predictor.preprocess_data()
    predictor.train_models()
    predictor.evaluate_models()
    predictor.get_best_model()
    predictor.visualize_results()
    predictor.save_best_model()
    
    print("\n" + "="*60)
    print("✅ Model training complete!")
    print("="*60)
    
    # Example prediction
    print("\n📋 Example Prediction:")
    example_features = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50
    }
    
    result = predictor.predict_single(example_features)
    print(f"\nPrediction Result:")
    print(f"  Model: {result['model']}")
    print(f"  Status: {result['prediction']}")
    print(f"  Probability: {result['confidence']}")


if __name__ == "__main__":
    main()
