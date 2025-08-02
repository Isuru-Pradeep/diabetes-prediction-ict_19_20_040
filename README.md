# Neural Network Model Development for Diabetes Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning project that develops a neural network model to predict diabetes occurrence using the Pima Indian Diabetes Dataset. This project demonstrates the complete ML workflow from data preprocessing to model evaluation and hyperparameter optimization.

## 📋 Project Overview

This project builds a binary classification neural network to predict diabetes in patients based on diagnostic measurements. The model achieves **78.45% accuracy** and **0.8663 ROC-AUC score**, demonstrating excellent discriminative ability suitable for clinical screening applications.

### 🎯 Key Features
- **Data Preprocessing**: Handles missing values, feature scaling, and stratified data splitting
- **Neural Network Architecture**: Feedforward network with optimized hyperparameters
- **Comprehensive Evaluation**: Multiple metrics including ROC-AUC, precision, recall, and F1-score
- **Feature Importance Analysis**: Permutation-based importance to identify key predictors
- **Hyperparameter Tuning**: Grid search optimization over 24 different configurations

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 78.45% |
| **Precision** | 76.67% |
| **Recall** | 56.10% |
| **F1-Score** | 0.6479 |
| **ROC-AUC** | 0.8663 |
| **Specificity** | 90.67% |

### 🏆 Optimal Configuration
- **Architecture**: 1 hidden layer with 32 neurons
- **Dropout Rate**: 0.3
- **Learning Rate**: 0.001
- **Activation**: ReLU (hidden), Sigmoid (output)
- **Optimizer**: Adam

## 📁 Project Structure

```
diabetes-prediction/
│
├── diabetes.csv              # Pima Indian Diabetes Dataset
├── model.ipynb               # Jupyter notebook with complete analysis
├── model.py                  # Python script version of the model
├── requirements.txt          # Required Python packages
├── README.md                 # Project documentation
└── output_[timestamp]/       # Generated during execution
    ├── training_log.log      # Detailed training logs
    ├── diabetes_model.h5     # Saved trained model
    ├── *.png                 # Generated visualizations
    └── *_model_results.txt   # Evaluation results
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Isuru-Pradeep/diabetes-prediction-ict_19_20_040.git
   cd diabetes-prediction-ict_19_20_040
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the model**
   
   **Option A: Jupyter Notebook (Recommended)**
   ```bash
   jupyter notebook model.ipynb
   ```
   
   **Option B: Python Script**
   ```bash
   python model.py
   ```

## 📈 Dataset Information

**Dataset**: Pima Indian Diabetes Dataset  
**Source**: UCI Machine Learning Repository  
**Size**: 768 instances, 8 features + 1 target  
**Target**: Binary classification (0: Non-diabetic, 1: Diabetic)

### Features Description
| Feature | Description | Unit |
|---------|-------------|------|
| `Pregnancies` | Number of pregnancies | Count |
| `Glucose` | Plasma glucose concentration (2-hour OGTT) | mg/dL |
| `BloodPressure` | Diastolic blood pressure | mmHg |
| `SkinThickness` | Triceps skinfold thickness | mm |
| `Insulin` | 2-hour serum insulin | μU/mL |
| `BMI` | Body Mass Index | kg/m² |
| `DiabetesPedigreeFunction` | Genetic predisposition score | - |
| `Age` | Age | years |

### 📊 Key Statistics
- **Class Distribution**: 65.1% Non-diabetic, 34.9% Diabetic
- **Data Quality**: 652 zero values imputed using median replacement
- **Splits**: 70% Training, 15% Validation, 15% Test (stratified)

## 🔧 Technical Implementation

### Data Preprocessing
- **Missing Value Handling**: Median imputation for biologically impossible zero values
- **Feature Scaling**: StandardScaler normalization (mean=0, std=1)
- **Data Splitting**: Stratified sampling to maintain class distribution

### Training Configuration
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Binary crossentropy
- **Batch Size**: 16
- **Regularization**: Dropout (0.3) + Early stopping + Learning rate reduction
- **Validation**: Monitor validation loss with patience=10

## 📊 Feature Importance Rankings

Based on permutation importance analysis:

1. **Glucose** (0.1236) - Primary diabetes indicator 🥇
2. **BMI** (0.0391) - Obesity-diabetes correlation 🥈
3. **Age** (0.0282) - Age-related risk factor 🥉
4. **Insulin** (0.0264) - Metabolic indicator
5. **SkinThickness** (0.0172) - Indirect obesity measure
6. **BloodPressure** (0.0052) - Cardiovascular factor
7. **DiabetesPedigreeFunction** (0.0046) - Genetic predisposition
8. **Pregnancies** (-0.0017) - Minimal impact

## 📋 Output Files

When you run the model, it automatically generates:

- **Timestamped output directory** with all results
- **Training logs** with detailed progress information
- **Saved model** in HDF5 format for future use
- **Evaluation plots** including ROC curves and confusion matrices
- **Results files** with comprehensive performance metrics

## 🔍 Model Interpretation

### Clinical Insights
- **High Specificity (90.67%)**: Excellent at identifying non-diabetic patients
- **Moderate Sensitivity (56.10%)**: Conservative approach, may miss some diabetic cases
- **Feature Alignment**: Top predictors match established medical literature
- **Screening Utility**: Suitable for population-level diabetes risk assessment

### Limitations
- **Dataset Scope**: Limited to Pima Indian population
- **Sample Size**: 768 patients may limit generalizability
- **Missing Factors**: Lacks lifestyle and detailed family history data
- **Temporal Aspect**: Single time-point measurements only

## 🛠️ Dependencies

Key libraries used in this project:

- **TensorFlow 2.15.0**: Neural network framework
- **scikit-learn 1.2.2**: Data preprocessing and evaluation metrics
- **pandas 2.3.1**: Data manipulation and analysis
- **numpy 1.26.4**: Numerical computations
- **matplotlib 3.10.5**: Data visualization
- **seaborn 0.13.2**: Statistical plotting

See `requirements.txt` for complete list with exact versions.

## 📚 Usage Examples

### Basic Prediction
```python
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load saved model
model = tf.keras.models.load_model('output_[timestamp]/diabetes_model.h5')

# Example patient data (after scaling)
patient_data = np.array([[...]])  # 8 features
prediction = model.predict(patient_data)
probability = prediction[0][0]

print(f"Diabetes Probability: {probability:.3f}")
print(f"Prediction: {'Diabetic' if probability > 0.5 else 'Non-diabetic'}")
```

### Feature Importance Visualization
```python
import matplotlib.pyplot as plt

features = ['Glucose', 'BMI', 'Age', 'Insulin', 'SkinThickness', 
           'BloodPressure', 'DiabetesPedigreeFunction', 'Pregnancies']
importance = [0.1236, 0.0391, 0.0282, 0.0264, 0.0172, 0.0052, 0.0046, -0.0017]

plt.barh(features, importance)
plt.xlabel('Feature Importance')
plt.title('Diabetes Prediction Feature Importance')
plt.show()
```

## 🎓 Academic Context

**Course**: ICT4302 – INTELLIGENT SYSTEMS  
**Institution**: Rajarata University of Sri Lanka, Department of Computing  
**Assignment**: Mini Project – 01  
**Focus**: Neural network development for healthcare applications

## 🤝 Contributing

This is an academic project, but suggestions and improvements are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Author**: Isuru Pradeep
**Email**: pradeepisuru31@gmail.com  

---

## 🙏 Acknowledgments

- TensorFlow and Keras teams for the excellent deep learning framework
- scikit-learn community for comprehensive machine learning tools
- Course instructors for guidance and support

---

**⭐ If you find this project helpful, please consider giving it a star!**
