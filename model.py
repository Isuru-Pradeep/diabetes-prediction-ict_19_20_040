import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import pkg_resources
from datetime import datetime
import os

# Configuration Section
CONFIG = {
    'RANDOM_STATE': 42,
    'BATCH_SIZE': 16,
    'MAX_EPOCHS': 100,
    'N_PERMUTATIONS': 15,
    'TEST_SIZE': 0.3,
    'VAL_SIZE': 0.5,
    'HIDDEN_LAYERS': [1, 2],
    'NEURONS': [ 8 , 16 , 32],
    'DROPOUT_RATE': [0.2, 0.3],
    'LEARNING_RATE': [0.01, 0.001]
}

np.random.seed(CONFIG['RANDOM_STATE'])
tf.random.set_seed(CONFIG['RANDOM_STATE'])

# Configure logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"output_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

log_file = os.path.join(output_dir, 'training_log.log')

# Clear existing logging handlers before setting new ones
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(f"Created output directory: {output_dir}")
logger.info(f"Logging to file: {log_file}")

# Function to save results to a file in the output directory
def save_results_to_file(phase, history, y_test, y_pred, y_pred_proba, feature_importance, X, output_dir):
    """
    Save training results to a file in the output directory.
    
    Args:
        phase (str): Training phase ('Initial' or 'Final').
        history: Training history from model.fit().
        y_test: True labels for test set.
        y_pred: Predicted labels for test set.
        y_pred_proba: Predicted probabilities for test set.
        feature_importance: List of (feature, importance) tuples.
        X: Feature DataFrame for column names.
        output_dir (str): Directory to save the results file.
    """
    filename = os.path.join(output_dir, f"{phase.lower()}_model_results.txt")
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    train_acc = max(history.history['accuracy'])
    val_acc = max(history.history['val_accuracy'])
    train_loss = min(history.history['loss'])
    val_loss = min(history.history['val_loss'])
    acc_gap = train_acc - val_acc
    loss_gap = val_loss - train_loss
    
    with open(filename, 'w') as f:
        f.write(f"===== {phase} Model Results =====\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("Performance Metrics:\n")
        f.write(f"  Test Accuracy: {accuracy:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall: {recall:.4f}\n")
        f.write(f"  F1-Score: {f1:.4f}\n")
        f.write(f"  ROC-AUC Score: {roc_auc:.4f}\n")
        f.write(f"\nConfusion Matrix:\n{conf_matrix}\n")
        f.write("\nROC Curve Interpretation:\n")
        f.write("The ROC curve shows the trade-off between sensitivity (TPR) and specificity (1-FPR).\n")
        f.write(f"An AUC of {roc_auc:.4f} indicates the model's ability to distinguish between classes.\n")
        f.write("\nFeature Importance (by permutation):\n")
        for feature, importance in feature_importance:
            f.write(f"  {feature}: {importance:.4f}\n")
        f.write("\nOverfitting Analysis:\n")
        f.write(f"  Training Accuracy: {train_acc:.4f}\n")
        f.write(f"  Validation Accuracy: {val_acc:.4f}\n")
        f.write(f"  Accuracy Gap: {acc_gap:.4f}\n")
        f.write(f"  Training Loss: {train_loss:.4f}\n")
        f.write(f"  Validation Loss: {val_loss:.4f}\n")
        f.write(f"  Loss Gap: {loss_gap:.4f}\n")
        if acc_gap > 0.1 or loss_gap > train_loss * 0.2:
            f.write("  Signs of overfitting detected\n")
            f.write("  Recommendations: Increase dropout, reduce model complexity, or collect more data\n")
        elif acc_gap < 0.02 and abs(loss_gap) < train_loss * 0.1:
            f.write("  Model appears well-balanced\n")
        else:
            f.write("  Mild overfitting, but acceptable\n")
    
    logger.info(f"Saved {phase} model results to {filename}")

# Check package versions
required_versions = {
    'numpy': '1.21.0',
    'pandas': '1.3.0',
    'matplotlib': '3.4.0',
    'seaborn': '0.11.0',
    'scikit-learn': '1.2.2',
    'tensorflow': '2.15.0'
}
for pkg, min_version in required_versions.items():
    try:
        installed_version = pkg_resources.get_distribution(pkg).version
        logger.info(f"{pkg} version: {installed_version}")
        if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(min_version):
            logger.warning(f"{pkg} version {installed_version} is below recommended {min_version}")
    except pkg_resources.DistributionNotFound:
        logger.error(f"{pkg} is not installed")
        raise ImportError(f"Please install {pkg}>={min_version}")

# Check GPU availability
logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

# Report Section: Problem Definition
logger.info("="*60)
logger.info("DIABETES PREDICTION USING NEURAL NETWORKS")
logger.info("="*60)
logger.info("Problem Statement:")
logger.info("Diabetes is a chronic condition affecting millions worldwide.")
logger.info("Early prediction enables better management and prevents complications.")
logger.info("This project builds a neural network to predict diabetes using the Pima Indian Diabetes Dataset.")

# Report Section: Data Preparation
logger.info("\n" + "="*60)
logger.info("DATA PREPARATION")
logger.info("="*60)

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
try:
    df = pd.read_csv(url, names=columns)
    logger.info("Dataset loaded successfully from URL!")
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise FileNotFoundError("Please ensure internet connection or provide local 'diabetes.csv'.")

# Dataset exploration
logger.info(f"\nDataset Shape: {df.shape}")
logger.info(f"Features: {df.columns.tolist()}")
logger.info("\nDataset Overview:")
logger.info(df.info())
logger.info("\nStatistical Summary:")
logger.info(df.describe())

# Target distribution
logger.info("\nTarget Variable Distribution:")
target_counts = df['Outcome'].value_counts()
logger.info(f"Non-diabetic (0): {target_counts[0]} ({target_counts[0]/len(df)*100:.1f}%)")
logger.info(f"Diabetic (1): {target_counts[1]} ({target_counts[1]/len(df)*100:.1f}%)")

# Check for missing/zero values
logger.info("\nMissing Values:")
logger.info(df.isnull().sum())
logger.info("\nZero Values Analysis:")
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    zero_count = (df[col] == 0).sum()
    logger.info(f"{col}: {zero_count} zeros ({zero_count/len(df)*100:.1f}%)")

# Handle zero values with median imputation (robust to outliers)
logger.info("\nHandling Missing/Zero Values:")
for col in zero_cols:
    if (df[col] == 0).sum() > 0:
        median_val = df[df[col] != 0][col].median()
        df[col] = df[col].replace(0, median_val)
        logger.info(f"Replaced {col} zeros with median: {median_val:.2f}")

# Data Visualization
# Target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Outcome', data=df, palette=['skyblue', 'salmon'])
plt.title('Target Distribution')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-diabetic', 'Diabetic'])
plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
plt.close()

# Feature distributions
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
plt.figure(figsize=(15, 8))
for i, feature in enumerate(features, 1):
    plt.subplot(2, 4, i)
    sns.histplot(df[feature], bins=20, kde=True, color='lightblue')
    plt.title(f'{feature} Distribution')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
plt.close()

# Feature scaling and data splitting
logger.info("\nData Splitting and Scaling:")
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data: 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=CONFIG['TEST_SIZE'], random_state=CONFIG['RANDOM_STATE'], stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=CONFIG['VAL_SIZE'], random_state=CONFIG['RANDOM_STATE'], stratify=y_temp)
logger.info(f"Training set: {X_train.shape[0]} samples (~70%)")
logger.info(f"Validation set: {X_val.shape[0]} samples (~15%)")
logger.info(f"Test set: {X_test.shape[0]} samples (~15%)")
logger.info("Training set distribution:")
logger.info(pd.Series(y_train).value_counts(normalize=True))
logger.info("Validation set distribution:")
logger.info(pd.Series(y_val).value_counts(normalize=True))
logger.info("Test set distribution:")
logger.info(pd.Series(y_test).value_counts(normalize=True))
logger.info("Note: Stratified splitting ensures class distribution is maintained across all sets.")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
logger.info("Feature scaling completed using StandardScaler")
logger.info("Scaling ensures features have mean=0 and variance=1, improving model convergence.")

# Report Section: Model Development
logger.info("\n" + "="*60)
logger.info("MODEL DEVELOPMENT")
logger.info("="*60)
logger.info("Training Phase Details:")
logger.info("- The model is trained on 70% of the data to learn patterns.")
logger.info("- Early stopping monitors validation loss to prevent overfitting.")
logger.info("- Learning rate reduction adjusts the learning rate if validation loss plateaus.")
logger.info("Suggestions for Training Improvement:")
logger.info("- Use k-fold cross-validation to reduce variance in performance estimates.")
logger.info("- Implement data augmentation techniques for small datasets.")
logger.info("- Use class weights to handle imbalanced data.")
logger.info("- Explore batch normalization to stabilize training.")

def create_model(hidden_layers=1, neurons=16, dropout_rate=0.2, learning_rate=0.001):
    """
    Create a feedforward neural network for binary classification.
    
    Args:
        hidden_layers (int): Number of hidden layers.
        neurons (int): Number of neurons per hidden layer.
        dropout_rate (float): Dropout rate for regularization.
        learning_rate (float): Learning rate for Adam optimizer.
    
    Returns:
        Sequential: Compiled Keras model.
    """
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(Dropout(dropout_rate))
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initial model training
logger.info("\nBuilding Initial Model:")
try:
    model = create_model()
    logger.info(model.summary())
except Exception as e:
    logger.error(f"Error building model: {e}")
    raise

# Train with early stopping and learning rate reduction
try:
    history = model.fit(X_train_scaled, y_train,
                       batch_size=CONFIG['BATCH_SIZE'],
                       epochs=CONFIG['MAX_EPOCHS'],
                       validation_data=(X_val_scaled, y_val),
                       verbose=1,
                       callbacks=[
                           tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
                           tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, monitor='val_loss')
                       ])
except Exception as e:
    logger.error(f"Error training initial model: {e}")
    raise

# Save initial model results
try:
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = (y_pred_proba > 0.5).astype(int)
    # Compute feature importance for initial model
    initial_feature_importance = []
    baseline_score = model.evaluate(X_test_scaled, y_test, verbose=0)[1]
    for i, feature in enumerate(X.columns):
        importance_scores = []
        for _ in range(CONFIG['N_PERMUTATIONS']):
            X_test_permuted = X_test_scaled.copy()
            X_test_permuted[:, i] = np.random.permutation(X_test_permuted[:, i])
            permuted_score = model.evaluate(X_test_permuted, y_test, verbose=0)[1]
            importance_scores.append(baseline_score - permuted_score)
        initial_feature_importance.append((feature, np.mean(importance_scores)))
    initial_feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    save_results_to_file("Initial", history, y_test, y_pred, y_pred_proba, initial_feature_importance, X, output_dir)
except Exception as e:
    logger.error(f"Error evaluating initial model: {e}")
    raise

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_history.png'))
plt.close()

# Report Section: Hyperparameter Tuning
logger.info("\n" + "="*60)
logger.info("HYPERPARAMETER TUNING")
logger.info("="*60)
logger.info("Hyperparameter tuning uses a grid search over the configuration parameters.")
logger.info("Validation set performance guides the selection of the best model parameters.")
logger.info("Suggestions for Tuning Improvement:")
logger.info("- Use random search or Bayesian optimization for efficiency.")
logger.info("- Expand the parameter grid for finer granularity.")
logger.info("- Incorporate learning rate schedules for adaptive learning.")

# Manual hyperparameter tuning
param_grid = {
    'hidden_layers': CONFIG['HIDDEN_LAYERS'],
    'neurons': CONFIG['NEURONS'],
    'dropout_rate': CONFIG['DROPOUT_RATE'],
    'learning_rate': CONFIG['LEARNING_RATE']
}

logger.info("Parameter grid for Manual Search:")
for param, values in param_grid.items():
    logger.info(f"  {param}: {values}")

best_score = 0
best_params = {}
results = []

logger.info("\nStarting hyperparameter tuning...")
for hidden_layers in param_grid['hidden_layers']:
    for neurons in param_grid['neurons']:
        for dropout_rate in param_grid['dropout_rate']:
            for learning_rate in param_grid['learning_rate']:
                try:
                    logger.info(f"Testing: layers={hidden_layers}, neurons={neurons}, "
                                f"dropout={dropout_rate}, lr={learning_rate}")
                    model_temp = create_model(hidden_layers, neurons, dropout_rate, learning_rate)
                    history_temp = model_temp.fit(X_train_scaled, y_train,
                                                 batch_size=CONFIG['BATCH_SIZE'],
                                                 epochs=CONFIG['MAX_EPOCHS'],
                                                 validation_data=(X_val_scaled, y_val),
                                                 verbose=0,
                                                 callbacks=[
                                                     tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
                                                 ])
                    val_score = max(history_temp.history['val_accuracy'])
                    results.append({
                        'hidden_layers': hidden_layers,
                        'neurons': neurons,
                        'dropout_rate': dropout_rate,
                        'learning_rate': learning_rate,
                        'val_accuracy': val_score
                    })
                    if val_score > best_score:
                        best_score = val_score
                        best_params = {
                            'hidden_layers': hidden_layers,
                            'neurons': neurons,
                            'dropout_rate': dropout_rate,
                            'learning_rate': learning_rate
                        }
                    logger.info(f"  Validation Accuracy: {val_score:.4f}")
                except Exception as e:
                    logger.error(f"Error in hyperparameter tuning: {e}")
                    continue

logger.info(f"\nBest parameters found:")
for param, value in best_params.items():
    logger.info(f"  {param}: {value}")
logger.info(f"Best validation accuracy: {best_score:.4f}")

# Train final model with best parameters
logger.info("\nTraining Final Model with Best Parameters:")
try:
    final_model = create_model(**best_params)
    final_history = final_model.fit(X_train_scaled, y_train,
                                   batch_size=CONFIG['BATCH_SIZE'],
                                   epochs=CONFIG['MAX_EPOCHS'],
                                   validation_data=(X_val_scaled, y_val),
                                   verbose=1,
                                   callbacks=[
                                       tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
                                       tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, monitor='val_loss')
                                   ])
except Exception as e:
    logger.error(f"Error training final model: {e}")
    raise

# Report Section: Model Evaluation
logger.info("\n" + "="*60)
logger.info("MODEL EVALUATION")
logger.info("="*60)
logger.info("Testing Phase Details:")
logger.info("- The test set (15% of data) evaluates the model's generalization to unseen data.")
logger.info("- Metrics like accuracy, precision, recall, F1-score, and ROC-AUC are computed.")
logger.info("Suggestions for Testing Improvement:")
logger.info("- Use stratified k-fold cross-validation for robust evaluation.")
logger.info("- Analyze misclassified samples to understand model weaknesses.")
logger.info("- Compute confidence intervals for metrics to assess stability.")

# Predictions and feature importance for final model
try:
    y_pred_proba = final_model.predict(X_test_scaled)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Feature Importance
    logger.info("\nFeature Importance Analysis:")
    feature_importance = []
    baseline_score = final_model.evaluate(X_test_scaled, y_test, verbose=0)[1]
    for i, feature in enumerate(X.columns):
        importance_scores = []
        for _ in range(CONFIG['N_PERMUTATIONS']):
            X_test_permuted = X_test_scaled.copy()
            X_test_permuted[:, i] = np.random.permutation(X_test_permuted[:, i])
            permuted_score = final_model.evaluate(X_test_permuted, y_test, verbose=0)[1]
            importance_scores.append(baseline_score - permuted_score)
        feature_importance.append((feature, np.mean(importance_scores)))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Save final model results
    save_results_to_file("Final", final_history, y_test, y_pred, y_pred_proba, feature_importance, X, output_dir)
except Exception as e:
    logger.error(f"Error evaluating final model: {e}")
    raise

# Metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

logger.info(f"Test Accuracy: {accuracy:.4f}")
logger.info(f"\nConfusion Matrix:\n{conf_matrix}")
logger.info(f"Precision: {precision:.4f}")
logger.info(f"Recall: {recall:.4f}")
logger.info(f"F1-Score: {f1:.4f}")
logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
logger.info("\nROC Curve Interpretation:")
logger.info("The ROC curve shows the trade-off between sensitivity (TPR) and specificity (1-FPR).")
logger.info(f"An AUC of {roc_auc:.4f} indicates the model's ability to distinguish between classes.")

logger.info("Feature Importance (by permutation):")
for feature, importance in feature_importance:
    logger.info(f"  {feature}: {importance:.4f}")

# Save evaluation plots
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-diabetic', 'Diabetic'],
            yticklabels=['Non-diabetic', 'Diabetic'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'evaluation_plots.png'))
plt.close()

# Report Section: Results and Discussion
logger.info("\n" + "="*60)
logger.info("RESULTS SUMMARY")
logger.info("="*60)

logger.info(f"Final Model Architecture:")
logger.info(f"  - Hidden Layers: {best_params['hidden_layers']}")
logger.info(f"  - Neurons per Layer: {best_params['neurons']}")
logger.info(f"  - Dropout Rate: {best_params['dropout_rate']}")
logger.info(f"  - Learning Rate: {best_params['learning_rate']}")
logger.info(f"  - Activation: ReLU (hidden), Sigmoid (output)")
logger.info(f"  - Optimizer: Adam")

logger.info(f"\nPerformance Metrics:")
logger.info(f"  - Test Accuracy: {accuracy:.4f}")
logger.info(f"  - Precision: {precision:.4f}")
logger.info(f"  - Recall: {recall:.4f}")
logger.info(f"  - F1-Score: {f1:.4f}")
logger.info(f"  - ROC-AUC Score: {roc_auc:.4f}")

# Overfitting analysis
train_acc = max(final_history.history['accuracy'])
val_acc = max(final_history.history['val_accuracy'])
train_loss = min(final_history.history['loss'])
val_loss = min(final_history.history['val_loss'])
acc_gap = train_acc - val_acc
loss_gap = val_loss - train_loss

logger.info("\nOverfitting Analysis:")
logger.info(f"  - Training Accuracy: {train_acc:.4f}")
logger.info(f"  - Validation Accuracy: {val_acc:.4f}")
logger.info(f"  - Accuracy Gap: {acc_gap:.4f}")
logger.info(f"  - Training Loss: {train_loss:.4f}")
logger.info(f"  - Validation Loss: {val_loss:.4f}")
logger.info(f"  - Loss Gap: {loss_gap:.4f}")
if acc_gap > 0.1 or loss_gap > train_loss * 0.2:
    logger.info("  - Signs of overfitting detected")
    logger.info("  - Recommendations: Increase dropout, reduce model complexity, or collect more data")
elif acc_gap < 0.02 and abs(loss_gap) < train_loss * 0.1:
    logger.info("  - Model appears well-balanced")
else:
    logger.info("  - Mild overfitting, but acceptable")

logger.info(f"\nTop Contributing Features:")
for feature, importance in feature_importance[:3]:
    logger.info(f"  - {feature}: {importance:.4f}")

logger.info(f"\nPossible Improvements:")
logger.info("  - Collect more training data")
logger.info("  - Feature engineering (polynomial features, interactions)")
logger.info("  - Try different architectures (deeper networks, different activations)")
logger.info("  - Ensemble methods")
logger.info("  - Advanced regularization techniques")

# Save model
try:
    final_model.save(os.path.join(output_dir, 'diabetes_model.h5'))
    logger.info(f"Model saved as {os.path.join(output_dir, 'diabetes_model.h5')}")
except Exception as e:
    logger.error(f"Error saving model: {e}")
    raise

logger.info("\n" + "="*60)
logger.info("MODEL DEVELOPMENT COMPLETED")
logger.info("="*60)