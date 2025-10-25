import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print('--- Testing TensorFlow Installation with a 3-Class DNN Model ---')

# --- 1. Data Preparation ---
print('--- 1. Preparing Data ---')
normalized_counts_path = '../data/normalized_gene_counts.csv'
normalized_df = pd.read_csv(normalized_counts_path, index_col=0)
X = normalized_df.T

metadata_path = '../data/MAGE_metadata.txt'
metadata_df = pd.read_csv(metadata_path)

etiologies_to_include = ['Dilated cardiomyopathy (DCM)', 'Hypertrophic cardiomyopathy (HCM)', 'Non-Failing Donor']
filtered_metadata = metadata_df[metadata_df['etiology'].isin(etiologies_to_include)]

common_samples = X.index.intersection(filtered_metadata['Run'])
X_filtered = X.loc[common_samples]
y_labels = X_filtered.index.map(filtered_metadata.set_index('Run')['etiology'])

label_mapping = {'Non-Failing Donor': 0, 'Dilated cardiomyopathy (DCM)': 1, 'Hypertrophic cardiomyopathy (HCM)': 2}
y = y_labels.map(label_mapping)

# --- Feature Selection: Use Top 1000 Most Variable Genes ---
print('--- Feature Selection: Using Top 1000 Most Variable Genes ---')

# Calculate the variance for each gene across all samples
print("Calculating variance for all genes... (This may take a moment)")
variances = X_filtered.var(axis=0)
print("Variance calculation complete.")

# Get the names of the top 1000 genes with the highest variance
print("Sorting genes by variance...")
top_1000_variable_genes = variances.sort_values(ascending=False).head(1000).index

# Filter the expression matrix to only include these genes
X_reduced = X_filtered[top_1000_variable_genes]
print(f'Reduced feature set from {X_filtered.shape[1]} to {X_reduced.shape[1]} genes.\n')

# Use the reduced feature set for training
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42, stratify=y)
print("Data splitting complete.")
print('Data preparation complete.\n')

# --- 2. Build and Evaluate DNN ---
print('--- 2. Building and Evaluating DNN ---')

print(f"TensorFlow Version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

dnn_model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax') # 3 output neurons for 3 classes
])

# Use sparse_categorical_crossentropy for integer-encoded multi-class labels
dnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("Training Deep Neural Network...")
history = dnn_model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stopping],
                        verbose=2)
print("DNN Training complete.")

# Evaluate the model
y_pred_proba = dnn_model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

print('\n--- DNN Test Set Evaluation ---\n')
print(classification_report(y_test, y_pred, target_names=label_mapping.keys()))

auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
print(f'ROC AUC Score (One-vs-Rest, Macro): {auc:.4f}\n')

# Plot and save confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_mapping.keys(),
            yticklabels=label_mapping.keys())
plt.title('Confusion Matrix: DNN', fontsize=14)
plt.ylabel('Actual')
plt.xlabel('Predicted')
cm_plot_path = '../data/confusion_matrix_DNN.png'
plt.savefig(cm_plot_path)
print(f"Confusion matrix for DNN saved to {cm_plot_path}\n")
plt.close()

print("--- TensorFlow test complete. ---")
