import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. Data Preparation ---
print('--- 1. Preparing Data for Modeling ---')
normalized_counts_path = '../data/normalized_gene_counts.csv'
normalized_df = pd.read_csv(normalized_counts_path, index_col=0)
X = normalized_df.T

metadata_path = '../data/MAGE_metadata.txt'
metadata_df = pd.read_csv(metadata_path)

etiology_map = metadata_df.set_index('Run')['etiology']
y_labels = X.index.map(etiology_map)
y = y_labels.map({'Dilated cardiomyopathy (DCM)': 1, 'Non-Failing Donor': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print('Data preparation complete.\n')

# --- 2. Train, Evaluate, and Collect Results ---
print('--- 2. Training and Evaluating Models ---')
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM (Linear Kernel)': SVC(kernel='linear', probability=True, random_state=42)
}

plot_results = {}

for name, model in models.items():
    print(f'--- Evaluating {name} ---')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f'\n{name} Test Set Evaluation:\n')
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(est, y_pred_proba)
    print(f'ROC AUC Score: {auc:.4f}\n')
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plot_results[name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}

print('--- Model evaluation complete ---\n')

# --- 3. Plot ROC Curves ---
print('--- 3. Plotting ROC Curves ---')
plt.figure(figsize=(12, 10))

linestyles = ['-', '--', ':', '-.']
model_linestyles = zip(plot_results.items(), linestyles)

for (name, data), linestyle in model_linestyles:
    plt.plot(data['fpr'], data['tpr'], label=f"{name} (AUC = {data['auc']:.4f})", linestyle=linestyle, linewidth=2)

plt.plot([0, 1], [0, 1], 'r--', label='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Model Comparison: ROC Curves', fontsize=16)
plt.legend(loc="lower right")
plt.grid(True)

output_path = '../data/model_roc_curves_final.png'
plt.savefig(output_path)

print(f"Consolidated ROC curve plot saved to {output_path}")
