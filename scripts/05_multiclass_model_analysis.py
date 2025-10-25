import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# --- 1. Data Preparation & Visualization ---
print('--- 1. Preparing Data and Visualizing Class Separation ---')
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

# --- PCA Plot for 3 Classes ---
print("Generating PCA plot for the 3 classes...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_filtered)
pc_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'], index=X_filtered.index)
pc_df['Condition'] = y_labels

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Condition', data=pc_df, s=80, alpha=0.9)
plt.title('PCA of 3 Heart Disease Classes', fontsize=16)
plt.xlabel(f'PC1 - {pca.explained_variance_ratio_[0]*100:.2f}% variance')
plt.ylabel(f'PC2 - {pca.explained_variance_ratio_[1]*100:.2f}% variance')
plt.legend(title='Condition')
plt.grid(True)
pca_plot_path = '../data/pca_3_class.png'
plt.savefig(pca_plot_path)
print(f"PCA plot saved to {pca_plot_path}\n")
plt.close()

X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.3, random_state=42, stratify=y)
print('Data preparation complete.')
print(f'Class distribution in full dataset:\n{y.value_counts()}\n')

# --- 2. Train, Evaluate, and Plot Models ---
print('--- 2. Training, Evaluating, and Plotting Results ---')

# Create a figure for the combined ROC plot
plt.figure(figsize=(12, 10))
linestyles = ['-', '--', ':', '-.']

models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42, multi_class='ovr'),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM (Linear Kernel)': SVC(kernel='linear', probability=True, random_state=42, decision_function_shape='ovr')
}

for (name, model), linestyle in zip(models.items(), linestyles):
    print(f'--- Evaluating {name} ---')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print(f'\n{name} Test Set Evaluation:\n')
    print(classification_report(y_test, y_pred, target_names=label_mapping.keys()))
    
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    print(f'ROC AUC Score (One-vs-Rest, Macro): {auc:.4f}\n')

    # --- Plotting Macro-Averaged ROC Curve for current model ---
    n_classes = len(np.unique(y))
    fpr = dict()
    tpr = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_proba[:, i])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Average it and compute AUC
    mean_tpr /= n_classes

    plt.plot(all_fpr, mean_tpr, label=f'{name} (macro AUC = {auc:.4f})', linestyle=linestyle, linewidth=2)

    # --- Compute and Plot Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    cm_fig, cm_ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_mapping.keys(), 
                yticklabels=label_mapping.keys(), ax=cm_ax)
    cm_ax.set_title(f'Confusion Matrix: {name}', fontsize=14)
    cm_ax.set_ylabel('Actual')
    cm_ax.set_xlabel('Predicted')
    cm_plot_path = f'../data/confusion_matrix_{name.replace(" ", "_").replace("(", "").replace(")", "")}.png'
    cm_fig.savefig(cm_plot_path)
    print(f"Confusion matrix for {name} saved to {cm_plot_path}\n")
    plt.close(cm_fig) # Close the confusion matrix figure

# Finalize and save the combined ROC plot
plt.plot([0, 1], [0, 1], 'r--', label='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Multi-Class Model Comparison: Macro-Averaged ROC Curves', fontsize=16)
plt.legend(loc="lower right")
plt.grid(True)
roc_plot_path = '../data/multiclass_roc_curves.png'
plt.savefig(roc_plot_path)
print(f"Combined ROC curve plot saved to {roc_plot_path}")
plt.close()

print('--- Multi-class model evaluation and plotting complete ---')
