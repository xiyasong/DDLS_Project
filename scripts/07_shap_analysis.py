import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import shap
import matplotlib.pyplot as plt
import warnings
import mygene

warnings.filterwarnings('ignore')

print("--- SHAP Analysis for Multi-Class SVM Model ---\n")

# --- 1. Data Preparation & Feature Selection ---
print("--- 1. Preparing Data and Selecting Features ---")
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

variances = X_filtered.var(axis=0)
top_1000_variable_genes = variances.sort_values(ascending=False).head(1000).index
X_reduced = X_filtered[top_1000_variable_genes]

# --- Gene Annotation ---
print("--- Annotating Gene IDs with Symbols using mygene ---")
gene_ids = X_reduced.columns.tolist()
mg = mygene.MyGeneInfo()

# Query for gene symbols
print(f"Querying gene names for {len(gene_ids)} genes...")
gene_info = mg.querymany(gene_ids, scopes='ensembl.gene', fields='symbol', species='human', as_dataframe=True)
print("Gene name query complete.")

# Create a mapping from Ensembl ID to gene symbol
gene_id_to_name = {}
for gene_id, row in gene_info.iterrows():
    symbol = row.get('symbol')
    if isinstance(symbol, str):
         gene_id_to_name[gene_id] = symbol
    elif isinstance(symbol, list) and len(symbol) > 0:
         gene_id_to_name[gene_id] = symbol[0]
    else:
        gene_id_to_name[gene_id] = gene_id # Fallback

# Rename columns in X_reduced
X_reduced.rename(columns=gene_id_to_name, inplace=True)
print("Column names updated with gene symbols.\n")


X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42, stratify=y)
print("Data preparation complete.\n")

# --- 2. Train SVM Model ---
print("--- 2. Training SVM (Linear Kernel) Model ---")
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
print("Model training complete.\n")

# --- 3. Calculate SHAP Values ---
print("--- 3. Calculating SHAP Values ---")
explainer = shap.KernelExplainer(svm_model.predict_proba, shap.sample(X_train, 50))
print("Calculating SHAP values for the test set...")
shap_values = explainer.shap_values(X_test)
print("SHAP value calculation complete.\n")

# --- 4. Generate and Save Summary Plots ---
print("--- 4. Generating SHAP Summary Plots ---")
class_names = list(label_mapping.keys())

for i, class_name in enumerate(class_names):
    safe_class_name = class_name.replace(" ", "_").replace("(", "").replace(")", "")
    plot_title = f"SHAP Summary for Class: {class_name}"
    output_path = f'../data/shap_summary_{safe_class_name}.png'
    
    print(f"\nProcessing plot for class: {class_name}")
    shap.summary_plot(shap_values[:, :, i], X_test, show=False)
    
    fig = plt.gcf()
    fig.suptitle(plot_title, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f"SHAP plot for class '{class_name}' saved to {output_path}")

print("\n--- SHAP analysis complete. ---")
