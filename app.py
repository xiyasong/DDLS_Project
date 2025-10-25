import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import mygene
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", page_title="RNA-Seq Classifier & DEG Explorer")

st.title("🧬 Interactive RNA-Seq Classifier & DEG Explorer")
st.markdown("A web application for heart disease classification using gene expression data, differential expression analysis, and model interpretability with SHAP.")

import requests
import os
import scanpy as sc # New import

# --- Data Loading (Cached) ---
@st.cache_data
def load_data():
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)

    # Zenodo URLs for direct download
    gene_matrix_url = "https://zenodo.org/record/7148045/files/gene_count_matrix.csv?download=1"
    metadata_url = "https://zenodo.org/record/7148045/files/MAGE_metadata.txt?download=1"

    gene_matrix_path = os.path.join(data_dir, 'gene_count_matrix.csv')
    metadata_path = os.path.join(data_dir, 'MAGE_metadata.txt')
    normalized_gene_counts_path = os.path.join(data_dir, 'normalized_gene_counts.csv')

    # Download gene count matrix if not exists
    if not os.path.exists(gene_matrix_path):
        st.info(f"Downloading {os.path.basename(gene_matrix_path)} from Zenodo...")
        response = requests.get(gene_matrix_url, stream=True)
        response.raise_for_status() # Raise an exception for HTTP errors
        with open(gene_matrix_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success(f"Downloaded {os.path.basename(gene_matrix_path)}.")

    # Download metadata if not exists
    if not os.path.exists(metadata_path):
        st.info(f"Downloading {os.path.basename(metadata_path)} from Zenodo...")
        response = requests.get(metadata_url, stream=True)
        response.raise_for_status()
        with open(metadata_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success(f"Downloaded {os.path.basename(metadata_path)}.")

    # Generate normalized_gene_counts.csv if not exists
    if not os.path.exists(normalized_gene_counts_path):
        st.info(f"Generating {os.path.basename(normalized_gene_counts_path)} using scanpy VST-like normalization...")
        raw_counts_df = pd.read_csv(gene_matrix_path, index_col=0)
        
        # Scanpy expects samples as rows, genes as columns. Our raw_counts_df has genes as rows, samples as columns.
        # So, we transpose it first for scanpy processing, then transpose back for our app's X.
        adata = sc.AnnData(raw_counts_df.T) 
        sc.pp.normalize_total(adata, target_sum=1e4) # Normalize each cell to total counts over all genes
        sc.pp.log1p(adata) # Log-transform the data (similar to VST for visualization/ML)
        normalized_counts_df = pd.DataFrame(adata.X.T, index=raw_counts_df.index, columns=raw_counts_df.columns)
        normalized_counts_df.to_csv(normalized_gene_counts_path)
        st.success(f"Generated {os.path.basename(normalized_gene_counts_path)}.")

    normalized_counts = pd.read_csv(normalized_gene_counts_path, index_col=0)
    metadata = pd.read_csv(metadata_path)
    deg_results = pd.read_csv('data/DEG_results.csv', index_col=0)
    return normalized_counts, metadata, deg_results

@st.cache_data
def get_gene_symbols(gene_ids_list):
    mg = mygene.MyGeneInfo()
    gene_info = mg.querymany(gene_ids_list, scopes='ensembl.gene', fields='symbol', species='human', as_dataframe=True)
    gene_id_to_name = {}
    for gene_id, row in gene_info.iterrows():
        symbol = row.get('symbol')
        if isinstance(symbol, str):
            gene_id_to_name[gene_id] = symbol
        elif isinstance(symbol, list) and len(symbol) > 0:
            gene_id_to_name[gene_id] = symbol[0]
        else:
            gene_id_to_name[gene_id] = gene_id # Fallback
    return gene_id_to_name

normalized_df, metadata_df, deg_results_df = load_data()

# --- Sidebar for Global Configuration ---
st.sidebar.header("Global Configuration")
page_selection = st.sidebar.radio(
    "Go to",
    ("Data Overview & DEGs", "ML Model Analysis")
)

classification_type = st.sidebar.radio(
    "Select Classification Type:",
    ("Binary (Control vs. DCM)", "Multi-class (Control vs. DCM vs. HCM)")
)

# --- Data Preparation (Cached & Dynamic) ---
@st.cache_data
def prepare_data(classification_type, normalized_df, metadata_df):
    X = normalized_df.T
    print(f"DEBUG: Shape of X (from normalized_df.T) before cleaning: {X.shape}")
    
    # Clean index of X (sample IDs) to match metadata 'Run' column
    original_index = X.index
    cleaned_index = [idx.replace('_stringtieRef', '') for idx in original_index]
    X.index = cleaned_index
    st.write(f"DEBUG: Shape of X after cleaning index: {X.shape}")
    st.write(f"DEBUG: First 5 cleaned X index: {X.index[:5].tolist()}")

    if classification_type == "Binary (Control vs. DCM)":
        etiologies_to_include = ['Dilated cardiomyopathy (DCM)', 'Non-Failing Donor']
        label_mapping = {'Non-Failing Donor': 0, 'Dilated cardiomyopathy (DCM)': 1}
    else: # Multi-class
        etiologies_to_include = ['Dilated cardiomyopathy (DCM)', 'Hypertrophic cardiomyopathy (HCM)', 'Non-Failing Donor']
        label_mapping = {'Non-Failing Donor': 0, 'Dilated cardiomyopathy (DCM)': 1, 'Hypertrophic cardiomyopathy (HCM)': 2}

    print(f"DEBUG: Selected etiologies to include: {etiologies_to_include}")
    print(f"DEBUG: Unique etiologies in metadata_df: {metadata_df['etiology'].unique().tolist()}")

    filtered_metadata = metadata_df[metadata_df['etiology'].isin(etiologies_to_include)]
    print(f"DEBUG: Shape of filtered_metadata: {filtered_metadata.shape}")
    print(f"DEBUG: First 5 Runs in filtered_metadata: {filtered_metadata['Run'].head().tolist()}")
    common_samples = X.index.intersection(filtered_metadata['Run'])
    print(f"DEBUG: Number of common samples (X.index and filtered_metadata['Run']): {len(common_samples)}")

    X_filtered = X.loc[common_samples]
    print(f"DEBUG: Shape of X_filtered (after common samples filter): {X_filtered.shape}")
    y_labels = X_filtered.index.map(filtered_metadata.set_index('Run')['etiology'])
    y = y_labels.map(label_mapping)
    print(f"DEBUG: Shape of y (labels): {y.shape}")
    print(f"DEBUG: Unique labels in y: {y.unique().tolist()}")

    # Feature Selection: Top 1000 Most Variable Genes
    variances = X_filtered.var(axis=0)
    top_1000_variable_genes = variances.sort_values(ascending=False).head(1000).index
    X_reduced = X_filtered[top_1000_variable_genes]
    print(f"DEBUG: Shape of X_reduced (after feature selection): {X_reduced.shape}")

    # Gene Annotation
    gene_id_to_name = get_gene_symbols(X_reduced.columns.tolist())
    X_reduced.rename(columns=gene_id_to_name, inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, label_mapping, X_reduced

X_train, X_test, y_train, y_test, label_mapping, X_reduced_annotated = prepare_data(classification_type, normalized_df, metadata_df)

# --- Page 1: Data Overview & DEGs ---
if page_selection == "Data Overview & DEGs":
    st.header("1. Data Overview & PCA")

    # PCA Plot
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_reduced_annotated)
    pc_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'], index=X_reduced_annotated.index)
    pc_df['Condition'] = X_reduced_annotated.index.map(metadata_df.set_index('Run')['etiology'])

    fig_pca, ax_pca = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Condition', data=pc_df, s=80, alpha=0.9, ax=ax_pca)
    ax_pca.set_title(f'PCA of {classification_type} (Top 1000 Variable Genes)', fontsize=16)
    ax_pca.set_xlabel(f'PC1 - {pca.explained_variance_ratio_[0]*100:.2f}% variance')
    ax_pca.set_ylabel(f'PC2 - {pca.explained_variance_ratio_[1]*100:.2f}% variance')
    ax_pca.grid(True)
    st.pyplot(fig_pca)
    plt.close(fig_pca)

    # --- Differential Expression Analysis (Binary Only) ---
    if classification_type == "Binary (Control vs. DCM)":
        st.header("2. Differential Expression Analysis")
        st.subheader("Volcano Plot")
        st.image('data/volcano_plot.png')
        st.subheader("Top Differentially Expressed Genes")
        st.dataframe(deg_results_df.head(20))
    else:
        st.info("Differential Expression Analysis is currently only available for Binary (Control vs. DCM) classification.")

# --- Page 2: ML Model Analysis ---
elif page_selection == "ML Model Analysis":
    st.header("1. Machine Learning Model Analysis")
    
    model_choice = st.sidebar.selectbox(
        "Choose a Machine Learning Model:",
        ('Logistic Regression', 'Random Forest', 'Decision Tree', 'SVM (Linear Kernel)', 'Neural Network (MLP)')
    )

    models_dict = {
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42, multi_class='ovr' if classification_type == "Multi-class (Control vs. DCM vs. HCM)" else 'auto'),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM (Linear Kernel)': SVC(kernel='linear', probability=True, random_state=42, decision_function_shape='ovr' if classification_type == "Multi-class (Control vs. DCM vs. HCM)" else 'ovr'),
        'Neural Network (MLP)': MLPClassifier(random_state=42, max_iter=1000, verbose=False)
    }

    model = models_dict[model_choice]

    st.subheader(f"Results for {model_choice}")
    with st.spinner(f'Training {model_choice}...'):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred, target_names=list(label_mapping.keys())))

    if classification_type == "Binary (Control vs. DCM)":
        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    st.metric(label="ROC AUC Score", value=f"{auc_score:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(label_mapping.keys()), yticklabels=list(label_mapping.keys()), ax=ax_cm)
    ax_cm.set_title(f'Confusion Matrix: {model_choice}')
    ax_cm.set_ylabel('Actual')
    ax_cm.set_xlabel('Predicted')
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    # --- SHAP Interpretation ---
    st.header("2. Model Interpretation (SHAP Values)")
    st.info("Calculating SHAP values can take several minutes, especially for complex models. Please be patient.")
    if st.button("Calculate SHAP Values"):
        with st.spinner('Calculating SHAP values...'):
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 50, random_state=42), random_state=42)
            shap_values = explainer.shap_values(X_test)

        if classification_type == "Binary (Control vs. DCM)":
            st.subheader(f"SHAP Summary for {model_choice} (Class: Dilated cardiomyopathy (DCM))")
            fig_shap, ax_shap = plt.subplots(figsize=(10, 8))

            shap_to_plot = None
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_to_plot = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 and shap_values.shape[2] == 2:
                shap_to_plot = shap_values[:, :, 1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                shap_to_plot = shap_values
            
            if shap_to_plot is not None:
                shap.summary_plot(shap_to_plot, X_test, show=False, plot_type="dot")
                st.pyplot(fig_shap)
            else:
                st.warning("SHAP output format not recognized for binary classification.")
            plt.close(fig_shap)
        else: # Multi-class
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                for i, class_name in enumerate(list(label_mapping.keys())):
                    st.subheader(f"SHAP Summary for {model_choice} (Class: {class_name})")
                    fig_shap, ax_shap = plt.subplots(figsize=(10, 8))
                    shap.summary_plot(shap_values[:, :, i], X_test, show=False, plot_type="dot")
                    st.pyplot(fig_shap)
                    plt.close(fig_shap)
            else:
                st.warning("SHAP output format not recognized for multi-class classification.")


st.sidebar.info("Adjust settings above and click 'Train and Evaluate' to update results.")
