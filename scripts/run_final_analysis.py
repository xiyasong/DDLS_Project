import mygene

# --- 1. Data Preparation & Feature Selection ---
print('--- 1. Preparing Data and Selecting Features ---')
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

print("Selecting top 1000 most variable genes...")
variances = X_filtered.var(axis=0)
top_1000_variable_genes = variances.sort_values(ascending=False).head(1000).index
X_reduced = X_filtered[top_1000_variable_genes]
print(f'Using {X_reduced.shape[1]} genes as features.\n')

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

# --- 2. PCA Visualization ---
print('--- 2. Visualizing Class Separation with PCA ---')
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_reduced)
pc_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'], index=X_reduced.index)
pc_df['Condition'] = y_labels

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Condition', data=pc_df, s=80, alpha=0.9)
plt.title('PCA of 3 Classes (Top 1000 Variable Genes)', fontsize=16)
plt.xlabel(f'PC1 - {pca.explained_variance_ratio_[0]*100:.2f}% variance')
plt.ylabel(f'PC2 - {pca.explained_variance_ratio_[1]*100:.2f}% variance')
plt.grid(True)
pca_plot_path = '../data/pca_3_class_final.png'
plt.savefig(pca_plot_path)
print(f"PCA plot saved to {pca_plot_path}\n")
plt.close()

# --- 3. Train, Evaluate, and Plot Models ---
print('--- 3. Training and Evaluating All Models ---')
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42, stratify=y)

models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM (Linear Kernel)': SVC(kernel='linear', probability=True, random_state=42),
    'Neural Network (MLP)': MLPClassifier(random_state=42, max_iter=1000, verbose=False)
}

roc_plot_fig, roc_plot_ax = plt.subplots(figsize=(12, 10))
linestyles = ['-', '--', ':', '-.', '-']

for (name, model), linestyle in zip(models.items(), linestyles):
    print(f'--- Evaluating {name} ---')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print(f'\n{name} Test Set Evaluation:\n')
    print(classification_report(y_test, y_pred, target_names=label_mapping.keys()))
    
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    print(f'ROC AUC Score (One-vs-Rest, Macro): {auc:.4f}\n')

    # Calculate Macro-Averaged ROC Curve for the combined plot
    n_classes = len(np.unique(y))
    fpr, tpr = dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_proba[:, i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    roc_plot_ax.plot(all_fpr, mean_tpr, label=f'{name} (macro AUC = {auc:.4f})', linestyle=linestyle, linewidth=2)

    # Compute and Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_fig, cm_ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys(), ax=cm_ax)
    cm_ax.set_title(f'Confusion Matrix: {name}', fontsize=14)
    cm_ax.set_ylabel('Actual')
    cm_ax.set_xlabel('Predicted')
    cm_plot_path = f'../data/confusion_matrix_{name.replace(" ", "_").replace("(", "").replace(")", "")}.png'
    cm_fig.savefig(cm_plot_path)
    print(f"Confusion matrix for {name} saved to {cm_plot_path}\n")
    plt.close(cm_fig)

# Finalize and save the combined ROC plot
roc_plot_ax.plot([0, 1], [0, 1], 'r--', label='Chance')
roc_plot_ax.set_xlim([0.0, 1.0])
roc_plot_ax.set_ylim([0.0, 1.05])
roc_plot_ax.set_xlabel('False Positive Rate', fontsize=12)
roc_plot_ax.set_ylabel('True Positive Rate', fontsize=12)
roc_plot_ax.set_title('Multi-Class Model Comparison: Macro-Averaged ROC Curves', fontsize=16)
roc_plot_ax.legend(loc="lower right")
roc_plot_ax.grid(True)
roc_plot_path = '../data/multiclass_roc_curves_final.png'
roc_plot_fig.savefig(roc_plot_path)
print(f"Combined ROC curve plot saved to {roc_plot_path}")
plt.close(roc_plot_fig)

print('--- Final analysis complete. ---')
