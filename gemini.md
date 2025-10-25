# Gemini Project Plan: Interactive RNA-Seq Classifier and DEG Explorer

This plan outlines the steps to implement the project described in `DDLS_project_design.pdf` and aligns with the assignment instructions.

## Project Overview
**Project Title:** Interactive RNA-Seq Classifier and DEG Explorer for Heart Disease Using the Magnetique Dataset.
**Scientific Question:** Can ML models on RNA-seq data distinguish cardiac disease from control samples, identify important genes using methods like SHAP, and confirm differential expression?
**Dataset:** Magnetique dataset (heart failure study), specifically comparing Dilated Cardiomyopathy (DCM), Hypertrophic Cardiomyopathy (HCM) vs. non-failing donor (NF).

## Assignment Phases & Deliverables

### Phase I — Dataset Exploration (Completed)
**Goal:** Understand the dataset, summarize distributions, imbalances, and key features. Split data into train/test sets.
**Completed Actions:**
1.  Loaded `gene_count_matrix.csv` and `MAGE_metadata.txt`.
2.  Performed data normalization using DESeq2 (R script `01_normalize.R`).
3.  Conducted advanced PCA analysis (`01a_Advanced_PCA_Analysis.ipynb`) to check for batch effects and visualize class separation.
4.  Identified top genes driving principal components.
5.  Annotated gene IDs with symbols using `mygene`.
**Deliverable:** Data exploration visualizations (PCA plots, distribution summaries).

### Phase II — Model Training, Evaluation & Improvement (Completed)
**Goal:** Train and evaluate classification models for 3-class problem (DCM, HCM, NF).
**Completed Actions:**
1.  Implemented a consolidated Python script (`run_final_analysis.py`) for multi-class classification.
2.  Trained and evaluated five models: Logistic Regression, Random Forest, Decision Tree, SVM (Linear Kernel), and Neural Network (MLPClassifier).
3.  **Note:** The TensorFlow DNN model was successfully run on Google Colab (using `scripts/ternsorflow.ipynb`) due to GPU availability, demonstrating its potential.
4.  Generated performance metrics (classification reports, ROC AUC scores) and visualizations (confusion matrices, macro-averaged ROC curves).
5.  Interpreted the SVM model using SHAP values (`07_shap_analysis.py`) to identify key contributing genes for each class (for the 3-class problem).
6.  **Additionally, a SHAP summary plot for the binary classification (Control vs. DCM) was generated (data/shap_summary_plot_binary_gene_names.png). This plot shows that genes like HBA2 and HBA1 (high expression associated with DCM) and SERPINA3, IL1RL1, SFRP4 (high expression associated with Control) are key drivers of the model's predictions.**
**Deliverable:** Working models with measurable performance, evidence of improvement attempts (multi-class, feature selection), and model interpretation (SHAP).

### Phase III — Accessibility Wrapping (Web Application - Option A)
**Goal:** Build a simple web application wrapping the final workflow.
**Planned Actions:**
1.  Develop a Streamlit web application (`app.py`).
2.  Integrate data loading, preprocessing, model selection, and result visualization.
3.  Allow users to upload data (future enhancement, currently uses pre-loaded data).
4.  Display PCA plots, DEG volcano plots, and model evaluation results (confusion matrices, ROC curves, SHAP plots).
**Deliverable:** A working web application with basic documentation.

## FAIR Data and Open Science Recommendations (To be implemented)
1.  **Public GitHub Repository:** Create a public repository for the project.
2.  **GitHub Topic:** Add `ddls-course-2025` topic.
3.  **README File:** Include an overview and usage instructions.
4.  **Code Documentation:** Document code properly.
5.  **Permissive License:** Add an MIT license.
6.  **Reproducibility:** Provide script/instructions to download dataset.
7.  **Model Weights:** Upload trained model weights to GitHub Releases or Zenodo (if applicable and feasible).

## Final Deliverables (Report Content)
**Report Title:** Interactive RNA-Seq Classifier and DEG Explorer for Heart Disease Using the Magnetique Dataset.
**Report Structure (max 5 pages main text, unlimited appendices):**
1.  **Abstract (≤100 words):** Problem, method, results.
2.  **Background and Motivation:** Why this dataset/task was chosen.
3.  **Dataset Summary:** Data source, preprocessing steps, data splits, distributions.
4.  **Method Description:** Workflow, models used, evaluation metrics.
5.  **Results:** Figures, tables, performance metrics vs. baseline (binary vs. multi-class comparison).
6.  **Conclusion & Discussion:** Findings, limitations, future directions.
7.  **Data and Code Availability:** Links to dataset and GitHub repository (per FAIR guidelines).
8.  **Acknowledgments:** Contributions, support, and note on GenAI tools used.
9.  **References:** Relevant literature.
10. **Appendices:** AI deep research log (mandatory - this transcript), prompts, agent transcripts, extra figures.

## Agent Demo
**Requirement:** 3–5 min screen recording (or asciinema) showing Gemini-CLI being used for dataset exploration and/or evaluation.

## Accessibility Wrapper & Repository
**Requirement:**
*   A simple web app (`app.py`) wrapping the final workflow.
*   A public GitHub repository containing code, documentation, and reproducibility instructions.