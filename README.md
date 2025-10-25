# Interactive RNA-Seq Classifier and DEG Explorer for Heart Disease

This repository contains the code for an interactive web application designed to classify heart disease using RNA-Seq gene expression data. The project focuses on the Magnetique dataset, providing tools for data exploration, machine learning model training, differential gene expression analysis, and model interpretability.

The application is built with Streamlit, making the complex bioinformatics and machine learning pipeline accessible through a user-friendly interface.

The web server is accessible on: https://xiyasong-ddls-project-app-k830az.streamlit.app/<img width="468" height="70" alt="image" src="https://github.com/user-attachments/assets/8bd0e24a-98c8-4a07-9db1-19c7d9cab962" />


## Features

*   **Dynamic Classification:** Choose between binary (Control vs. Dilated Cardiomyopathy) and multi-class (Control vs. DCM vs. Hypertrophic Cardiomyopathy) classification.
*   **Machine Learning Models:** Evaluate various scikit-learn models including Logistic Regression, Random Forest, Decision Tree, SVM (Linear Kernel), and a Multi-Layer Perceptron (MLP) Neural Network.
*   **Data Exploration:** Visualize sample separation with Principal Component Analysis (PCA).
*   **Differential Expression Analysis (DEG):** View volcano plots and top differentially expressed genes for binary classification.
*   **Model Interpretability:** Understand model predictions with on-demand SHAP (SHapley Additive exPlanations) summary plots, showing key gene contributions.
*   **Gene Annotation:** All gene IDs are automatically annotated with human-readable gene symbols for enhanced interpretability.

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/xiyasong/DDLS_Project.git
    cd DDLS_Project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Streamlit web application:

1.  **Ensure your virtual environment is activated** (if you created one).
2.  **Run the app:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your default web browser.

## Project Structure

*   `app.py`: The main Streamlit web application.
*   `scripts/`: Contains various Python and R scripts used for data preprocessing, analysis, and model training.
    *   `01_normalize.R`: R script for DESeq2 VST normalization.
    *   `02_DEG_analysis.R`: R script for Differential Expression Analysis.
    *   `run_final_analysis.py`: Consolidated Python script for multi-class ML model training and evaluation.
    *   `07_shap_analysis.py`: Python script for SHAP interpretation.
    *   `01a_Advanced_PCA_Analysis.ipynb`: Jupyter notebook for advanced PCA.
    *   `tensorflow.ipynb`: Jupyter notebook for TensorFlow DNN (run on Google Colab).
*   `data/`: Contains derived data files (e.g., `DEG_results.csv`, `volcano_plot.png`) and will dynamically download raw data from Zenodo.
*   `requirements.txt`: Lists all Python dependencies.
*   `report.md`: The comprehensive project report.
*   `GEMINI.md`: Internal project plan and notes.

## Data Source

The project utilizes the Magnetique dataset, available on Zenodo:
[https://zenodo.org/record/7148045](https://zenodo.org/record/7148045)

Raw data files (`gene_count_matrix.csv` and `MAGE_metadata.txt`) are downloaded dynamically by `app.py` from this Zenodo record.

## Project Report

A detailed project report, covering background, methods, results, and discussion, is available in `report.md`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was developed with significant assistance from the Gemini-CLI AI agent.
