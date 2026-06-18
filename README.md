# Radiogenomics-Slicer-Extension
Standalone Molecular Subtyping Extension for 3D Slicer using Calibrated XGBoost + MLP Ensembles
Technical Pipeline & Development Workflow

The Radiogenomics Genomic Subtyper is a production-grade, standalone desktop extension integrated directly into the 3D Slicer medical platform. This module bridges the gap between raw radiomic feature extractions and actionable clinical decision support, moving beyond isolated machine learning models to establish a reproducible, secure, and intuitive diagnostic workspace.

End-to-End Processing Pipeline

The computational pipeline executes in a modular, sequential manner to process patient data securely without altering Slicer's native environment:

1. Pre-Flight File Validation and Encoding Safeguard

When a spreadsheet is uploaded, a structural validator inspects the columns before any machine learning process is initiated. The script screens the file for mandatory structural biomarkers (such as sphericity, contrast, and entropy metrics).
To prevent platform-level failures caused by varying operating systems and spreadsheet tools, the ingestion layer automatically neutralizes Byte Order Mark (BOM) variations and ignores blank trailing rows. If the spreadsheet does not match the standardized parameters, the UI issues a clean warning dialog and halts execution, avoiding raw application tracebacks.

2. Vector Realignment and Standardization

Medical data from different clinics often contains missing features or out-of-order column layouts. The pipeline compares the incoming spreadsheet with a frozen, ordered reference of expected features. It automatically aligns the rows, fills missing data points safely with zero-values, and passes the normalized array through a StandardScaler to scale features based on historical training boundaries.

3. Statistical Dimensionality Reduction

To protect the machine learning models from overfitting and scale-skewing, the pipeline applies univariate feature selection. An Analysis of Variance (ANOVA) test calculates the statistical significance of each radiomic metric relative to the target patient populations. The pipeline automatically filters and discards the least informative variables, passing only the top statistically significant features to the active estimators.

4. Parallel Hybrid Ensemble Processing

The selected features are passed simultaneously to a parallel, dual-topology ensemble.

The first branch utilizes a high-capacity Extreme Gradient Boosting (XGBoost) estimator to map decision split boundaries.

The second branch utilizes a deep Multi-Layer Perceptron (MLP) Neural Network to capture continuous non-linear feature interactions.
The outputs of both estimators are combined using weighted soft voting, giving high weight to the gradient booster while retaining the continuous pattern feedback from the neural network.

5. Target-Specific Coordinate Routing

Through testing on authentic patient profiles, we identified distinct indexing systems between different target receptors:

For HER2, class index 0 represents a Positive status, and class index 1 represents a Negative status.

For ER and PR hormone receptors, class index 1 represents a Positive status, and class index 0 represents a Negative status.
The backend pipeline dynamically routes the indexing arrays based on the target marker being evaluated, entirely resolving target inversion bugs.

6. Linear Threshold Calibration

To map the machine learning decision boundaries (such as a highly strict 25% positive confidence boundary for HER2) onto Slicer's default 50% progress bar layouts, an inline piecewise linear mapping transformation is executed. This scales and shifts probabilities on-the-fly, ensuring that any prediction mathematically deemed positive falls above 50% on Slicer's visual gauges, and any prediction deemed negative falls below 50%.

7. Clinical Confidence Calibration

To prevent display anomalies where a certain negative prediction shows low visual progress (e.g., a 10% positive probability showing as "10% Confidence"), the pipeline implements a correction layer. If the diagnosis is negative, the gauge calculates true diagnostic confidence as 100% minus the visual probability. This translates predictions into accurate clinical confidence rankings: High, Moderate, or Low confidence.

Development and Refinement Workflow

Building this extension required moving beyond static Jupyter Notebooks into clinical-grade software engineering. Our development process involved navigating and resolving several technical challenges:

Moving from Jupyter to 3D Slicer Desktop

The primary objective of this project was to bring predictive modeling into the active workspace of a neuro-radiologist. We transformed a sequence of Python models into an interactive, event-driven 3D Slicer scripted module using Qt and CTK frameworks. The resulting extension responds dynamically to user interactions, managing file transfers and calculations without locking Slicer's primary workspace thread.

Implementing Portable Zero-Configuration Setup

A major challenge was ensuring that the extension runs out-of-the-box on different computers without manual environment setup. We designed a self-repairing dependency analyzer. On startup, Slicer dynamically scans the user's local Python interpreter for mandatory machine learning libraries. If any package is missing, the tool prompts the user and auto-installs it directly inside Slicer's internal virtual environment. Additionally, we removed all platform-specific directory paths, allowing Slicer to locate its executable naturally on Windows, macOS, and Linux.

Resolving the Hormone Receptor Status Inversion

During early validation trials, the system successfully predicted HER2, but consistently flipped the positive and negative diagnoses for ER and PR. By tracking the probabilities directly to the underlying model layers, we uncovered the target-specific coordinate routing discrepancies. Implementing explicit branching routes based on the target marker fully resolved this issue, resulting in highly accurate predictions matching clinical standards.

Lightweight Model Explainability (XAI)

To make the AI predictions trustworthy to clinicians, we integrated model explainability. Because heavy frameworks like SHAP frequently crash Slicer's native event loops, we built a lightweight, recursive parser. The algorithm traces backward through standard scaling pipelines and voting ensembles to extract raw feature split weights directly from the active XGBoost models. It then pairs these weights with a custom translation dictionary, translating abstract radiomic parameters (such as Run Length Non-Uniformity) into plain clinical diagnostic terms (such as structural tissue texture heterogeneity).

Dynamic UI Adaptability and Export Routines

To accommodate different clinical environments, we upgraded the tool to support both single-patient diagnostics and cohort audits. The input handler automatically analyzes the shape of the uploaded dataset.

For a Single Patient, Slicer presents dial gauges and explainability diagnostics.

For a Patient Cohort, the interface dynamically switches to an interactive, multi-column database table.
Finally, we engineered clinical report generators, enabling doctors to export structured text summaries complete with signatures, diagnostic classifications, and legal disclaimers.
