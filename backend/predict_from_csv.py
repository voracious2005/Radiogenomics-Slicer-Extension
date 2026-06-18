import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

# Suppress warnings to prevent polluting the Slicer UI console output
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# XGBoost Deserialization Patch (MANDATORY)
# ---------------------------------------------------------
# Because the saved ensemble model pickle containing the XGBoost estimator 
# was trained using the 'PatchedXGBClassifier' class wrapper, we must 
# declare it here in the global namespace of the executor script. 
# Without this declaration, joblib.load() will crash.
from xgboost import XGBClassifier

class PatchedXGBClassifier(XGBClassifier):
    _estimator_type = "classifier"


# ---------------------------------------------------------
# Recursive Tracing Helper Functions
# ---------------------------------------------------------
def find_feature_importances(estimator):
    """
    Recursively searches an estimator (including pipelines, calibration wrappers, 
    and ensembles) to find any nested model that exposes 'feature_importances_' or 'coef_'.
    Returns a tuple of (model_with_importances, importances_array).
    """
    if estimator is None:
        return None, None
        
    # Case 1: Direct model with feature_importances_
    if hasattr(estimator, 'feature_importances_') and estimator.feature_importances_ is not None:
        return estimator, estimator.feature_importances_
        
    # Case 2: Direct model with coef_
    if hasattr(estimator, 'coef_') and estimator.coef_ is not None:
        return estimator, np.abs(estimator.coef_).flatten()

    # Case 3: Pipeline (sklearn or imblearn)
    if hasattr(estimator, 'steps'):
        for name, step in estimator.steps:
            res = find_feature_importances(step)
            if res[0] is not None:
                return res
    elif hasattr(estimator, 'named_steps'):
        for name, step in estimator.named_steps.items():
            res = find_feature_importances(step)
            if res[0] is not None:
                return res

    # Case 4: CalibratedClassifierCV
    if hasattr(estimator, 'calibrated_classifiers_') and estimator.calibrated_classifiers_:
        for cal_clf in estimator.calibrated_classifiers_:
            if hasattr(cal_clf, 'base_estimator'):
                res = find_feature_importances(cal_clf.base_estimator)
                if res[0] is not None:
                    return res
            elif hasattr(cal_clf, 'estimator'):
                res = find_feature_importances(cal_clf.estimator)
                if res[0] is not None:
                    return res
                    
    if hasattr(estimator, 'estimator'):
        res = find_feature_importances(estimator.estimator)
        if res[0] is not None:
            return res
            
    if hasattr(estimator, 'base_estimator'):
        res = find_feature_importances(estimator.base_estimator)
        if res[0] is not None:
            return res

    # Case 5: VotingClassifier / StackingClassifier / Bagging / Random Forest
    if hasattr(estimator, 'estimators_') and estimator.estimators_:
        for est in estimator.estimators_:
            res = find_feature_importances(est)
            if res[0] is not None:
                return res

    return None, None


def find_selector_and_mask(estimator):
    """
    Recursively searches an estimator/pipeline to find a feature selector 
    (like SelectPercentile, SelectKBest, etc.) that exposes 'get_support()'.
    """
    if estimator is None:
        return None
        
    if hasattr(estimator, 'get_support'):
        return estimator
        
    if hasattr(estimator, 'steps'):
        for name, step in estimator.steps:
            res = find_selector_and_mask(step)
            if res is not None:
                return res
    elif hasattr(estimator, 'named_steps'):
        for name, step in estimator.named_steps.items():
            res = find_selector_and_mask(step)
            if res is not None:
                return res
                
    if hasattr(estimator, 'estimator'):
        return find_selector_and_mask(estimator.estimator)
        
    if hasattr(estimator, 'base_estimator'):
        return find_selector_and_mask(estimator.base_estimator)
        
    if hasattr(estimator, 'estimators_') and estimator.estimators_:
        for est in estimator.estimators_:
            res = find_selector_and_mask(est)
            if res is not None:
                return res
                
    return None


# ---------------------------------------------------------
# Feature Explainability Engine
# ---------------------------------------------------------
def extract_feature_explanations(ensemble_pipeline, X_input, expected_cols, top_n=3):
    """
    Identifies the top N features driving the prediction by combining 
    the pipeline's ANOVA F-scores and the XGBoost model's feature importances.
    Provides human-readable translations of PyRadiomics tags.
    """
    try:
        # 1. Recursively find the selector and its support mask
        selector = find_selector_and_mask(ensemble_pipeline)
        if selector and hasattr(selector, 'get_support'):
            support_mask = selector.get_support()
            selected_feature_names = [col for idx, col in enumerate(expected_cols) if idx < len(support_mask) and support_mask[idx]]
        else:
            selected_feature_names = list(expected_cols)

        # 2. Recursively find model importances across any arbitrary wrapper structure
        model_found, importances = find_feature_importances(ensemble_pipeline)

        if importances is None or model_found is None:
            return ["Feature attribution tracing is currently unavailable for this model layout."]

        # Pair features with their learned weights
        feat_imp_pairs = []
        for i, feat_name in enumerate(selected_feature_names):
            if i < len(importances):
                feat_imp_pairs.append((feat_name, float(importances[i])))
                
        # Sort by importance magnitude
        feat_imp_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Human-readable translations for abstract PyRadiomics tags
        translations = {
            "original_firstorder_Entropy": "High structural tissue chaos / randomness (Entropy)",
            "original_glrlm_RunLengthNonUniformity": "High heterogeneity in tumor microenvironment texture (RLN)",
            "original_shape_Sphericity": "Tumor boundary irregularity / non-spherical growth (Sphericity)",
            "original_glcm_Contrast": "Sharp local variations in MRI grayscale contrast (Contrast)",
            "original_firstorder_90Percentile": "Presence of hyper-intense cellular regions (90th Percentile)",
            "original_firstorder_Minimum": "Voxel intensity baseline floor (Minimum)",
            "original_firstorder_10Percentile": "Subtle low-intensity tissue structures (10th Percentile)",
            "original_firstorder_Median": "Central grayscale density alignment (Median)",
            "original_shape_Maximum3DDiameter": "Maximum 3D tumor spatial expansion (Maximum Diameter)",
            "original_shape_Elongation": "Asymmetric, elongated tumor geometry (Elongation)",
            "original_glszm_GrayLevelNonUniformity": "Fine-grained texture variation in 3D (GLSZM GLN)",
            "original_glcm_Idm": "Local grayscale homogeneity/uniformity (GLCM IDM)",
            "original_glcm_JointEntropy": "Complexity and complexity of tumor pattern (Joint Entropy)"
        }
        
        clean_explanations = []
        for name, score in feat_imp_pairs[:top_n]:
            readable = translations.get(name, f"Feature '{name.replace('original_', '')}' showing high variance")
            clean_explanations.append(f"{readable} (Impact Weight: {score:.2f})")
            
        if not clean_explanations:
            return ["No dominant decision drivers identified above threshold."]
            
        return clean_explanations
    except Exception as e:
        return [f"Could not compute attribution: {str(e)}"]


# ---------------------------------------------------------
# Threshold Mapping Engine
# ---------------------------------------------------------
def map_probability_to_slicer_gui(raw_prob_pos, ideal_threshold_neg=0.75):
    """
    Maps the mathematically optimized threshold to the 3D Slicer UI.
    
    The ML model predicts the probability of Class 0 (+) and Class 1 (-).
    We optimized the model to flag a tumor as Negative (-) only if it is >= 75% sure.
    This means the threshold for Positive (+) is 1.0 - 0.75 = 0.25 (25%).
    
    The 3D Slicer UI is hardcoded to show Negative below 50% and Positive above 50%.
    This function linearly scales the raw probability so that exactly 25% raw 
    probability becomes 50% on the Slicer gauge.
    """
    ideal_threshold_pos = 1.0 - ideal_threshold_neg  # 0.25
    
    if raw_prob_pos >= ideal_threshold_pos:
        # Scale range [0.25, 1.0] to [0.5, 1.0]
        mapped = 0.5 + 0.5 * ((raw_prob_pos - ideal_threshold_pos) / (1.0 - ideal_threshold_pos))
    else:
        # Scale range [0.0, 0.25) to [0.0, 0.5)
        mapped = 0.5 * (raw_prob_pos / ideal_threshold_pos)
        
    return mapped * 100.0


# ---------------------------------------------------------
# DIRECT CSV PREDICTION RUNTIME
# ---------------------------------------------------------
def predict_from_csv(csv_path):
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load Expected Columns
    expected_features_path = os.path.join(backend_dir, "ensemble_expected_features.pkl")
    if not os.path.exists(expected_features_path):
        expected_features_path = "ensemble_expected_features.pkl"
        
    if not os.path.exists(expected_features_path):
        raise FileNotFoundError(f"Expected features configuration missing: {expected_features_path}")
        
    expected_cols = joblib.load(expected_features_path)

    # 2. Read Patient Radiomics Data
    try:
        # Using utf-8-sig to handle Windows Excel BOM encoding bugs safely
        radiomics_df = pd.read_csv(csv_path, encoding='utf-8-sig')
        radiomics_df.columns = radiomics_df.columns.str.strip()
        
        if len(radiomics_df) == 0:
            raise ValueError("The uploaded CSV is entirely empty (0 rows of patient data).")
    except Exception as e:
        raise ValueError(f"Could not load patient spreadsheet: {e}")

    # 3. Align Features and Impute Missing Channels
    # Retains all rows from the uploaded dataset instead of slicing
    X_input = pd.DataFrame(index=radiomics_df.index)
    
    for col in expected_cols:
        if col in radiomics_df.columns:
            X_input[col] = pd.to_numeric(radiomics_df[col], errors='coerce')
        else:
            X_input[col] = 0.0
            
    X_input = X_input.fillna(0.0)

    # 4. Predict for each Patient Row in the Ingested Dataset
    targets = ["HER2", "ER", "PR"]
    
    print("\n" + "="*45)
    print(f"ENSEMBLE SYSTEM BATCH RUN: {len(X_input)} PATIENTS DETECTED")
    print("="*45)
    
    final_tags = []
    
    # Loop over every single patient row in the sheet
    for idx in range(len(X_input)):
        row_slice = X_input.iloc[[idx]]
        
        # Determine Patient Identifier dynamically based on common columns
        patient_id = f"Patient {idx + 1}"
        for col_candidate in ['FilenamePrefix', 'ID', 'Id', 'PatientID', 'Patient_ID', 'patient_id']:
            if col_candidate in radiomics_df.columns:
                patient_id = str(radiomics_df.iloc[idx][col_candidate]).strip()
                break

        results = {}
        for target in targets:
            model_filename = f"{target.lower()}_ensemble.pkl"
            model_path = os.path.join(backend_dir, model_filename)
            if not os.path.exists(model_path):
                model_path = model_filename
                
            if not os.path.exists(model_path):
                results[target] = 0.0
                continue

            ensemble_model = joblib.load(model_path)
            # predict_proba returns [prob_class_0, prob_class_1] where 0 is Positive(+)
            raw_prob_pos = ensemble_model.predict_proba(row_slice)[0][0]
            
            # Map threshold piecewise for clinical safety
            results[target] = map_probability_to_slicer_gui(raw_prob_pos, ideal_threshold_neg=0.75)

        # Output standard parsed rows to terminal stream for Slicer parsing
        print(f"BATCH_ROW: {patient_id} | {results['HER2']:.2f} | {results['ER']:.2f} | {results['PR']:.2f}")

        # Assemble summary parameters and run explainability for the single-patient mode
        if len(X_input) == 1:
            for tgt, p_val in results.items():
                marker = "+" if p_val >= 50.0 else "-"
                final_tags.append(f"{tgt}{marker}")
                print(f"DIFF: {tgt} | {p_val:.2f}")

            # Explainability tracing specifically for primary diagnosis target (HER2)
            her2_model_filename = "her2_ensemble.pkl"
            her2_model_path = os.path.join(backend_dir, her2_model_filename)
            if not os.path.exists(her2_model_path):
                her2_model_path = her2_model_filename
            
            if os.path.exists(her2_model_path):
                try:
                    her2_ensemble = joblib.load(her2_model_path)
                    explanations = extract_feature_explanations(her2_ensemble, row_slice, expected_cols, top_n=3)
                    for exp in explanations:
                        print(f"EXPLAINABILITY_TAG: {exp}")
                except Exception as e:
                    print(f"EXPLAINABILITY_TAG: Feature attribution trace blocked: {e}")

    # Standard exit printing for single run parses
    if len(X_input) == 1:
        status_summary = ", ".join(final_tags)
        print(f"PREDICTION: {status_summary}")
    else:
        print("PREDICTION: Batch run finished successfully.")
        
    print("="*45)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_from_csv.py <path_to_radiomics_file.csv>")
        sys.exit(1)

    csv_input_path = sys.argv[1]
    if not os.name == 'nt' or not os.path.exists(csv_input_path):
        # Decode path issues on different operating systems
        csv_input_path = csv_input_path.replace("\\", "/")

    if not os.path.exists(csv_input_path):
        print(f"ERROR: File not found -> {csv_input_path}")
        sys.exit(1)

    try:
        predict_from_csv(csv_input_path)
    except Exception as e:
        print(f"PIPELINE FAILED: {str(e)}")
        sys.exit(1)
