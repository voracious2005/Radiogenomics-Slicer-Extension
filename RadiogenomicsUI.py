import os
import qt
import ctk
import slicer
import subprocess
import traceback
import shutil
from datetime import datetime
from slicer.ScriptedLoadableModule import *


class RadiogenomicsUI(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Radiogenomics Genomic Subtyper"
        self.parent.categories = ["Neuro-Oncology"]
        self.parent.dependencies = []
        self.parent.contributors = ["Aswin Sunil (B230216BT), BSED, NITC"]
        self.parent.helpText = "Standalone Molecular Subtyping Module: Upload Extracted 3D Radiomics (.csv) -> Run Weighted Ensembles -> Predict Subtypes."


class RadiogenomicsUIWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        try:
            # SECTION 1: PIPELINE CONTROLS (CSV ONLY)
            # ================================================================
            parametersCollapsibleButton = ctk.ctkCollapsibleButton()
            parametersCollapsibleButton.text = "Clinical Subtyping Pipeline"
            self.layout.addWidget(parametersCollapsibleButton)
            
            # Slicer native form layout
            parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

            self.uploadButton = qt.QPushButton("1. Upload Extracted Radiomics (.csv)")
            parametersFormLayout.addRow(self.uploadButton)

            self.filePathBox = qt.QLineEdit()
            self.filePathBox.setReadOnly(True)
            self.filePathBox.placeholderText = "No file selected..."
            parametersFormLayout.addRow("Selected File:", self.filePathBox)

            self.predictButton = qt.QPushButton("2. Predict Genomic Subtype")
            self.predictButton.enabled = False
            # Native styling uses fonts instead of CSS
            predictFont = self.predictButton.font
            predictFont.setBold(True)
            self.predictButton.setFont(predictFont)
            parametersFormLayout.addRow(self.predictButton)

            self.statusLabel = qt.QLabel("Status: Waiting for radiomics data...")
            parametersFormLayout.addRow(self.statusLabel)

            # SECTION 2: DOCKED DIAGNOSTIC RESULT PANEL
            # ================================================================
            self.resultsCollapsible = ctk.ctkCollapsibleButton()
            self.resultsCollapsible.text = "Diagnostic Results"
            self.resultsCollapsible.collapsed = True
            self.layout.addWidget(self.resultsCollapsible)
            resultsLayout = qt.QVBoxLayout(self.resultsCollapsible)

            # --- BATCH CLINICAL DATA TABLE VIEW (Hidden by default) ---
            self.batchTable = qt.QTableWidget()
            self.batchTable.setColumnCount(4)
            self.batchTable.setHorizontalHeaderLabels(["Patient Identifier", "HER2 Profile", "ER Profile", "PR Profile"])
            self.batchTable.horizontalHeader().setSectionResizeMode(qt.QHeaderView.Stretch)
            self.batchTable.verticalHeader().setVisible(False)
            self.batchTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
            self.batchTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
            self.batchTable.setVisible(False)
            resultsLayout.addWidget(self.batchTable)

            # Main Single-Patient Result Card (Visible by default)
            self.resultCard = qt.QFrame()
            self.resultCard.setFrameShape(qt.QFrame.StyledPanel)
            self.resultCard.setFrameShadow(qt.QFrame.Raised)
            resultCardLayout = qt.QVBoxLayout(self.resultCard)
            resultsLayout.addWidget(self.resultCard)

            headerLabel = qt.QLabel("Diagnostic Report")
            headerFont = headerLabel.font
            headerFont.setPointSize(12)
            headerFont.setBold(True)
            headerLabel.setFont(headerFont)
            headerLabel.setAlignment(qt.Qt.AlignCenter)
            resultCardLayout.addWidget(headerLabel)

            # Basic Patient Info
            patientInfoFrame = qt.QFrame()
            patientInfoLayout = qt.QFormLayout(patientInfoFrame)
            
            self.resultTimestamp = qt.QLabel("—")
            patientInfoLayout.addRow("Timestamp:", self.resultTimestamp)

            self.resultFilename = qt.QLabel("—")
            self.resultFilename.setWordWrap(True)
            patientInfoLayout.addRow("Input Data:", self.resultFilename)

            resultCardLayout.addWidget(patientInfoFrame)

            # --- 1. PRIMARY FOCUS CARD (HER2) ---
            self.her2Card = qt.QFrame()
            self.her2Card.setFrameShape(qt.QFrame.StyledPanel)
            self.her2Card.setFrameShadow(qt.QFrame.Sunken)
            her2Layout = qt.QVBoxLayout(self.her2Card)
            
            her2Header = qt.QLabel("PRIMARY DIAGNOSIS: HER2 STATUS")
            boldFont = her2Header.font
            boldFont.setBold(True)
            her2Header.setFont(boldFont)
            her2Header.setAlignment(qt.Qt.AlignCenter)
            her2Layout.addWidget(her2Header)
            
            self.her2Badge = qt.QLabel("Awaiting Analysis")
            badgeFont = self.her2Badge.font
            badgeFont.setPointSize(18)
            badgeFont.setBold(True)
            self.her2Badge.setFont(badgeFont)
            self.her2Badge.setAlignment(qt.Qt.AlignCenter)
            her2Layout.addWidget(self.her2Badge)
            
            # Native OS Progress Bar
            self.her2Pbar = qt.QProgressBar()
            self.her2Pbar.setRange(0, 100)
            self.her2Pbar.setValue(0)
            self.her2Pbar.setTextVisible(False)
            her2Layout.addWidget(self.her2Pbar)
            
            self.her2Tier = qt.QLabel("")
            self.her2Tier.setFont(boldFont)
            self.her2Tier.setAlignment(qt.Qt.AlignCenter)
            her2Layout.addWidget(self.her2Tier)

            resultCardLayout.addWidget(self.her2Card)

            # --- 2. SECONDARY MARKERS CARD (ER & PR) ---
            self.secondaryCard = qt.QFrame()
            self.secondaryCard.setFrameShape(qt.QFrame.StyledPanel)
            self.secondaryCard.setFrameShadow(qt.QFrame.Sunken)
            secondaryLayout = qt.QVBoxLayout(self.secondaryCard)
            
            secHeader = qt.QLabel("SECONDARY MOLECULAR PROFILE")
            secHeader.setFont(boldFont)
            secHeader.setAlignment(qt.Qt.AlignCenter)
            secondaryLayout.addWidget(secHeader)
            
            self.secondaryGrid = qt.QGridLayout()
            self.target_widgets = {}
            for idx, target in enumerate(["ER", "PR"]):
                w = {
                    'pbar': qt.QProgressBar(), 
                    'val_lbl': qt.QLabel("—"),
                    'title': qt.QLabel(f"{target}:")
                }
                w['title'].setFont(boldFont)
                w['val_lbl'].setFont(boldFont)
                
                w['pbar'].setRange(0, 100)
                w['pbar'].setValue(0)
                w['pbar'].setTextVisible(False)
                
                self.secondaryGrid.addWidget(w['title'], idx, 0)
                self.secondaryGrid.addWidget(w['val_lbl'], idx, 1)
                self.secondaryGrid.addWidget(w['pbar'], idx, 2)
                self.secondaryGrid.setColumnStretch(2, 1)
                self.target_widgets[target] = w
                
            secondaryLayout.addLayout(self.secondaryGrid)
            resultCardLayout.addWidget(self.secondaryCard)

            # --- Disclaimer ---
            disclaimerLabel = qt.QLabel(
                "⚠ AI-generated result — For clinical decision support only.\n"
                "Must be validated by a qualified medical professional."
            )
            disclaimerLabel.setWordWrap(True)
            disclaimerLabel.setAlignment(qt.Qt.AlignCenter)
            disclaimerFont = disclaimerLabel.font
            disclaimerFont.setItalic(True)
            disclaimerLabel.setFont(disclaimerFont)
            resultsLayout.addWidget(disclaimerLabel)

            # --- Export Button ---
            self.exportButton = qt.QPushButton("Export Clinical Report (.txt)")
            self.exportButton.enabled = False
            resultsLayout.addWidget(self.exportButton)

            # SECTION 3: CONNECTIONS & STATE
            # ============================================================
            self.layout.addStretch(1)

            self.uploadButton.connect('clicked(bool)', self.onUploadClicked)
            self.predictButton.connect('clicked(bool)', self.onPredictClicked)
            self.exportButton.connect('clicked(bool)', self.onExportClicked)

            self.generated_csv_path = ""
            self.last_prediction_data = {}
            self.is_batch_mode = False

            self.module_dir = os.path.dirname(os.path.abspath(__file__))
            self.backend_dir = os.path.join(self.module_dir, "backend")

            # Auto-check and install required ML libraries for portability
            self.checkDependencies()

        except Exception as e:
            error_msg = f"Failed to build UI. Error:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(error_msg)
            qt.QMessageBox.critical(slicer.util.mainWindow(), "UI Crash", error_msg)

    def checkDependencies(self):
        """Ensures that anyone who downloads this extension gets the required packages automatically installed."""
        needed_packages = []
        try:
            import pandas
        except ImportError:
            needed_packages.append("pandas")
        try:
            import xgboost
        except ImportError:
            needed_packages.append("xgboost")
        try:
            import sklearn
        except ImportError:
            needed_packages.append("scikit-learn")
        try:
            import joblib
        except ImportError:
            needed_packages.append("joblib")
        try:
            import imblearn
        except ImportError:
            needed_packages.append("imbalanced-learn")

        if needed_packages:
            msg = f"This module requires the following ML libraries to run locally:\n\n{', '.join(needed_packages)}\n\nWould you like 3D Slicer to install them now? (This only happens once)."
            if slicer.util.confirmOkCancelDisplay(msg, "Install Missing Dependencies?"):
                self.statusLabel.text = "Status: Installing dependencies... Please wait."
                slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
                slicer.app.processEvents()
                try:
                    slicer.util.pip_install(needed_packages)
                    self.statusLabel.text = "Status: Dependencies installed. Ready for CSV upload."
                except Exception as e:
                    self.statusLabel.text = "Status: Dependency installation failed."
                    slicer.util.errorDisplay(f"Failed to install packages: {e}")
                finally:
                    slicer.app.restoreOverrideCursor()
            else:
                self.statusLabel.text = "Status: Missing dependencies. Models cannot run."

    def validate_csv_columns(self, file_path):
        """
        Pre-flight check to validate whether the uploaded radiomics sheet contains a valid set of features.
        Prevents pipeline crashes and presents native Slicer error popups.
        """
        try:
            import csv
            with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                reader = csv.reader(f)
                headers = [h.strip() for h in next(reader)]
                
            # Define a set of critical radiomic biomarkers that MUST be in the file
            mandatory_markers = [
                "original_shape_Sphericity", 
                "original_firstorder_Entropy", 
                "original_glcm_Contrast"
            ]
            
            missing_markers = [m for m in mandatory_markers if m not in headers]
            
            if missing_markers:
                msg = (
                    "Invalid Radiomics File Format!\n\n"
                    "The uploaded CSV is missing critical 3D structural features:\n"
                    f"{', '.join(missing_markers)}\n\n"
                    "Please ensure the CSV was generated using your standardized 3D PyRadiomics extraction script."
                )
                slicer.util.errorDisplay(msg, windowTitle="Incompatible Dataset")
                return False
                
            return True
        except Exception as e:
            slicer.util.errorDisplay(f"File validation failed to read columns: {e}")
            return False

    # RESULT PANEL 
    # ================================================================
    def _resetResultPanel(self):
        self.her2Badge.setText("Awaiting Analysis")
        self.her2Badge.setStyleSheet("") # Clear coloring
        self.her2Pbar.setValue(0)
        self.her2Tier.setText("")
        self.her2Tier.setStyleSheet("")

        for target in ["ER", "PR"]:
            self.target_widgets[target]['pbar'].setValue(0)
            self.target_widgets[target]['val_lbl'].setText("—")

        self.resultTimestamp.setText("—")
        self.resultFilename.setText("—")
        self.exportButton.enabled = False
        self.resultsCollapsible.collapsed = True
        
        # Reset Batch table layout
        self.batchTable.setVisible(False)
        self.batchTable.setRowCount(0)
        self.resultCard.setVisible(True)
        self.last_prediction_data = {}

    def _populateResultPanel(self, prediction, confidence=None, differentials=None, explainability=None):
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d  %H:%M:%S")
        filename = os.path.basename(self.generated_csv_path) if self.generated_csv_path else "—"

        self.resultTimestamp.setText(timestamp_str)
        self.resultFilename.setText(filename)

        status_dict = {}
        export_target_data = {}

        if differentials:
            for diff_name, prob_str in differentials:
                try:
                    prob_val = float(prob_str.replace("%", "").strip())
                    marker = "+" if prob_val >= 50.0 else "-"

                    t_key = None
                    if "ER" in diff_name and "HER2" not in diff_name: t_key = "ER"
                    elif "PR" in diff_name: t_key = "PR"
                    elif "HER2" in diff_name: t_key = "HER2"

                    if t_key:
                        status_dict[t_key] = f"{t_key}{marker}"
                        
                        # --- Calculate true diagnostic confidence ---
                        true_confidence = prob_val if prob_val >= 50.0 else (100.0 - prob_val)
                        export_target_data[t_key] = {"status": marker, "conf": true_confidence}

                        # If HER2, update the main card
                        if t_key == "HER2":
                            self.her2Badge.setText(f"HER2 [{marker}]")
                            self.her2Pbar.setValue(int(prob_val))
                            
                            # Simple color text styling that works cleanly on Slicer Light/Dark modes
                            if true_confidence >= 80:
                                self.her2Tier.setText(f"CONFIDENCE: {true_confidence:.1f}% (HIGH)")
                                self.her2Tier.setStyleSheet("color: #228B22;") # Forest Green
                            elif true_confidence >= 60:
                                self.her2Tier.setText(f"CONFIDENCE: {true_confidence:.1f}% (MODERATE)")
                                self.her2Tier.setStyleSheet("color: #D2691E;") # Chocolate/Orange
                            else:
                                self.her2Tier.setText(f"CONFIDENCE: {true_confidence:.1f}% (LOW - REVIEW)")
                                self.her2Tier.setStyleSheet("color: #B22222;") # Firebrick Red
                        
                        # If ER or PR, update the secondary grid
                        elif t_key in ["ER", "PR"]:
                            w = self.target_widgets[t_key]
                            w['pbar'].setValue(int(prob_val))
                            w['val_lbl'].setText(f"[{marker}]  |  {true_confidence:.1f}%")

                except ValueError:
                    pass

        # Assemble Master String for export
        if 'ER' in status_dict and 'PR' in status_dict and 'HER2' in status_dict:
            receptor_status = f"{status_dict['ER']}, {status_dict['PR']}, {status_dict['HER2']}"
        else:
            receptor_status = prediction if prediction else "Unknown Subtype"

        self.last_prediction_data = {
            "timestamp": timestamp_str,
            "input_file": self.generated_csv_path,
            "prediction": receptor_status,
            "is_batch": False,
            "targets": export_target_data,
            "explainability": explainability if explainability else []
        }

        self.exportButton.enabled = True
        self.resultsCollapsible.collapsed = False

    def _parseBackendOutput(self, stdout):
        lines = stdout.strip().split("\n")
        prediction = None
        confidence = None
        differentials = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("PREDICTION:"):
                prediction = stripped.replace("PREDICTION:", "").strip()
            elif stripped.startswith("CONFIDENCE:"):
                try: confidence = float(stripped.replace("CONFIDENCE:", "").strip())
                except: confidence = None
            elif stripped.startswith("DIFF:"):
                parts = stripped.replace("DIFF:", "").strip().split("|")
                if len(parts) == 2:
                    differentials.append((parts[0].strip(), parts[1].strip() + "%"))

        if prediction is None:
            for line in reversed(lines):
                if line.strip():
                    prediction = line.strip()
                    break

        if prediction is None: prediction = "Unknown"
        return prediction, confidence, differentials

    
    # PIPELINE CALLBACKS 
    # ================================================================
    def onUploadClicked(self):
        fileDialog = qt.QFileDialog()
        fileDialog.setFileMode(qt.QFileDialog.ExistingFile)
        fileDialog.setNameFilter("Patient Radiomics Sheet (*.csv)")

        if fileDialog.exec_():
            selected_file = fileDialog.selectedFiles()[0]
            
            # --- RUN PRE-FLIGHT VALIDATION GUARD ---
            if not self.validate_csv_columns(selected_file):
                self._resetResultPanel()
                self.filePathBox.text = ""
                self.predictButton.enabled = False
                self.statusLabel.text = "Status: Upload aborted. Invalid file format."
                return

            self.filePathBox.text = selected_file
            self._resetResultPanel()

            self.generated_csv_path = selected_file
            
            # Check row count safely (handles BOM-encoded and lock-guarded CSVs seamlessly)
            try:
                with open(selected_file, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    lines = [line for line in f if line.strip()]
                    row_count = max(0, len(lines) - 1)
            except Exception as e:
                row_count = 1
                print(f"Could not read row count safely: {e}")

            self.predictButton.enabled = True 
            if row_count > 1:
                self.statusLabel.text = f"Status: Batch CSV verified! {row_count} patients detected."
                self.is_batch_mode = True
            else:
                self.statusLabel.text = "Status: Radiomics CSV uploaded! Ready for prediction."
                self.is_batch_mode = False

    def onPredictClicked(self):
        self.statusLabel.text = "Status: Ensemble Models predicting subtype..."
        slicer.app.processEvents()

        # Portable Slicer Native Environment execution across multiple platforms
        python_exe = shutil.which("PythonSlicer")
        if not python_exe:
            ext = ".exe" if os.name == 'nt' else ""
            root_path = os.path.join(slicer.app.slicerHome, f"PythonSlicer{ext}")
            bin_path = os.path.join(slicer.app.slicerHome, "bin", f"PythonSlicer{ext}")
            
            if os.path.exists(root_path):
                python_exe = root_path
            elif os.path.exists(bin_path):
                python_exe = bin_path
            else:
                python_exe = f"PythonSlicer{ext}"

        ensemble_script = os.path.join(self.backend_dir, "predict_from_csv.py")

        try:
            process = subprocess.Popen(
                [python_exe, ensemble_script, self.generated_csv_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            stdout, stderr = process.communicate()

            print("--- PIPELINE STDOUT ---\n", stdout)
            if stderr:
                print("--- PIPELINE STDERR ---\n", stderr)

            if process.returncode != 0:
                self.statusLabel.text = "Status: Prediction Failed!"
                print("--- ML SCRIPT ERROR ---\n", stderr)
                return

            # PARSE STDOUT FOR BATCH RUNS, EXPLAINABILITY, AND CLINICAL DIFFERENTIALS
            batch_rows = []
            explainability_lines = []
            for line in stdout.strip().split("\n"):
                stripped = line.strip()
                if stripped.startswith("BATCH_ROW:"):
                    parts = stripped.replace("BATCH_ROW:", "").strip().split("|")
                    if len(parts) == 4:
                        batch_rows.append([p.strip() for p in parts])
                elif stripped.startswith("EXPLAINABILITY_TAG:"):
                    exp_val = stripped.replace("EXPLAINABILITY_TAG:", "").strip()
                    explainability_lines.append(exp_val)

            # RENDER MODE ROUTING
            if len(batch_rows) > 1 and self.is_batch_mode:
                self.resultCard.setVisible(False)
                self.batchTable.setVisible(True)
                self.batchTable.setRowCount(len(batch_rows))
                
                for row_idx, data in enumerate(batch_rows):
                    patient_id, her2_mapped, er_mapped, pr_mapped = data
                    
                    # Convert mapped probability back to native labels and True Confidence levels
                    # HER2
                    h_val = float(her2_mapped)
                    h_marker = "+" if h_val >= 50.0 else "-"
                    h_conf = h_val if h_val >= 50.0 else (100.0 - h_val)
                    
                    # ER
                    e_val = float(er_mapped)
                    e_marker = "+" if e_val >= 50.0 else "-"
                    e_conf = e_val if e_val >= 50.0 else (100.0 - e_val)
                    
                    # PR
                    p_val = float(pr_mapped)
                    p_marker = "+" if p_val >= 50.0 else "-"
                    p_conf = p_val if p_val >= 50.0 else (100.0 - p_val)
                    
                    self.batchTable.setItem(row_idx, 0, qt.QTableWidgetItem(patient_id))
                    self.batchTable.setItem(row_idx, 1, qt.QTableWidgetItem(f"[{h_marker}] ({h_conf:.1f}%)"))
                    self.batchTable.setItem(row_idx, 2, qt.QTableWidgetItem(f"[{e_marker}] ({e_conf:.1f}%)"))
                    self.batchTable.setItem(row_idx, 3, qt.QTableWidgetItem(f"[{p_marker}] ({p_conf:.1f}%)"))
                
                self.statusLabel.text = "Status: Batch Prediction Complete"
                self.resultsCollapsible.collapsed = False
                
                self.last_prediction_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "input_file": self.generated_csv_path,
                    "is_batch": True,
                    "batch_data": batch_rows
                }
                self.exportButton.enabled = True
            else:
                # Fallback warning if batch data structure was predicted but not parsed as multiple rows
                if self.is_batch_mode and len(batch_rows) <= 1:
                    print("WARNING: Batch mode active but backend did not return multiple results. Check if backend/predict_from_csv.py is updated.")
                
                # Standard single patient visual dial modes
                prediction, confidence, differentials = self._parseBackendOutput(stdout)
                self.statusLabel.text = "Status: Prediction Complete"
                self._populateResultPanel(prediction, confidence, differentials, explainability_lines)

        except Exception as e:
            self.statusLabel.text = f"Status: Error - {str(e)}"

    def onExportClicked(self):
        if not self.last_prediction_data:
            return

        save_dialog = qt.QFileDialog()
        save_path = save_dialog.getSaveFileName(
            slicer.util.mainWindow(),
            "Save Diagnostic Report",
            os.path.join(os.path.dirname(self.generated_csv_path), "Clinical_Biomarker_Report.txt"),
            "Text Files (*.txt)"
        )

        if not save_path: return

        data = self.last_prediction_data

        # --- WRITE BATCH CLINICAL REPORT ---
        if "is_batch" in data and data["is_batch"]:
            report_lines = [
                "=" * 72,
                "           BATCH CLINICAL MOLECULAR SUBTYPING REPORT",
                "              Radiogenomics Standalone Subtyper",
                "=" * 72,
                "",
                f"  Timestamp:       {data['timestamp']}",
                f"  Input Dataset:   {os.path.basename(data['input_file'])}",
                f"  Total Patients:  {len(data['batch_data'])}",
                "",
                "-" * 72,
                f"  {'Patient ID':<25} | {'HER2 Status':<12} | {'ER Status':<12} | {'PR Status':<12}",
                "-" * 72,
            ]
            for row in data["batch_data"]:
                patient_id, her2_mapped, er_mapped, pr_mapped = row
                
                h_val = float(her2_mapped)
                h_marker = "+" if h_val >= 50.0 else "-"
                h_conf = h_val if h_val >= 50.0 else (100.0 - h_val)
                
                e_val = float(er_mapped)
                e_marker = "+" if e_val >= 50.0 else "-"
                e_conf = e_val if e_val >= 50.0 else (100.0 - e_val)
                
                p_val = float(pr_mapped)
                p_marker = "+" if p_val >= 50.0 else "-"
                p_conf = p_val if p_val >= 50.0 else (100.0 - p_val)
                
                her2_str = f"[{h_marker}] ({h_conf:.1f}%)"
                er_str = f"[{e_marker}] ({e_conf:.1f}%)"
                pr_str = f"[{p_marker}] ({p_conf:.1f}%)"
                
                report_lines.append(f"  {patient_id:<25} | {her2_str:<12} | {er_str:<12} | {pr_str:<12}")
            
            report_lines.append("")
        else:
            # --- WRITE SINGLE PATIENT CLINICAL REPORT ---
            report_lines = [
                "=" * 56,
                "           ASSISTED DIAGNOSTIC REPORT",
                "        Radiogenomics Auto-Subtyper",
                "=" * 56,
                "",
                f"  Timestamp:       {data['timestamp']}",
                f"  Input Data:      {os.path.basename(data['input_file'])}",
                f"  Full Path:       {data['input_file']}",
                "",
                "-" * 56,
                "  PRIMARY BIOMARKER",
                "-" * 56,
                ""
            ]

            if "targets" in data and data["targets"] and "HER2" in data["targets"]:
                stat = data["targets"]["HER2"]["status"]
                conf = data["targets"]["HER2"]["conf"]
                report_lines.extend([
                    f"  HER2 STATUS: [{stat}]",
                    f"  Confidence:  {conf:.1f}%",
                    ""
                ])

            report_lines.extend([
                "-" * 56,
                "  SECONDARY MOLECULAR PROFILE",
                "-" * 56,
                ""
            ])

            if "targets" in data and data["targets"]:
                for tgt in ["ER", "PR"]:
                    if tgt in data["targets"]:
                        stat = data["targets"][tgt]["status"]
                        conf = data["targets"][tgt]["conf"]
                        report_lines.append(f"  {tgt:<5} Status: [{stat}]   |   Confidence: {conf:.1f}%")
                report_lines.append("")

            # Decision drivers append
            if "explainability" in data and data["explainability"]:
                report_lines.extend([
                    "-" * 56,
                    "  DECISION DRIVERS & MODEL EXPLAINABILITY",
                    "-" * 56,
                    ""
                ])
                for exp in data["explainability"]:
                    report_lines.append(f"  {exp}")
                report_lines.append("")

        report_lines.extend([
            "-" * 56,
            "  DISCLAIMER",
            "-" * 56,
            "",
            "  This is an AI-generated result intended for clinical",
            "  decision support ONLY. It must be validated by a",
            "  qualified medical professional before any clinical",
            "  action is taken.",
            "",
            "=" * 56,
            "  Reviewing Physician: _________________________",
            "",
            "  Signature:           _________________________",
            "",
            "  Date:                _________________________",
            "=" * 56,
        ])

        try:
            with open(save_path, "w") as f:
                f.write("\n".join(report_lines))
            self.statusLabel.text = f"Status: Report exported to {os.path.basename(save_path)}"
        except Exception as e:
            self.statusLabel.text = f"Status: Export failed - {str(e)}"

class RadiogenomicsUITest(ScriptedLoadableModuleTest):
    def setUp(self): slicer.mrmlScene.Clear(0)
    def runTest(self): self.setUp(); self.delayDisplay('Test passed successfully!')
