# 🧬 miimansa-nlp-project
### Medical Named Entity Recognition (NER) and Entity Linking  

This project performs **Named Entity Recognition (NER)** on medical forum posts to identify mentions of **Drugs, Diseases, Symptoms, and Adverse Drug Reactions (ADRs)**.  
It further attempts to link ADR mentions to **SNOMED-CT medical codes** using **fuzzy matching** and **sentence embeddings**.  

---

## 📂 Dataset  

We use the **CADEC (Corpus of Adverse Drug Event) dataset**.  

Expected directory structure after downloading and extracting `CADEC.v2.zip`:  

cadec/
│── text/ # Raw forum posts
│── original/ # Human-annotated entities (ADR, Drug, Disease, Symptom)
│── meddra/ # ADR-specific annotations with MedDRA codes
│── sct/ # ADR annotations linked to SNOMED-CT


👉 Download CADEC.v2 from the [official repository](https://data.csiro.au/dap/landingpage?pid=csiro:20712).  
Unzip it and place inside `cadec/` directory.  

---

## 🚀 Project Workflow  

The project is divided into **six tasks**, each implemented as a Python script and/or Jupyter notebook.  

---

### **Task 1 – Entity Enumeration**  
**File:** `Task1.ipynb` / `step1_entity_enumeration.py`  
- Parses `.ann` files from `cadec/original/`.  
- Counts distinct entities for each label type: **ADR, Drug, Disease, Symptom**.  
- Output: `outputs/task1_entities.csv`  

---

### **Task 2 – NER using Pre-trained Biomedical Model**  
**File:** `Task2.ipynb` / `step2_llm_sequence_labelling.py`  
- Model used: **`d4data/biomedical-ner-all`** (HuggingFace).  
- Process:  
  1. Load text from `cadec/text/`.  
  2. Apply NER pipeline → BIO tags.  
  3. Convert BIO format into CADEC `.ann` style.  
- Output: `outputs/task2_predictions.json`  

---

### **Task 3 – Evaluation (Strict Matching)**  
**File:** `Task3.ipynb` / `step3_evaluate_predictions.py`  
- Compares Task 2 predictions with ground truth in `cadec/original/`.  
- Metrics: **Precision, Recall, F1-score**.  
- Matching type: **Strict** (exact span + label match).  
- Output: `outputs/task3_metrics.csv`  

---

### **Task 4 – ADR-focused Evaluation with MedDRA**  
**File:** `Task4.ipynb` / `step4.py`  
- Focus only on **ADR entities**.  
- Ground truth: `cadec/meddra/`.  
- Metrics: Precision, Recall, F1-score.  
- Output: `outputs/task4_metrics.csv`  

---

### **Task 5 – Relaxed Evaluation (Random 50 Posts)**  
**File:** `Task5_Random50.ipynb` / `step5_relaxed_eval.py`  
- Samples **50 random posts** from `cadec/text/`.  
- Predictions from Task 2 are re-evaluated with **relaxed overlap matching**.  
- Reports **Macro Precision, Recall, F1-score**.  
- Outputs:  
  - `outputs/task5/task5_metrics.csv`  
  - `outputs/task5/task5_metrics.json`  

---

### **Task 6 – Linking ADRs to SNOMED-CT**  
**File:** `Task6_Merge_SCT.ipynb` / `step6.py`  
- Combines data from `cadec/original/` and `cadec/sct/`.  
- For each ADR entity: retrieves SNOMED-CT **code + description**.  
- Matching methods:  
  1. **Fuzzy String Matching** (`fuzzywuzzy`).  
  2. **Sentence Embeddings** (`sentence-transformers/all-MiniLM-L6-v2`).  
- Output: `outputs/task6/task6_matches.csv`  

---

## 📊 Results Snapshot  

- **Task 3 (Strict Evaluation):**  
  Precision ≈ 0.46 | Recall ≈ 0.42 | F1 ≈ 0.44  

- **Task 5 (Random 50, Relaxed Evaluation):**  
  Macro Precision ≈ 0.48 | Macro Recall ≈ 0.43 | Macro F1 ≈ 0.42  

- **Task 6 (Entity Linking):**  
  - Fuzzy match = stronger for exact matches.  
  - Embedding method = better semantic linking (e.g., *“muscle pain” → “Myalgia”*).  

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
