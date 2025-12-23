# 🧠 SepsisPredict — Early Sepsis Detection using MIMIC-IV
**Clinical Machine Learning for Early Risk Stratification**

<p align="center">
  <img src="https://img.shields.io/badge/Healthcare-AI-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Dataset-MIMIC--IV-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Model-Random%20Forest-orange?style=for-the-badge"/>
</p>

---

## 📌 Overview
**SepsisPredict** is an end-to-end machine learning pipeline for **early sepsis detection** using structured clinical data from the **MIMIC-IV** database.

The project focuses on **feature engineering, exploratory analysis, and interpretable modeling** to establish a reproducible baseline for clinical deterioration prediction in ICU-style datasets.

---

## 🎯 Objective
- Detect early signs of sepsis using patient demographics, vitals, and laboratory data  
- Build a **reproducible ML baseline** for clinical risk prediction  
- Provide visual and statistical insights into key sepsis indicators  

---

## 📁 Dataset Sources
A curated subset of the **MIMIC-IV (v2.2)** dataset is used:

- `ADMISSIONS.csv`
- `PATIENTS.csv`
- `D_LABITEMS.csv`
- `LABEVENTS.csv`
- `structured_medical_records.csv`

All files are stored under the `data/` directory and processed to construct patient-level features.

> **Note:** Access to MIMIC-IV requires PhysioNet credentialing.

---

## 🔬 Key Features
- 🔍 Exploratory Data Analysis (EDA) using Seaborn & Matplotlib  
- 🧼 Preprocessing of vitals, lab measurements, and demographics  
- 🌲 **Random Forest Classifier** for sepsis prediction  
- 📊 Feature importance analysis for clinical interpretability  
- 💾 Automatic saving of results and visualizations  

---

## 🛠 Tech Stack
<p align="left">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="38"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="38"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" width="38"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/scikitlearn/scikitlearn-original.svg" width="38"/>
</p>

**Visualization**
- Matplotlib
- Seaborn

---

## 📂 Project Structure

SepsisPredict_MIMIC_Demo/
│
├── data/
│   ├── ADMISSIONS.csv
│   ├── PATIENTS.csv
│   ├── LABEVENTS.csv
│   ├── D_LABITEMS.csv
│   ├── structured_medical_records.csv
│   ├── output_visuals.png
│   └── results.txt
│
├── main.py              # Data processing, modeling & evaluation
├── requirements.txt
└── README.md

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

git clone https://github.com/Fastian-afk/SepsisPredict_MIMIC_Demo.git
cd SepsisPredict_MIMIC_Demo

### 2️⃣ Set Up Python Environment

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

### 3️⃣ Run the Pipeline

python main.py

---

## ✅ Outputs

* 📈 Visualizations saved to:
  `data/output_visuals.png`
* 📄 Model evaluation report saved to:
  `data/results.txt`

---

## 📊 Why This Project Matters

* Sepsis is **life-threatening and time-critical**
* Early detection significantly reduces mortality
* Demonstrates **real-world clinical ML workflows**
* Aligns with ICU monitoring and decision-support systems

---

## 🙌 Acknowledgments

* **MIMIC-IV Database** — PhysioNet
* MIT Laboratory for Computational Physiology (MIT-LCP)
* Sepsis-3 Clinical Guidelines

---

## 📜 License

This project is released under the **MIT License**.
Free to use for research and educational purposes with attribution.
