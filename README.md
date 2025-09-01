# Blink Project  

This repository contains research scripts and datasets for analyzing **eye-blink behavior** in relation to **mental fatigue, attentional engagement, and stress**. It was developed as part of a larger **VR-based fatigue assessment project** combining **eye-tracking, blink metrics, and statistical modeling**.  

The aim is to provide robust tools to quantify how blinking patterns change across fatigue states and build towards **objective fatigue scoring algorithms**.  

---

## 📂 Repository Structure  

blink_project/
├── Group_By_Analysis_Plots/ # Visualization outputs for grouped analysis
├── Stage_Category_KMeans/ # KMeans clustering of fatigue stages
├── plots_stress_score/ # Stress score visualizations
├── summary/ # Summary CSVs and processed data
│
├── README.md # Project documentation
├── blink_analysis_batcher.py # Batch processing of blink datasets
├── blink_duration_analysis.py # Blink duration analysis (fatigue & immersion)
├── boxcox_analysis.py # Box-Cox transformations for normalization
├── burst_counts_analysis.py # Blink burst detection & analysis
├── grid_coutns_analysis.py # Grid coverage per blink metrics
├── group_by_analysis.py # Grouped statistical comparisons
├── group_by_analysis_result.csv # Grouped results (precomputed)
├── mismatch_analysis.py # Blink mismatch metrics
├── parametric_analysis.py # Parametric tests (t-tests, ANOVA, etc.)
├── rawdata.py # Raw data processing & cleaning pipeline
├── summary.csv # Main summary dataset (blink features)
└── total_blinks_analysis.py # Total blink count analysis

---

## 🔬 Key Features  

- **Blink Duration Analysis**  
  Study how blink duration reflects both **fatigue (longer blinks under fatigue)** and **immersion (longer blinks in non-fatigued states)**.  

- **Blink Frequency Analysis**  
  Show how **higher blink frequency** is linked to mental fatigue when sustained attention declines.  

- **Blink Bursts (Flurries)**  
  Detect clusters of rapid blinks as indicators of cognitive overload.  

- **Mismatch Analysis**  
  Identify irregular or inconsistent blink timing patterns.  

- **Grid Counts per Blink**  
  Measure how gaze coverage interacts with blinking during tasks.  

- **Statistical Rigor**  
  Includes **parametric testing**, **Box-Cox transformations**, and **Pearson/Spearman correlations** for robust analysis.  

- **Machine Learning**  
  Implements **KMeans clustering** to group participants into fatigue stages.  

- **Visualization**  
  Provides group-by plots, stress score distributions, and boxplots for interpretation.  

---

## 📊 Data & Workflow  

1. **Input Data**: Participant-level CSV files with blink timings, durations, and gaze context.  
2. **Preprocessing**: Z-score normalization, log transformations, and outlier handling.  
3. **Analysis**:  
   - Correlation testing (Pearson, Spearman)  
   - Group comparisons (high vs low fatigue)  
   - Parametric testing (t-test, ANOVA)  
   - KMeans clustering of fatigue stages  
4. **Outputs**:  
   - Statistical results (`summary/`, `group_by_analysis_result.csv`)  
   - Plots (`Group_By_Analysis_Plots/`, `plots_stress_score/`)  

---

## 📈 Research Context  

This project is part of a **VR-based eye-tracking system** designed to measure **brain fatigue** through blink metrics.  

- Conducted with **1,000+ participants** in VR arithmetic and attention tasks.  
- Metrics analyzed: blink frequency, duration, bursts, mismatch, and inter-blink intervals.  
- Findings:  
  - **High fatigue** → more frequent blinks, more bursts, longer inter-blink intervals.  
  - **Low fatigue** → blink duration may reflect **immersion** instead of fatigue.  
  - Supports building **adaptive algorithms** that distinguish between fatigue vs engagement.  

---

## 🛠 Tech Stack  

- **Python**  
  - pandas  
  - NumPy  
  - statsmodels  
  - scikit-learn  
  - matplotlib  
  - seaborn  
  - dcor  

---

## 🚀 Usage  

Clone the repository:  
```bash
git clone https://github.com/KimShota/blink_project.git
cd blink_project

Run an analysis (for example: blink duration):

python blink_duration_analysis.py

Outputs will be saved in the appropriate folders (summary/, Group_By_Analysis_Plots/, etc.).
