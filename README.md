# Blink Project  

A research-focused repository for analyzing **eye-blink patterns** and their relationship to **mental fatigue, attention, and stress**. This project uses Python-based statistical analysis, machine learning, and visualization methods to process large-scale blink datasets collected from VR-based experiments.  

The analyses here contribute to developing **objective fatigue assessment algorithms**, validated with **gaze/blink datasets from 1,000+ participants** in VR cognitive tasks.  

---

## 📂 Repository Structure  

blink_project/
├── Group_By_Analysis_Plots/ # Visualization outputs for grouped analysis
├── Stage_Category_KMeans/ # KMeans clustering of fatigue stages
├── plots_stress_score/ # Stress score visualizations
├── summary/ # Summary CSVs and processed data
│
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
├── total_blinks_analysis.py # Total blink count analysis
│
└── README.md # Project documentation


---

## 🔬 Analysis Features  

This repository implements **multiple blink-related metrics**, including:  

- **Blink Duration Analysis** – correlation with fatigue and attentional engagement.  
- **Blink Frequency Analysis** – increased blinking under high fatigue states.  
- **Burst Counts Analysis** – detection of blink flurries during fatigue.  
- **Inter-Blink Interval Analysis** – changes in cognitive load and attention.  
- **Mismatch Analysis** – irregular blink-timing events across tasks.  
- **Grid Counts per Blink** – how gaze and blinks interact with visual exploration.  
- **Parametric & Box-Cox Analysis** – ensuring statistical robustness and normality.  
- **Group Comparisons (High vs Low Fatigue)** – statistical testing across participant groups.  
- **Stage-Based KMeans Clustering** – unsupervised grouping of fatigue stages.  
- **Stress Score Plots** – visualization of blink-derived fatigue/stress scores.  

---

## 📊 Data & Workflow  

1. **Input Data**: CSV datasets (per participant) with blink timing, duration, and gaze context.  
2. **Preprocessing**: Normalization, log transformations, and outlier reduction.  
3. **Statistical Analysis**: Pearson, Spearman, Welch’s t-test, ANOVA, Box-Cox.  
4. **Machine Learning**: KMeans clustering for fatigue stage categorization.  
5. **Visualization**: Group-by plots, stress score distributions, boxplots, and heatmaps.  
6. **Output**: Summaries in `summary/`, plots in `plots_stress_score/` and `Group_By_Analysis_Plots/`.  

---

## 📈 Research Context  

This repository is part of a larger **VR-based fatigue research project**, which uses:  

- **VR experiments** (Flash Mental Arithmetic, focus tasks, distraction-based trials).  
- **Eye-tracking** (blink frequency, duration, bursts, mismatch).  
- **Large-scale datasets** (1,000+ participants).  
- **Goal**: Build a **practical brain fatigue scoring model** with predictive accuracy.  

Findings so far:  
- Higher blink frequency, bursts, and long inter-blink intervals correlate with **higher fatigue**.  
- Longer blink duration can reflect **fatigue in high-fatigue states**, but also **immersion/engagement in low-fatigue states**.  
- These nuanced patterns motivate building **adaptive fatigue algorithms**.  

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

Run an analysis (example: blink duration):

python blink_duration_analysis.py

Results will be saved into the respective folders (summary/, Group_By_Analysis_Plots/, etc.).
