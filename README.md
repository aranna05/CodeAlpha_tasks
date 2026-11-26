<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:14213D,50:2A9D8F,100:E9C46A&height=220&section=header&text=CodeAlpha%20Internship&fontSize=52&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Data%20Science%20%7C%20Machine%20Learning%20%7C%20Data%20Storytelling&descAlignY=60&descSize=18" alt="CodeAlpha Internship banner" width="100%"/>

<a href="https://github.com/aranna05/CodeAlpha_tasks">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1200&color=2A9D8F&center=true&vCenter=true&width=700&lines=Two+end-to-end+data+science+projects;From+raw+CSV+to+insight+and+prediction;Clean+%E2%86%92+Explore+%E2%86%92+Model+%E2%86%92+Explain" alt="Typing animation" />
</a>

<br/>

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-4C8CBF?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

![License](https://img.shields.io/github/license/aranna05/Internship_CodeAlpha?style=flat-square&color=2A9D8F)
![Last commit](https://img.shields.io/github/last-commit/aranna05/Internship_CodeAlpha?style=flat-square&color=E76F51)
![Repo size](https://img.shields.io/github/repo-size/aranna05/Internship_CodeAlpha?style=flat-square&color=E9C46A)

</div>

---

## 👋 About This Repository

This repository showcases the work I completed during my **Data Science internship at CodeAlpha**. Each project is a complete, reproducible workflow: a polished Jupyter notebook, a standalone Python script, the source data, and ready-made charts and outputs.

| # | Project | Type | Headline result |
|:-:|---------|------|-----------------|
| 1 | [📉 Unemployment Analysis in India](#-project-1--unemployment-analysis-in-india) | Exploratory data analysis | Unemployment jumped from about **7.6%** to **24.2%** in April 2020 |
| 2 | [📈 Sales Prediction (Sales Signal)](#-project-2--sales-prediction-sales-signal) | Regression / ML | **R² = 0.988** on held-out data with a degree-2 polynomial model |

---

## 🗂️ Repository Structure

```text
Internship_CodeAlpha/
│
├── 📉 Unemployment_Analysis_India_CodeAlpha/
│   ├── Unemployment_Analysis_India.ipynb     # guided, presentation-ready notebook
│   ├── unemployment_analysis.py              # script version of the analysis
│   ├── Unemployment in India.csv             # area-level source data
│   ├── Unemployment_Rate_upto_11_2020.csv    # state-level source data
│   └── outputs/                              # 6 charts + cleaned CSVs
│
├── 📈 Sales_Prediction_CodeAlpha/
│   ├── Sales_Signal_Sales_Prediction.ipynb   # EDA, modelling, diagnostics
│   ├── sales_prediction.py                   # standalone training pipeline
│   ├── data/Advertising.csv                  # TV, Radio, Newspaper → Sales
│   └── outputs/                              # model, metrics, charts
│
├── LICENSE
└── README.md
```

---

## 📉 Project 1 · Unemployment Analysis in India

> **How hard did COVID-19 hit India's job market, and who felt it most?**

A reproducible analysis of two datasets covering Indian unemployment before and during the pandemic.

### 🎯 What it does

- 🧹 **Cleans** whitespace, dates, numeric fields, blanks and duplicates (740 area-level and 267 state-level valid rows)
- 🧮 **Rebuilds a weighted monthly rate** from the implied unemployed count, `unemployed = employed × rate / (100 − rate)`, which is more defensible than a plain average of regional rates
- ⏱️ **Compares** the pre-COVID period (to Feb 2020) with March 2020 onward
- 🏙️ **Explores** rural vs urban gaps, state-level hotspots and labour-force participation

### 📊 Demo

<div align="center">

<img src="Unemployment_Analysis_India_CodeAlpha/outputs/01_covid_shock_area_level.png" alt="COVID shock on the unemployment rate" width="85%"/>

**The shock:** the weighted rate hovered around 7–8%, then spiked to a peak of **24.2% in April 2020**.

</div>

<br/>

<table>
<tr>
<td width="50%" align="center">
<img src="Unemployment_Analysis_India_CodeAlpha/outputs/04_april_2020_state_hotspots.png" alt="April 2020 state hotspots"/>
<br/><b>🔥 State hotspots</b><br/>
In April 2020 Puducherry reached about 76%, with Tamil Nadu, Jharkhand, Bihar and Haryana all above 40%.
</td>
<td width="50%" align="center">
<img src="Unemployment_Analysis_India_CodeAlpha/outputs/03_rural_urban_comparison.png" alt="Rural vs urban unemployment"/>
<br/><b>🏙️ Urban vs rural</b><br/>
Urban observations average <b>13.2%</b> against <b>10.3%</b> for rural ones.
</td>
</tr>
</table>

<div align="center">

<img src="Unemployment_Analysis_India_CodeAlpha/outputs/06_labour_participation.png" alt="Labour participation rate" width="85%"/>

**Participation fell too:** the labour-force participation rate dropped from about 44% to roughly 35% at its low point, then began to recover.

</div>

### 🔑 Key takeaways

- 📈 Unemployment roughly tripled in a single month once lockdowns began
- 🗺️ The impact was highly uneven across states
- 👥 People also left the labour force, not just lost jobs

### ▶️ Run it

```bash
cd Unemployment_Analysis_India_CodeAlpha
python -m pip install pandas numpy matplotlib seaborn jupyter
python unemployment_analysis.py            # regenerates charts and cleaned CSVs
# or explore interactively
jupyter notebook Unemployment_Analysis_India.ipynb
```

> ⚠️ **Note:** The reconstructed national series is an estimate, not an official statistic. The March 2020 split is descriptive and does not prove a causal COVID effect.

---

## 📈 Project 2 · Sales Prediction (Sales Signal)

> **Can advertising spend on TV, Radio and Newspaper predict sales?**

An end-to-end regression project on the classic `Advertising.csv` dataset: 200 rows, three ad channels, one sales target, and no missing values.

### 🎯 What it does

- 🔍 Validates the data and explores relationships between spend and sales
- 🏁 **Compares 5 models** under the same 80/20 split and shuffled 5-fold cross-validation
- 🧠 Explains the winner with **permutation importance** on unseen data
- 💾 Saves the trained pipeline (`.joblib`) so it can be reused for predictions

### 🏆 Model leaderboard

| Rank | Model | CV MAE ↓ | Test MAE ↓ | Test RMSE ↓ | Test R² ↑ |
|:----:|-------|:--------:|:----------:|:-----------:|:---------:|
| 🥇 | **Polynomial Regression (deg 2)** | **0.441** | **0.485** | **0.621** | **0.988** |
| 🥈 | Random Forest | 0.665 | 0.661 | 0.791 | 0.980 |
| 🥉 | Gradient Boosting | 0.707 | 0.695 | 0.854 | 0.977 |
| 4 | Linear Regression | 1.237 | 1.461 | 1.782 | 0.899 |
| 5 | Ridge Regression | 1.237 | 1.464 | 1.787 | 0.899 |

### 📊 Demo

<table>
<tr>
<td width="50%" align="center">
<img src="Sales_Prediction_CodeAlpha/outputs/actual_vs_predicted.png" alt="Actual vs predicted sales"/>
<br/><b>🎯 Actual vs predicted</b><br/>
Points hug the diagonal, so predictions track real sales closely.
</td>
<td width="50%" align="center">
<img src="Sales_Prediction_CodeAlpha/outputs/feature_importance.png" alt="Feature importance"/>
<br/><b>📡 Which channel matters?</b><br/>
TV carries the most predictive signal, Radio is second, and Newspaper adds almost nothing.
</td>
</tr>
</table>

<div align="center">

<img src="Sales_Prediction_CodeAlpha/outputs/residual_distribution.png" alt="Prediction error distribution" width="60%"/>

**Errors are centred on zero**, with most predictions within about ±1 sales unit.

</div>

### 🔑 Key takeaways

- 🚀 Adding polynomial terms lifts R² from **0.899** (linear) to **0.988**, which suggests the channels interact
- 📺 TV and 📻 Radio drive the predictions; 📰 Newspaper is practically noise
- ⚠️ Predictive importance is **not** causal ROI. Test with controlled experiments before moving real budgets

### ▶️ Run it

```bash
cd Sales_Prediction_CodeAlpha
pip install pandas numpy matplotlib scikit-learn joblib jupyter
python sales_prediction.py                 # trains, evaluates, saves outputs/
# or explore interactively
jupyter notebook Sales_Signal_Sales_Prediction.ipynb
```

---

## 🛠️ Tech Stack

| Area | Tools |
|------|-------|
| **Language** | Python |
| **Data wrangling** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn |
| **Machine learning** | scikit-learn (pipelines, cross-validation, permutation importance), joblib |
| **Workflow** | Jupyter Notebook, Git & GitHub |

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/aranna05/Internship_CodeAlpha.git
cd Internship_CodeAlpha

# 2. Install everything once
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter

# 3. Run either project
python Unemployment_Analysis_India_CodeAlpha/unemployment_analysis.py
python Sales_Prediction_CodeAlpha/sales_prediction.py
```

---

## 🙌 What I Learned

- Cleaning messy real-world data and checking its quality before analysing it
- Choosing honest aggregation methods and stating a model's limits clearly
- Comparing several models fairly with a fixed split and cross-validation
- Turning numbers into charts and takeaways that others can act on

---

## 📜 License

Released under the terms of the [LICENSE](LICENSE) file in this repository.
Thank You

<div align="center">

### ⭐ If you found this useful, consider giving the repo a star!

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:E9C46A,50:2A9D8F,100:14213D&height=100&section=footer" alt="footer" width="100%"/>

</div>
