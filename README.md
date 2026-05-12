# 📊 MATH 3410: Probability & Statistics I — Fall 2025
**Weber State University** · Interactive Streamlit Apps

> Streamlit apps for the Fall 2025 semester covering distributions, statistical inference, and regression — with enhanced gradient descent functionality.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Apps-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📁 Project Structure

```
math3410_fall25/
├── README.md                  ← You are here
├── requirements.txt           ← Python dependencies
│
├── apps/
│   ├── distributions/         ← Distributions & density functions
│   │   ├── CLT.py
│   │   └── PoissonMLE.py
│   ├── inference/             ← Hypothesis testing & confidence intervals
│   │   ├── CI.py
│   │   └── HypothesisTestDeploy.py
│   └── regression/            ← Regression & gradient descent
│       ├── LR.py
│       └── MultipleLinearRegression.py
│
└── src/                       ← Shared utilities
    └── boot.py
```

---

## 🗂️ Apps by Topic

### 📈 Distributions — `apps/distributions/`

| App | Description | Run |
|-----|-------------|-----|
| `CLT.py` | Central Limit Theorem visualizer — sample size vs. normality | `streamlit run apps/distributions/CLT.py` |
| `PoissonMLE.py` | Poisson distribution with Maximum Likelihood Estimation | `streamlit run apps/distributions/PoissonMLE.py` |

### 🔬 Statistical Inference — `apps/inference/`

| App | Description | Run |
|-----|-------------|-----|
| `CI.py` | Confidence interval explorer | `streamlit run apps/inference/CI.py` |
| `HypothesisTestDeploy.py` | Hypothesis testing with T-test model | `streamlit run apps/inference/HypothesisTestDeploy.py` |

### 📉 Regression — `apps/regression/`

| App | Description | Run |
|-----|-------------|-----|
| `LR.py` | Linear regression with enhanced gradient descent and UI | `streamlit run apps/regression/LR.py` |
| `MultipleLinearRegression.py` | Multiple linear regression with interactive inputs | `streamlit run apps/regression/MultipleLinearRegression.py` |

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/JishanAhmed2019/math3410_fall25.git
cd math3410_fall25

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run any app
streamlit run apps/distributions/CLT.py
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io) | Interactive Python web apps |
| [NumPy / SciPy](https://scipy.org) | Statistical computation |
| [Matplotlib / Plotly](https://plotly.com) | Data visualization |

---

## 👤 Author

**Jishan Ahmed** — [JishanAhmed2019](https://github.com/JishanAhmed2019)
Weber State University · Department of Mathematics

---

*See also: [WSUMath3410](https://github.com/JishanAhmed2019/WSUMath3410) for the full app collection.*
