import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bootstrap Inference Tool", layout="wide")
st.title("📊 Bootstrap Sampling for Mean or Proportion")
st.markdown("""
Upload your data, explore the original sample distribution, then **click to perform bootstrap inference**.
This stepwise approach supports conceptual clarity in statistical reasoning.
""")

# === File Upload ===
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success(f"✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
else:
    st.info("📁 Please upload a CSV or Excel file to begin.")
    st.stop()

# === Identify suitable columns ===
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
binary_cols = []

for col in df.select_dtypes(include=['object', 'category']).columns:
    unique_vals = df[col].dropna().unique()
    if len(unique_vals) == 2:
        binary_cols.append(col)

available_cols = numeric_cols + binary_cols

if not available_cols:
    st.warning("⚠️ No numeric or binary columns found for analysis.")
    st.stop()

selected_cols = st.multiselect(
    "Select columns to explore",
    options=available_cols,
    default=available_cols[:1]
)

if not selected_cols:
    st.info("🔍 Please select at least one column.")
    st.stop()

# Slider for bootstrap samples (global, but only used when triggered)
n_bootstrap = st.slider("Number of bootstrap resamples (used when activated)", 
                        min_value=100, max_value=10000, value=2000, step=100)

np.random.seed(42)  # For reproducibility

# === Helper: detect binary ===
def is_binary_series(series):
    return series.dropna().nunique() == 2

# === Per-column: Explore first, Bootstrap on demand ===
for col in selected_cols:
    st.markdown(f"### 📌 Column: `{col}`")
    series = df[col].dropna()
    if series.empty:
        st.warning(f"Column `{col}` has no valid data.")
        continue

    # Infer type
    if col in numeric_cols and not is_binary_series(series):
        analysis_type = "mean"
        clean_data = series.values.astype(float)
        true_stat = np.mean(clean_data)
    elif col in binary_cols or is_binary_series(series):
        analysis_type = "proportion"
        unique_vals = sorted(series.dropna().unique())
        label_0, label_1 = str(unique_vals[0]), str(unique_vals[1])
        # Compute observed proportion of label_1
        prop_1 = np.mean(series == unique_vals[1])
        prop_0 = 1 - prop_1
    else:
        st.warning(f"Column `{col}` is not suitable for inference.")
        continue

    # === STEP 1: Original Sample Summary & Distribution ===
    st.markdown("#### 📊 Original Sample Summary")

    if analysis_type == "mean":
        summary_df = pd.DataFrame({
            'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', 'Q1 (25%)', 'Median (50%)', 'Q3 (75%)', 'Max'],
            'Value': [
                len(clean_data),
                np.mean(clean_data),
                np.std(clean_data, ddof=1),
                np.min(clean_data),
                np.percentile(clean_data, 25),
                np.median(clean_data),
                np.percentile(clean_data, 75),
                np.max(clean_data)
            ]
        })
        st.dataframe(summary_df.set_index('Statistic').T)

        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(clean_data, kde=True, ax=ax, color='lightblue', edgecolor='black')
        ax.set_title("Original Sample Distribution")
        ax.set_xlabel(col)
        st.pyplot(fig)

    else:  # binary
        count_0 = np.sum(series == unique_vals[0])
        count_1 = np.sum(series == unique_vals[1])
        summary_df = pd.DataFrame({
            'Category': [label_0, label_1],
            'Count': [count_0, count_1],
            'Proportion': [prop_0, prop_1]
        })
        st.dataframe(summary_df.set_index('Category'))

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar([label_0, label_1], [count_0, count_1], color=['#ff9999', '#66b3ff'])
        ax.set_title("Original Sample Distribution")
        ax.set_ylabel("Count")
        plt.xticks(rotation=0)
        st.pyplot(fig)

    # === STEP 2: Bootstrap on Demand ===
    st.markdown("#### 🔁 Bootstrap Inference")
    bootstrap_key = f"bootstrap_{col}"
    
    if st.button(f"▶️ Perform Bootstrap for `{col}`", key=bootstrap_key):
        if analysis_type == "mean":
            boot_samples = np.array([
                np.mean(np.random.choice(clean_data, size=len(clean_data), replace=True))
                for _ in range(n_bootstrap)
            ])
            observed_stat = true_stat
            stat_label = "Mean"
        else:
            # ✅ CORRECTED: Resample original categorical data, compute proportion of label_1
            orig_vals = series.values  # original category labels
            n = len(orig_vals)
            boot_props = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(orig_vals, size=n, replace=True)
                prop = np.mean(sample == unique_vals[1])  # proportion of label_1
                boot_props.append(prop)
            boot_samples = np.array(boot_props)
            observed_stat = prop_1
            stat_label = f"Proportion ({label_1})"

        # Compute 95% CI
        ci_low, ci_high = np.percentile(boot_samples, [2.5, 97.5])

        col1, col2, col3 = st.columns(3)
        col1.metric(f"Observed {stat_label}", f"{observed_stat:.4f}")
        col2.metric("95% CI Lower", f"{ci_low:.4f}")
        col3.metric("95% CI Upper", f"{ci_high:.4f}")

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.histplot(boot_samples, kde=True, ax=ax2, color='steelblue', alpha=0.7)
        ax2.axvline(observed_stat, color='red', linestyle='--', label=f'Observed {stat_label}')
        ax2.axvline(ci_low, color='green', linestyle=':', label='95% CI')
        ax2.axvline(ci_high, color='green', linestyle=':')
        ax2.set_title(f'Bootstrap Sampling Distribution ({stat_label})')
        ax2.set_xlabel(f'Bootstrap {stat_label}')
        ax2.set_ylabel('Density')
        ax2.legend()
        st.pyplot(fig2)

        st.markdown(f"**Interpretation**: We are 95% confident the true population {stat_label.lower()} lies between **{ci_low:.4f}** and **{ci_high:.4f}**.")

    st.markdown("---")

st.caption("Designed for teaching statistical inference • Dr. Jishan Ahmed, Department of Mathematics, Weber State University")