"""
Central Limit Theorem — Interactive Classroom Demo
Author: Prof. Jishan Ahmed style demo (Weber State University)
Run with:  streamlit run clt_demo_app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Central Limit Theorem Demo", layout="wide")

st.title("🎲 Central Limit Theorem — Interactive Demo")
st.markdown(
    """
Even when a **population is heavily skewed**, the distribution of **sample means**
becomes approximately **normal (bell-shaped)** as we take many samples.
Let's see it happen live!
"""
)

# ---------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------
st.sidebar.header("⚙️ Controls")

dist_choice = st.sidebar.selectbox(
    "Skewed population distribution",
    ["Exponential (right-skewed)", "Log-normal (right-skewed)", "Chi-square (right-skewed)"],
)

seed = st.sidebar.number_input("Random seed (for reproducibility)", value=42, step=1)

num_samples = st.sidebar.slider(
    "Number of random samples to draw", min_value=100, max_value=200, value=150, step=10
)

sample_size = st.sidebar.slider(
    "Size of each sample (n)", min_value=10, max_value=30, value=20, step=1
)

rng = np.random.default_rng(int(seed))

# ---------------------------------------------------------------
# STEP 1: Generate 100 numbers from a skewed distribution
# ---------------------------------------------------------------
st.header("Step 1 — Generate 100 numbers from a skewed population")

if dist_choice.startswith("Exponential"):
    population = rng.exponential(scale=10, size=100)
elif dist_choice.startswith("Log-normal"):
    population = rng.lognormal(mean=2, sigma=0.6, size=100)
else:
    population = rng.chisquare(df=3, size=100) * 5

population = np.round(population, 2)

pop_mean = population.mean()
pop_std = population.std(ddof=0)
pop_skew = stats.skew(population)

c1, c2, c3 = st.columns(3)
c1.metric("Population Mean (μ)", f"{pop_mean:.2f}")
c2.metric("Population Std Dev (σ)", f"{pop_std:.2f}")
c3.metric("Skewness", f"{pop_skew:.2f}")

# ---------------------------------------------------------------
# STEP 2: Show the numbers as a 10 x 10 matrix
# ---------------------------------------------------------------
st.header("Step 2 — The 100 numbers (10 × 10 table)")

matrix = population.reshape(10, 10)
df_matrix = pd.DataFrame(
    matrix,
    columns=[f"C{j+1}" for j in range(10)],
    index=[f"R{i+1}" for i in range(10)],
)
st.dataframe(df_matrix.style.format("{:.2f}").background_gradient(cmap="YlOrRd"), 
             use_container_width=True)

st.caption("Darker cells = larger values. Notice a few big values pulling the tail to the right!")

# ---------------------------------------------------------------
# STEP 3: Histogram of the population
# ---------------------------------------------------------------
st.header("Step 3 — Histogram of the population (clearly skewed!)")

fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.hist(population, bins=15, color="#ff7f0e", edgecolor="white")
ax1.axvline(pop_mean, color="red", linestyle="--", linewidth=2, label=f"Mean = {pop_mean:.2f}")
ax1.set_xlabel("Value")
ax1.set_ylabel("Frequency")
ax1.set_title(f"Population Distribution: {dist_choice}")
ax1.legend()
st.pyplot(fig1)

# ---------------------------------------------------------------
# STEP 4: Draw samples, compute means, plot histogram of means
# ---------------------------------------------------------------
st.header("Step 4 — Sampling distribution of the mean ✨")

st.markdown(
    f"""
We now draw **{num_samples} random samples**, each of size **n = {sample_size}**
(with replacement) from those 100 numbers, compute the **mean of each sample**,
and plot all {num_samples} means.
"""
)

if st.button("🚀 Draw samples and plot the sample means!", type="primary"):
    sample_means = np.array([
        rng.choice(population, size=sample_size, replace=True).mean()
        for _ in range(num_samples)
    ])

    means_mean = sample_means.mean()
    means_std = sample_means.std(ddof=0)
    theoretical_se = pop_std / np.sqrt(sample_size)

    m1, m2, m3 = st.columns(3)
    m1.metric("Mean of sample means", f"{means_mean:.2f}",
              delta=f"{means_mean - pop_mean:+.2f} vs μ")
    m2.metric("Std dev of sample means", f"{means_std:.2f}")
    m3.metric("Theory: σ/√n", f"{theoretical_se:.2f}")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(sample_means, bins=20, density=True,
             color="#1f77b4", edgecolor="white", alpha=0.8,
             label="Sample means")

    # Overlay the CLT-predicted normal curve
    x = np.linspace(sample_means.min(), sample_means.max(), 300)
    ax2.plot(x, stats.norm.pdf(x, loc=pop_mean, scale=theoretical_se),
             color="red", linewidth=2.5,
             label=f"Normal(μ={pop_mean:.1f}, σ/√n={theoretical_se:.2f})")

    ax2.axvline(pop_mean, color="red", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Sample Mean")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Distribution of {num_samples} Sample Means (n = {sample_size})")
    ax2.legend()
    st.pyplot(fig2)

    skew_means = stats.skew(sample_means)
    st.success(
        f"🎉 **The magic of the CLT!** The population skewness was **{pop_skew:.2f}**, "
        f"but the sample means have skewness of only **{skew_means:.2f}** — "
        f"much closer to a symmetric bell curve. "
        f"Also notice the spread shrank from σ = {pop_std:.2f} to about σ/√n = {theoretical_se:.2f}."
    )

    with st.expander("💡 Discussion questions for students"):
        st.markdown(
            """
1. What happens to the histogram of sample means if you **increase n** from 10 to 30?
2. What happens if you increase the **number of samples** from 100 to 200?
3. Why does the spread of the sample means equal roughly **σ/√n**?
4. Would the CLT still work if the population were *even more* skewed? Try Log-normal!
"""
        )
else:
    st.info("👆 Set your sample settings in the sidebar, then click the button to see the CLT in action!")
