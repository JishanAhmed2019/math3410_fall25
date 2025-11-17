import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import expon, norm

# === Page config ===
st.set_page_config(
    page_title="CLT Demo: Exponential Distribution",
    layout="centered"
)

# === Title ===
st.title("🔍 Central Limit Theorem Demo")
st.markdown("**Exponential Distribution • Time-to-Failure Modeling**")
st.markdown("""
Even with **skewed failure times**, the distribution of **sample means** becomes **approximately normal**—demonstrating the **Central Limit Theorem (CLT)**.
""")

# === Sidebar: Distribution Parameter ===
with st.sidebar:
    st.header("⚙️ Population Settings")
    lambda_val = st.slider("Failure rate (λ)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    scale = 1.0 / lambda_val  # population mean

# === Part 1: Exponential PDF ===
st.header("1. Population Distribution")
x = np.linspace(0, np.percentile(expon(scale=scale).rvs(100000), 99.5), 500)
pdf = expon.pdf(x, scale=scale)

fig1, ax1 = plt.subplots(figsize=(8, 3))
ax1.fill_between(x, pdf, color='steelblue', alpha=0.4)
ax1.plot(x, pdf, color='steelblue', lw=2.2)
ax1.axvline(scale, color='darkred', linestyle='--', lw=2, label=f'True Mean = {scale:.2f}')
ax1.set_xlabel("Failure Time (t)")
ax1.set_ylabel("Density")
ax1.set_title(f"Exponential Distribution (λ = {lambda_val:.1f})")
ax1.legend()
ax1.grid(alpha=0.3, linestyle=':')
st.pyplot(fig1)

# === Part 2: 100 Samples in Grid ===
st.header("2. Simulated Failure Times (100 Observations)")
np.random.seed(42)
samples = np.random.exponential(scale=scale, size=100)
samples_int = np.round(samples).astype(int)

grid_df = pd.DataFrame(
    samples_int.reshape(10, 10),
    columns=[f"C{i+1}" for i in range(10)],
    index=[f"R{i+1}" for i in range(10)]
)
st.table(grid_df)

# === Part 3: Student Means Input OR Simulation ===
st.header("3. Sample Means for CLT")

tab_manual, tab_sim = st.tabs(["✍️ Enter Student Means", "🤖 Simulate Means"])

student_means = None

# --- Tab 1: Manual Entry ---
with tab_manual:
    means_input = st.text_area(
        "Enter sample means (comma or space separated):",
        value="1.2, 0.9, 1.1, 1.3, 0.8, 1.0, 1.4, 1.2, 0.95, 1.05"
    )
    if means_input.strip():
        try:
            parsed = [float(x.strip()) for x in means_input.replace(',', ' ').split() if x.strip()]
            student_means = np.array(parsed)
        except ValueError:
            st.error("❌ Invalid input. Please enter numbers only.")

# --- Tab 2: Simulation ---
with tab_sim:
    st.markdown("Simulate sample means from the exponential distribution above.")
    col1, col2 = st.columns(2)
    with col1:
        n_per_mean = st.number_input("Sample size per mean (n)", min_value=1, max_value=1000, value=30, step=5)
    with col2:
        n_sim_means = st.number_input("Number of simulated means", min_value=1, max_value=5000, value=100, step=10)
    if st.button("🎲 Generate Simulated Means"):
        rng = np.random.default_rng()
        data = rng.exponential(scale=scale, size=(n_sim_means, n_per_mean))
        student_means = data.mean(axis=1)

# === Part 4: Plot CLT Results ===
if student_means is not None and len(student_means) > 0:
    st.header("4. Distribution of Sample Means (CLT)")

    n_means = len(student_means)
    mean_of_means = np.mean(student_means)
    std_of_means = np.std(student_means, ddof=1) if n_means > 1 else 0.0

    st.success(f"📊 Analyzing {n_means} sample means → mean = {mean_of_means:.3f}")

    # Plot
    fig2, ax2 = plt.subplots(figsize=(8.5, 4))
    
    if n_means == 1:
        ax2.axvline(student_means[0], color='lightcoral', lw=5, label='Single Mean')
        ax2.set_xlim(mean_of_means - 1, mean_of_means + 1)
    else:
        # Histogram
        bins = min(30, max(8, n_means // 10)) if n_means <= 200 else 30
        ax2.hist(student_means, bins=bins, density=True, alpha=0.65,
                 color='lightcoral', edgecolor='white', linewidth=0.7, label='Sample Means')
        # Normal approximation
        x_norm = np.linspace(student_means.min() - 0.3, student_means.max() + 0.3, 300)
        if n_means > 1:
            clt_pdf = norm.pdf(x_norm, loc=mean_of_means, scale=std_of_means)
            ax2.plot(x_norm, clt_pdf, 'b--', lw=2.4, label='Normal Fit (CLT)')

    # True population mean
    ax2.axvline(scale, color='darkgreen', linestyle=':', lw=2.5, label=f'Population Mean = {scale:.2f}')

    ax2.set_xlabel("Sample Mean")
    ax2.set_ylabel("Density")
    ax2.set_title("Sampling Distribution of the Mean → CLT in Action!", fontsize=13)
    ax2.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
    ax2.grid(alpha=0.35, linestyle=':')
    st.pyplot(fig2)

    if n_means < 10:
        st.info("💡 The CLT becomes visually clear with more sample means (e.g., ≥20). Try simulating 100+ means!")

# === Signature ===
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 0.95em; color: #555; margin-top: 1em;'>
        Developed by <strong>Dr. Jishan Ahmed</strong><br>
        Assistant Professor of Data Science • Weber State University
    </div>
    """,
    unsafe_allow_html=True
)

# === Theory Notes ===
with st.expander("📘 Theory & Teaching Notes"):
    st.markdown(r"""
    - **Exponential PDF**: $ f(t) = \lambda e^{-\lambda t} $ for $ t \geq 0 $  
      → Mean = $ \mu = 1/\lambda $, Variance = $ 1/\lambda^2 $
    - **Central Limit Theorem**: For large $n$, $ \bar{X} \overset{\text{approx}}{\sim} \mathcal{N}(\mu, \sigma^2/n) $
    - **Why this works**: Even though individual failures are skewed, **averaging reduces skewness**.
    """)