import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

# Function to calculate MLE and create plot
def mle_plot(data):
    lambda_estimate = np.mean(data)
    lambda_range = np.linspace(0.1, 10, 200)
    likelihoods = [np.prod([poisson.pmf(k, lambda_val) for k in data]) for lambda_val in lambda_range]
    data_hist, bins = np.histogram(data, bins=range(int(np.max(data)) + 2), density=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # MLE plot
    ax1.plot(lambda_range, likelihoods, color='tab:blue', label='Likelihood Curve')
    ax1.axvline(x=lambda_estimate, color='tab:red', linestyle='--', label=f'Estimated λ: {lambda_estimate:.2f}')
    ax1.set_title('MLE Computation')
    ax1.set_xlabel('λ Value')
    ax1.set_ylabel('Likelihood')
    ax1.legend()

    # Data distribution plot
    ax2.bar(bins[:-1], data_hist, width=0.5, color='tab:orange', alpha=0.6, label='Data Histogram')
    poisson_dist = [poisson.pmf(k, lambda_estimate) for k in range(len(bins)-1)]
    ax2.plot(poisson_dist, color='tab:blue', marker='o', linestyle='-', lw=2, label=f'Poisson Dist (λ: {lambda_estimate:.2f})')
    ax2.set_title('Observed Data Distribution')
    ax2.set_xlabel('Number of Cases')
    ax2.set_ylabel('Probability/Frequency')
    ax2.legend()

    plt.tight_layout()
    return fig

# Streamlit app layout
st.markdown("<h1 style='text-align: center; color: blue;'>Poisson MLE Visualization</h1>", unsafe_allow_html=True)

# Data input options: file upload or manual list entry
data_input_mode = st.radio("Choose how to input data:", ['Upload a file', 'Enter a list of numbers'])

data = None

if data_input_mode == 'Upload a file':
    uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        # Read the file into a pandas dataframe
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        data = data.iloc[:, 0].tolist()  # Assuming the data is in the first column

elif data_input_mode == 'Enter a list of numbers':
    numbers_input = st.text_area("Enter a list of numbers, separated by commas:")
    if numbers_input:
        try:
            data = [float(num.strip()) for num in numbers_input.split(',')]
        except ValueError:
            st.error("Please enter valid numbers, separated by commas.")

# Compute MLE and plot
if data is not None and st.button('Compute MLE and Show Plot'):
    fig = mle_plot(data)
    st.pyplot(fig)

# Signature
st.markdown("<h4 style='text-align: center; color: purple;'>Created by Dr. Jishan Ahmed</h4>", unsafe_allow_html=True)
