import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
# Function to load data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    return df

# Function to display summary statistics
def display_summary(df):
    summary_df = df.describe().transpose()
    missing = df.isna().sum()
    missing_percent = (missing / len(df)) * 100
    summary_df['Missing (%)'] = missing_percent
    return summary_df

# Function to create a Q-Q plot and a box plot side by side
def qq_and_box_plot(df, column):
    sample_data = df[column].dropna()

    # Create a matplotlib figure and axes for two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Generate Q-Q plot on the first subplot
    sm.qqplot(sample_data, color='darkorange', line='45', ax=ax1, fit=True)
    ax1.set_title('Q-Q plot to Check Normality Assumption')

    # Generate Box plot on the second subplot
    sns.boxplot(y=sample_data,color='crimson', ax=ax2)
    ax2.set_title(f'Box Plot of {column}')

    # Show the plots in Streamlit
    st.pyplot(fig)

# Function to perform a one-sample t-test and plot results
def one_sample_t_test_plot(df, column, popmean, alpha):
    sample_data = df[column].dropna()
    sample_size = len(sample_data)
    sample_mean = np.mean(sample_data)
    sample_std = np.std(sample_data, ddof=1)
    standard_error = sample_std / np.sqrt(sample_size)
    t_statistic = (sample_mean - popmean) / standard_error
    df = sample_size - 1
    p_value = 2 * stats.t.sf(np.abs(t_statistic), df)
    ci = stats.t.interval(1-alpha, df, loc=sample_mean, scale=standard_error)
    
    # Plotting
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(sample_data, kde=True, color="lime", alpha=0.7, ax=ax[0])
    ax[0].axvline(sample_mean, color='red', linestyle='dashed', linewidth=2)
    ax[0].set_title(f'Distribution of {column}')
    ax[0].text(sample_mean+0.5, ax[0].get_ylim()[1]*0.9, f'Mean: {sample_mean:.2f}\nSD: {sample_std:.2f}', color='red')

    x = np.linspace(stats.t.ppf(0.001, df), stats.t.ppf(0.999, df), 100)
    ax[1].plot(x, stats.t.pdf(x, df), 'b-', lw=2, alpha=0.6, label='t-distribution')
    ax[1].axvline(x=t_statistic, color='green', linestyle='--', label=f'T-statistic = {t_statistic:.2f}')
    ax[1].fill_between(x, 0, stats.t.pdf(x, df), where=(x >= stats.t.ppf(1-alpha/2, df)), color='red', alpha=0.5, label='Rejection region')
    ax[1].fill_between(x, 0, stats.t.pdf(x, df), where=(x <= stats.t.ppf(alpha/2, df)), color='red', alpha=0.5)
    
    ax[1].annotate(f'$H_0: \mu={popmean}$', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12)
    ax[1].annotate(f'$H_a: \mu\\neq{popmean}$', xy=(0.05, 0.90), xycoords='axes fraction', fontsize=12)
    ax[1].legend()
    ax[1].set_title('T-distribution with Rejection Region')

    st.pyplot(fig)

    # Decision based on p-value
    decision = "Reject the null hypothesis" if p_value < alpha else "Fail to reject the null hypothesis"
    decision_color = "red" if p_value < alpha else "green"

    # Display results in a formatted table
    result_df = pd.DataFrame({
        'Metric': ['T-statistic', 'P-value', 'Confidence Interval'],
        'Value': [f"{t_statistic:.3f}", f"{p_value:.3f}", f"({ci[0]:.3f}, {ci[1]:.3f})"]
    })
    return decision, decision_color, result_df

# Set Streamlit layout
st.set_page_config(layout="wide")

# Streamlit UI
st.title('Hypothesis Testing App')

with st.sidebar:
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        test_type = st.selectbox('Select Test Type', ['One-sample t-test'])
        column = st.selectbox('Select Column', df.columns)
        popmean = st.number_input('Population Mean (μ0)', value=0.0)
        alpha = st.slider('Significance Level (α)', 0.01, 0.10, 0.05)
        test_button = st.button('Perform One-sample t-test')

# Main Area
if uploaded_file is not None:
    st.markdown("<h2 style='color:blue;'>Data Preview</h2>", unsafe_allow_html=True)
    st.dataframe(df.head())
    st.markdown("<h2 style='color:blue;'>Summary Statistics</h2>", unsafe_allow_html=True)
    summary_df = display_summary(df)
    st.dataframe(summary_df)
    st.markdown("<h2 style='color:blue;'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    qq_and_box_plot(df, column) # Displaying the Q-Q plot
    

    if test_button and test_type == 'One-sample t-test':
        st.markdown("<h2 style='color:orangered;'>Hypothesis Testing</h2>", unsafe_allow_html=True)
        decision, decision_color, result_df = one_sample_t_test_plot(df, column, popmean, alpha)
        st.markdown("<h2 style='color:magenta;'>Results</h2>", unsafe_allow_html=True)
        #st.write("### Test Results")
        st.dataframe(result_df)
        st.markdown(f"<h4 style='color:{decision_color};'>{decision}</h4>", unsafe_allow_html=True)
