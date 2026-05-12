import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm

st.set_page_config(page_title="Confidence Interval Visualization", layout="wide")

# Adding a title and explanation to the app
st.title("Confidence Interval Visualization App")
st.markdown("""
    This app demonstrates the concept of confidence intervals in statistics.
    Given a set of observations, we can calculate the mean and determine the range
    within which we expect the true mean of the population to lie, with a certain level of confidence.
""")

# Streamlit sidebar options
with st.sidebar:
    st.header("User Input Parameters")
    num_columns = st.number_input('Number of Samples', min_value=1, value=100)
    num_observations = st.number_input('Number of Observations', min_value=1, value=10000)
    confidence_level = st.slider('Confidence Level', min_value=0.0, max_value=1.0, value=0.90)
    #st.text("Created by Dr. Jishan Ahmed")

# Generate the data
np.random.seed(42)
data = np.random.randn(num_observations, num_columns)

# Compute the mean for each variable
means = np.mean(data, axis=0)

# Compute the standard error of the mean (SEM)
stderr = np.std(data, axis=0, ddof=1) / np.sqrt(num_observations)

# Calculate the z-scores for the confidence interval
z_score = norm.ppf(1 - (1 - confidence_level) / 2)

# Calculate the margins of error for each mean
margins_of_error = z_score * stderr

# Calculate the confidence intervals
ci_lower = means - margins_of_error
ci_upper = means + margins_of_error

# Create a boolean array: True if the CI includes the true mean (0), False otherwise
ci_includes_zero = (ci_lower <= 0) & (ci_upper >= 0)

# Create the figure
fig = go.Figure()

# Add traces for CIs containing the true mean
fig.add_trace(go.Scatter(
    x=np.arange(num_columns)[ci_includes_zero],
    y=means[ci_includes_zero],
    error_y=dict(
        type='data',
        symmetric=True,
        array=margins_of_error[ci_includes_zero],
        visible=True
    ),
    mode='markers',
    name='CI Including True Mean',
    marker=dict(size=10, color='green')
))

# Add traces for CIs not containing the true mean
fig.add_trace(go.Scatter(
    x=np.arange(num_columns)[~ci_includes_zero],
    y=means[~ci_includes_zero],
    error_y=dict(
        type='data',
        symmetric=True,
        array=margins_of_error[~ci_includes_zero],
        visible=True
    ),
    mode='markers',
    name='CI Not Including True Mean',
    marker=dict(size=10, color='red')
))

# Add a line for the true population mean
fig.add_hline(y=0, line=dict(color='blue', dash='dash'), name='True Population Mean (0)')

# Customize the layout
fig.update_layout(
    title='Sample Means and Z-Interval Confidence Intervals',
    xaxis_title='Variable Number',
    yaxis_title='Mean Value',
    showlegend=True,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# Display the figure in the Streamlit app
st.plotly_chart(fig, use_container_width=True)

# Signature at the bottom
st.markdown("---")
st.markdown("Created by Dr. Jishan Ahmed")
