import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.graphics.regressionplots import influence_plot
from sklearn.impute import SimpleImputer
# Set seeds for reproducibility
import random
np.random.seed(42)  # For NumPy operations
random.seed(42)     # For Python's random operations
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            return pd.read_excel(uploaded_file)

def delete_columns(data):
    cols_to_delete = st.multiselect("Select columns to delete", data.columns)
    return data.drop(columns=cols_to_delete, errors='ignore')

def impute_missing_values(data, strategy):
    imputer = SimpleImputer(strategy=strategy)
    for col in data.columns:
        if data[col].isnull().any():
            data[col] = imputer.fit_transform(data[[col]])
    return data

def plot_diagnostic_plots(fitted_model, X, residuals):
    plt.figure(figsize=(8, 6))
    plt.scatter(X @ fitted_model.params, residuals, alpha=0.5, color='blue')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values', color='green')
    st.pyplot(plt)

    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Normal Q-Q Plot', color='green')
    st.pyplot(plt)

    plt.figure(figsize=(8, 6))
    influence_plot(fitted_model, criterion="cooks", plot_alpha=0.5, ax=plt.gca())
    plt.title('Residuals vs Leverage', color='green')
    st.pyplot(plt)

def summary_statistics(data):
    summary = pd.DataFrame({'Dtype': data.dtypes})
    summary['Missing %'] = 100 * (1 - data.count() / len(data))
    summary['Unique'] = data.nunique()
    summary['Most Common'] = data.apply(lambda x: x.mode()[0] if x.dtype == 'object' else np.nan)
    summary['Mean'] = data.mean(numeric_only=True)
    summary['Std'] = data.std(numeric_only=True)
    return summary

def correlation_heatmap(data):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='viridis')
    plt.title('Correlation Matrix', color='green')
    st.pyplot(plt)

def perform_regression(data, formula):
    model = smf.ols(formula, data).fit()
    st.write(model.summary())
    residuals = model.resid
    st.markdown("<h2 style='color:orangered;'>Diagnostic Plots</h2>", unsafe_allow_html=True)
    #st.subheader("Diagnostic Plots")
    plot_diagnostic_plots(model, model.model.exog, residuals)
    return residuals, model

def plot_tornado_diagram(model):
    coeff = model.params
    coeff = coeff.iloc[(coeff.abs()*-1.0).argsort()]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=coeff.values, y=coeff.index, orient='h', palette='Paired')
    plt.title('Tornado Diagram of Standardized Coefficients', color='green')
    st.pyplot(plt)

def predict_with_model(model, new_data):
    prediction = model.get_prediction(new_data)
    return prediction.summary_frame(alpha=0.05)
    
# Footer for the signature
footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: crimson;
    text-align: center;
    padding: 10px;
    font-size: 16px;
}
</style>
<div class='footer'>
     <p><b>Created by Dr. Jishan Ahmed</b></p>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)   
st.markdown("<h1 style='color: darkred;'>Statistical Inference with Multiple Linear Regression</h1>", unsafe_allow_html=True)

#st.title("Statistical Inference with Multiple Linear Regression")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    if data is not None:
        st.markdown("<h2 style='color:blue;'>Data Load:</h2>", unsafe_allow_html=True)
        st.write(data)
        st.markdown("<h2 style='color:blue;'>Data Preprocessing</h2>", unsafe_allow_html=True)
        #st.subheader("Data Management")
        data = delete_columns(data)
        st.markdown("<h2 style='color:blue;'>Data After Column Deletion:</h2>", unsafe_allow_html=True)
        #st.write("Data after column deletion:")
        st.write(data)
        st.markdown("<h2 style='color:blue;'>Summary Statistics:</h2>", unsafe_allow_html=True)
        #st.subheader("Summary Statistics")
        st.write(summary_statistics(data))

        if data.isnull().values.any():
            imputation_method = st.selectbox("Select imputation method for missing values", 
                                             ["mean", "median", "most_frequent"])
            data = impute_missing_values(data, imputation_method)
            st.write("Data after imputation:")
            st.write(data)

        numeric_cols = data.select_dtypes(include=np.number).columns
        if numeric_cols.size > 0:
            st.markdown("<h2 style='color:blue;'>Correlation Matrix Heatmap</h2>", unsafe_allow_html=True)
            #st.subheader("Correlation Matrix Heatmap")
            correlation_heatmap(data[numeric_cols])

        y = st.selectbox("Select target variable", numeric_cols)
        st.markdown(f"<h2 style='color: blue;'>Distribution of Target:( {y} ) Variable</h2>", unsafe_allow_html=True)
        #st.subheader(f"Distribution of {y}")
        plt.figure(figsize=(8, 4))
        sns.histplot(data[y], kde=True, color='lime')
        st.pyplot(plt)

        features = st.multiselect("Select feature variables", data.columns.drop(y))

        if len(features) > 0:
            formula = f"{y} ~ " + " + ".join(["C({})".format(feature) if data[feature].dtype == 'O' else feature for feature in features])
            residuals, model = perform_regression(data, formula)

            if residuals is not None:
                st.markdown("<h2 style='color:orangered;'>Shapiro-Wilk Test for Normality of Residuals</h2>", unsafe_allow_html=True)
                #st.subheader("Shapiro-Wilk Test for Normality of Residuals")
                shapiro_test = stats.shapiro(residuals)
                st.write(f"Shapiro-Wilk Test Statistic: {shapiro_test[0]}, P-value: {shapiro_test[1]}")

                if shapiro_test[1] < 0.05:
                    transform_option = st.selectbox("Select a transformation for the response variable", ["None", "Log", "Square Root"])
                    if transform_option != "None":
                        original_y = data[y].copy()
                        if transform_option == "Log":
                            data[y] = np.log(data[y]+1)
                        elif transform_option == "Square Root":
                            data[y] = np.sqrt(data[y])
                        #st.markdown("<h2 style='color:orangered;'>Hypothesis Testing</h2>", unsafe_allow_html=True)
                        st.markdown(f"<h2 style='color: orangered;'>Regression Analysis after {transform_option} Transformation</h2>", unsafe_allow_html=True)
                        #st.subheader(f"Regression Analysis after {transform_option} Transformation")
                        residuals, model = perform_regression(data, formula)
                        data[y] = original_y

            if model is not None:
                st.markdown("<h2 style='Tornado Diagram of Standardized Coefficients</h2>", unsafe_allow_html=True)
                #st.subheader("Tornado Diagram of Standardized Coefficients")
                plot_tornado_diagram(model)

                st.markdown("<h2 style='color:magenta;'>Prediction with New Data</h2>", unsafe_allow_html=True)
                #st.subheader("Prediction with New Data")
                new_data = {}
                for feature in features:
                    if data[feature].dtype == 'O':
                        options = data[feature].unique()
                        new_data_value = st.selectbox(f"Select value for {feature}", options)
                    else:
                        new_data_value = st.number_input(f"Input value for {feature}", value=0)
                    new_data[feature] = new_data_value

                if st.button("Predict"):
                    prediction_df = pd.DataFrame([new_data])
                    prediction_result = predict_with_model(model, prediction_df)
                    st.write(prediction_result)
