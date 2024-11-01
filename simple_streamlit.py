import streamlit as st
import pandas as pd
import numpy as np

# Generate sample macroeconomic data
def generate_sample_data():
    np.random.seed(0)
    data = {
        'GDP_Growth': np.random.normal(2, 1, 100),
        'Inflation_Rate': np.random.normal(2.5, 0.5, 100),
        'Unemployment_Rate': np.random.normal(5, 1, 100),
        'Interest_Rate': np.random.normal(3, 0.5, 100),
        'Exchange_Rate': np.random.normal(1, 0.1, 100),
        'Consumer_Confidence': np.random.normal(100, 10, 100),
        'Industrial_Production': np.random.normal(3, 1, 100),
        'Stock_Market_Index': np.random.normal(10000, 500, 100)
    }
    return pd.DataFrame(data)

# Create the Streamlit app
st.title("Macroeconomic Random Forest Parameter Selection")

# Generate and display the sample dataframe
# df = generate_sample_data()
df = pd.read_csv('./csvs/final_dataset.csv', index_col=0)
st.subheader("Sample Macroeconomic Data")
st.dataframe(df.head())

# Allow user to select multiple dependent variables
st.subheader("Select Dependent Variables")
dependent_vars = st.multiselect("Choose the variables to be predicted:", df.columns)

# Allow user to select one independent variable
st.subheader("Select Independent Variable")
independent_var = st.selectbox("Choose the predictor variable:", 
                               [col for col in df.columns if col not in dependent_vars])

# Display the selected variables
if dependent_vars and independent_var:
    st.subheader("Selected Variables")
    st.write("Dependent Variables:", ", ".join(dependent_vars))
    st.write("Independent Variable:", independent_var)
    
    # Display correlations
    st.subheader("Correlations")
    for dep_var in dependent_vars:
        correlation = df[[dep_var, independent_var]].corr().iloc[0, 1]
        st.write(f"Correlation between {dep_var} and {independent_var}: {correlation:.4f}")

    # Display scatter plots
    st.subheader("Scatter Plots")
    for dep_var in dependent_vars:
        st.write(f"{dep_var} vs {independent_var}")
        st.scatter_chart(df[[independent_var, dep_var]])

# Add a note about next steps
st.markdown("""
---
**Note:** After selecting your variables, you can proceed with:
1. Data preprocessing
2. Splitting the data into training and testing sets
3. Implementing the random forest model
4. Evaluating the model's performance
""")
