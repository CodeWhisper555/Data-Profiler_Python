import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Analytics Engine", page_icon="📊", layout="wide")

st.title("🐍 Advanced Statistical Engine")
st.markdown("---")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Sidebar stats
    st.sidebar.header("Dataset Overview")
    st.sidebar.write(f"Total Rows: {df.shape[0]}")
    st.sidebar.write(f"Total Columns: {df.shape[1]}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

    with col2:
        st.subheader("Key Statistics")
        st.write(df.describe())

    st.markdown("---")
    
    # AI Analysis Section: Correlation Heatmap
    st.subheader("Feature Correlation Matrix")
    st.info("Automated analysis of relationships between numerical variables.")
    
    # Filter only numeric columns for the heatmap
    numeric_df = df.select_dtypes(include=['number'])
    
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        # Using the 'mako' palette looks very professional/modern
        sns.heatmap(numeric_df.corr(), annot=True, cmap='mako', fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numerical features found to calculate correlations.")

    st.markdown("---")
    st.caption("Developed as part of the Intelligent Data Profiler Suite.")