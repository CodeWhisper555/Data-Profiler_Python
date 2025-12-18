import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Page Config for a professional feel
st.set_page_config(
    page_title="Data Intelligence Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner, high-tech look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef; }
    .stPlotlyChart { border-radius: 10px; }
    </style>
    """, unsafe_allow_name_with_html=True)

st.title("Data Intelligence & Correlation Engine")
st.caption("Advanced Statistical Handshake | Dual-Tier Architecture")
st.markdown("---")

# Sidebar for metadata and configuration
with st.sidebar:
    st.header("⚙️ Engine Settings")
    uploaded_file = st.file_uploader("Sync CSV for Cloud Analysis", type="csv")
    if uploaded_file:
        st.success("Data Handshake Successful")

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    numeric_df = df.select_dtypes(include=[np.number])

    # --- 1. EXECUTIVE SUMMARY (High-Level Metrics) ---
    st.subheader("I. Executive Summary")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Records", len(df))
    with m2:
        st.metric("Feature Count", len(df.columns))
    with m3:
        st.metric("Numeric Features", len(numeric_df.columns))
    with m4:
        st.metric("Missing Data Points", df.isnull().sum().sum())

    st.markdown("---")

    # --- 2. DEEP STATISTICAL MOMENTS (Enhanced Features) ---
    st.subheader("II. Distribution Intelligence")
    st.info("Analyzing the 'Shape' of your data. High Skewness (>1 or <-1) suggests the need for data transformation.")
    
    if not numeric_df.empty:
        # Calculate Skewness and Kurtosis (Advanced Math Features)
        moments_df = pd.DataFrame({
            'Skewness': numeric_df.skew(),
            'Kurtosis': numeric_df.kurtosis(),
            'Mean': numeric_df.mean()
        })
        st.table(moments_df.head(10)) # Showing first 10 for clean UI
        
        # --- 3. CORRELATION ARCHITECTURE ---
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.subheader("III. Pearson Correlation Matrix")
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt=".2f", ax=ax)
            plt.title("Inter-Feature Dependency Map")
            st.pyplot(fig)
        
        with col_right:
            st.subheader("IV. Key Insights")
            # Automatically find the highest correlation
            if len(corr) > 1:
                sol = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                      .stack()
                      .sort_values(ascending=False))
                for i in range(min(3, len(sol))):
                    st.write(f"🔹 **Strong Link:** {sol.index[i][0]} & {sol.index[i][1]} ({sol[i]:.2f})")
            else:
                st.write("Not enough features for correlation comparison.")

    # --- 4. PREDICTIVE VISUALIZATION ---
    st.markdown("---")
    st.subheader("V. Regression Analysis & Trend Estimation")
    
    if len(numeric_df.columns) >= 2:
        target_cols = numeric_df.columns.tolist()
        c1, c2 = st.columns(2)
        x_axis = c1.selectbox("Independent Variable (X)", target_cols, index=0)
        y_axis = c2.selectbox("Dependent Variable (Y)", target_cols, index=1)
        
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.regplot(data=df, x=x_axis, y=y_axis, ax=ax2, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.title(f"Predictive Trend: {y_axis} vs {x_axis}")
        st.pyplot(fig2)
        st.caption("The red line represents the Linear Regression model fitting these two variables.")

else:
    st.warning("Waiting for data handshake... Please upload the CSV in the sidebar.")
