import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Data Intelligence Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Title and Subtitle
st.title("Data Intelligence & Correlation Engine")
st.caption("Professional Statistical Analysis | Dual-Tier Architecture")
st.divider()

# 3. Sidebar for Data Handshake
with st.sidebar:
    st.header("⚙️ System Sync")
    uploaded_file = st.file_uploader("Upload CSV for Cloud Processing", type="csv")
    if uploaded_file:
        st.success("Data Handshake Successful")

# 4. Main Engine Logic
if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    numeric_df = df.select_dtypes(include=[np.number])

    # --- SECTION I: EXECUTIVE SUMMARY ---
    st.subheader("I. Executive Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", len(df))
    m2.metric("Total Features", len(df.columns))
    m3.metric("Numeric Features", len(numeric_df.columns))
    m4.metric("Missing Values", df.isnull().sum().sum())

    st.divider()

    # --- SECTION II: DISTRIBUTION INTELLIGENCE ---
    if not numeric_df.empty:
        st.subheader("II. Distribution Intelligence")
        st.info("Analyzing Skewness (Symmetry) and Kurtosis (Tail-heaviness) to determine data normality.")
        
        # Calculate Statistical Moments
        stats_summary = pd.DataFrame({
            'Mean': numeric_df.mean(),
            'Skewness': numeric_df.skew(),
            'Kurtosis': numeric_df.kurtosis()
        })
        st.dataframe(stats_summary, use_container_width=True)
        
        

        # --- SECTION III: CORRELATION ARCHITECTURE ---
        st.divider()
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.subheader("III. Feature Dependency Map")
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt=".2f", ax=ax)
            st.pyplot(fig)
        
        with col_right:
            st.subheader("IV. Key Insights")
            if len(corr.columns) > 1:
                # Logic to find top 3 correlations
                sol = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                      .stack()
                      .sort_values(ascending=False))
                
                st.write("📈 **Top Positive Correlations:**")
                for i in range(min(3, len(sol))):
                    st.write(f"• {sol.index[i][0]} & {sol.index[i][1]}: `{sol[i]:.2f}`")
            else:
                st.write("Add more numeric columns for correlation mapping.")

        # --- SECTION IV: PREDICTIVE TRENDS ---
        st.divider()
        st.subheader("V. Predictive Regression Analysis")
        
        if len(numeric_df.columns) >= 2:
            c1, c2 = st.columns(2)
            x_col = c1.selectbox("Independent Variable (X)", numeric_df.columns, index=0)
            y_col = c2.selectbox("Dependent Variable (Y)", numeric_df.columns, index=1)
            
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            sns.regplot(data=df, x=x_col, y=y_col, ax=ax2, 
                        scatter_kws={'alpha':0.4, 'color':'#0984e3'}, 
                        line_kws={'color':'#d63031'})
            plt.title(f"Linear Trend: {y_col} vs {x_col}")
            st.pyplot(fig2)
            
            

else:
    st.warning("Awaiting Data Handshake... Please upload the CSV in the sidebar to begin analysis.")
