import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Data Engine", layout="wide")

st.title("🧠 Advanced Statistical Engine")
st.markdown("---")

uploaded_file = st.file_uploader("Upload the same CSV for deep analysis", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # 1. High-Level Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Features", len(df.columns))
    with col2:
        st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024:.2f} KB")
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    # 2. Pearson Correlation Matrix
    st.subheader("🔗 Pearson Correlation Heatmap")
    st.info("This heatmap identifies linear relationships between variables.")
    
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        # The Math: r = Σ((x - μx)(y - μy)) / (nx * ny)
        corr = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns found for correlation analysis.")

    # 3. Automated Insights
    st.subheader("📝 Statistical Summary")
    st.write(df.describe())

    # 4. Export as CSV (The "Report")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Statistical Profile (CSV)",
        data=csv,
        file_name='data_profile.csv',
        mime='text/csv',
    )
