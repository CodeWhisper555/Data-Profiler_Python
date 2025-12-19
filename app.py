import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. High-End Page Config
st.set_page_config(page_title="Intelligence Studio v2.0", layout="wide")

st.title("Systems Architecture: Advanced Data Engine")
st.caption("Status: Operational | Tier: Research Grade | Logic: PCA & Multi-Variate Analysis")

# 2. Sidebar Command Center
with st.sidebar:
    st.header(" Command Center")
    uploaded_file = st.file_uploader("Sync Data Stream", type="csv")
    
    if uploaded_file:
        st.divider()
        st.subheader("Pipeline Configuration")
        impute_strategy = st.selectbox("Missing Value Strategy", ["None", "Mean", "Median", "Mode"])
        enable_pca = st.checkbox("Enable PCA (Dimensionality Reduction)")
        enable_scaling = st.checkbox("Apply Z-Score Scaling")

# 3. Processing Pipeline
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    numeric_df = df.select_dtypes(include=[np.number])
    
    # --- STAGE 1: SMART IMPUTATION ---
    if impute_strategy != "None":
        for col in numeric_df.columns:
            if numeric_df[col].isnull().any():
                if impute_strategy == "Mean":
                    numeric_df[col] = numeric_df[col].fillna(numeric_df[col].mean())
                elif impute_strategy == "Median":
                    numeric_df[col] = numeric_df[col].fillna(numeric_df[col].median())
        st.success(f"Pipeline: {impute_strategy} Imputation Complete")

    # --- STAGE 2: FEATURE SCALING ---
    processed_df = numeric_df.copy()
    if enable_scaling and not processed_df.empty:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(processed_df)
        processed_df = pd.DataFrame(scaled_data, columns=processed_df.columns)
        st.success("Pipeline: Z-Score Scaling Applied")

    # --- STAGE 3: VISUALIZATION MODULES ---
    tab1, tab2, tab3 = st.tabs(["📊 Diagnostic Summary", "📈 Correlation Matrix", "🧬 Advanced PCA"])

    with tab1:
        st.subheader("Data Health Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(df))
        c2.metric("Features", len(df.columns))
        c3.metric("Null Count", numeric_df.isnull().sum().sum())
        
        st.dataframe(numeric_df.describe(), use_container_width=True)

    with tab2:
        if not numeric_df.empty:
            st.subheader("Pearson Multi-Variate Correlation")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='magma', fmt=".2f")
            st.pyplot(fig)
        else:
            st.warning("Insufficient numeric data for correlation.")

    with tab3:
        if enable_pca and len(processed_df.columns) >= 2:
            st.subheader("Principal Component Analysis (PCA)")
            st.info("PCA reduces high-dimensional data into 2 components while preserving variance.")
            
            pca = PCA(n_components=2)
            components = pca.fit_transform(processed_df.dropna())
            pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=pca_df, x='PC1', y='PC2', alpha=0.7, color="#6c5ce7")
            plt.title(f"PCA Map (Variance Explained: {sum(pca.explained_variance_ratio_)*100:.1f}%)")
            st.pyplot(fig2)
            st.write(f"**Insight:** The data has been compressed. Clusters here represent hidden similarities in the records.")
        else:
            st.info("Enable PCA in the sidebar to visualize high-dimensional clusters.")

    # --- STAGE 4: EXPORT ---
    st.divider()
    csv = numeric_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Engineered Dataset", data=csv, file_name="engineered_data.csv", mime="text/csv")

else:
    st.info("System idling... Awaiting CSV handshake via Sidebar.")
