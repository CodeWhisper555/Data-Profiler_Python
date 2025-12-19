import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. ROBUST CONFIGURATION
# We use the built-in 'dark' theme capabilities instead of custom CSS to prevent 3.13 crashes
st.set_page_config(page_title="DATA INTELLIGENCE SYSTEM", layout="wide")

# 2. EXECUTIVE HEADER (Using Standard Markdown)
st.title("INTELLIGENCE ANALYTICS ENGINE")
st.text("INTERNAL CLOUD INFRASTRUCTURE | VERSION 4.1.0-STABLE")
st.divider()

# 3. CONTROL INTERFACE
with st.sidebar:
    st.header("SYSTEM PARAMETERS")
    uploaded_file = st.file_uploader("SOURCE DATA UPLOAD (.CSV)", type="csv")
    
    if uploaded_file:
        st.divider()
        st.subheader("PROCESSING PIPELINE")
        impute = st.toggle("AUTOMATED MEAN IMPUTATION", value=True)
        scale_data = st.toggle("Z-SCORE NORMALIZATION", value=False)
        st.write("CORE STATUS: OPERATIONAL")

# 4. DATA ARCHITECTURE LOGIC
if uploaded_file is not None:
    # Safe Data Ingestion
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"DATA_INGESTION_ERROR: {e}")
        st.stop()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # DIAGNOSTIC CALCULATION
    null_count = df.isnull().sum().sum()
    total_cells = df.size if df.size > 0 else 1
    integrity_index = 100 - ((null_count / total_cells) * 100)

    # EXECUTIVE METRIC ROW
    m1, m2, m3 = st.columns(3)
    m1.metric("RECORDS", f"{len(df):,}")
    m2.metric("FEATURES", len(df.columns))
    m3.metric("INTEGRITY", f"{integrity_index:.2f}%")

    # ANALYSIS MODULES
    tab1, tab2, tab3 = st.tabs(["[01] INVENTORY", "[02] CORRELATION", "[03] PROJECTION"])

    with tab1:
        st.subheader("STRUCTURED DATA VIEW")
        st.dataframe(df, use_container_width=True, height=400)
        
    with tab2:
        st.subheader("PEARSON MULTI-VARIATE ANALYSIS")
        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            # Use standard matplotlib styles for stability
            corr = df[num_cols].corr()
            sns.heatmap(corr, annot=True, cmap='RdBu', center=0, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("INSUFFICIENT NUMERIC DIMENSIONS.")

    with tab3:
        st.subheader("PRINCIPAL COMPONENT PROJECTION (PCA)")
        if len(num_cols) >= 2:
            # Data Cleaning for PCA
            pca_input = df[num_cols].fillna(df[num_cols].mean())
            if scale_data:
                pca_input = StandardScaler().fit_transform(pca_input)
            
            # PCA Implementation
            pca = PCA(n_components=2)
            components = pca.fit_transform(pca_input)
            pca_res = pd.DataFrame(components, columns=['PC1', 'PC2'])
            
            # Grouping Selection
            group_col = st.selectbox("IDENTIFY CLUSTER GROUP:", cat_cols) if cat_cols else None
            
            fig_pca, ax_pca = plt.subplots(figsize=(10, 5))
            if group_col:
                pca_res[group_col] = df[group_col].values
                sns.scatterplot(data=pca_res, x='PC1', y='PC2', hue=group_col, palette="viridis", ax=ax_pca)
            else:
                sns.scatterplot(data=pca_res, x='PC1', y='PC2', color="#007BFF", ax=ax_pca)
            
            st.pyplot(fig_pca)
            plt.close(fig_pca)
        else:
            st.error("PCA REQUIRES MINIMUM 2 NUMERIC FEATURES.")
else:
    st.info("SYSTEM_STANDBY: Awaiting CSV synchronization.")
