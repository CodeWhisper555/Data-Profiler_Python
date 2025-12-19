import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import io

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="NEURAL ANALYTICS ENGINE", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0A0C10; }
    .stMetric { background-color: #101214; border: 1px solid #30363D; padding: 15px; border-radius: 5px; }
    .stButton>button { width: 100%; border-radius: 4px; font-family: 'Roboto Mono'; }
</style>
""", unsafe_allow_html=True)

def purify_headers(columns):
    # Standardizes headers to alphanumeric and underscores only
    return [re.sub(r'[^\w]', '_', col).strip('_') for col in columns]

# --- 2. EXECUTIVE SIDEBAR ---
with st.sidebar:
    st.title("ENGINE_CORE")
    st.caption("VERSION 4.2.2-STABLE")
    
    uploaded_file = st.file_uploader("UPLOAD SANITIZED STREAM (.CSV)", type="csv")
    
    if uploaded_file:
        st.divider()
        st.subheader("PIPELINE_CONTROLS")
        clean_headers = st.toggle("PURIFY_HEADERS", value=True)
        impute_strategy = st.selectbox("IMPUTATION_LOGIC", ["Mean", "Median", "Zero Fill"])
        scaling = st.toggle("NORMALIZE_VECTORS", value=True)
        
        st.divider()
        st.subheader("EXPORT_INTERFACE")

# --- 3. DATA ARCHITECTURE & PROCESSING ---
if uploaded_file:
    # Read and immediately fix common penguin data anomalies
    df = pd.read_csv(uploaded_file)
    
    # Header Purification Logic
    if clean_headers:
        df.columns = purify_headers(df.columns)

    # De-noise specific known issues (dots in Sex column)
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].replace('.', np.nan)

    # Detect Features
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Apply Imputation to ensure PCA works
    if num_cols:
        if impute_strategy == "Mean":
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        elif impute_strategy == "Median":
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        else:
            df[num_cols] = df[num_cols].fillna(0)

    # --- DOWNLOAD LOGIC ---
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    with st.sidebar:
        st.download_button(label="DOWNLOAD_REFINED_DATA", data=csv_buffer.getvalue(), 
                         file_name="refined_data.csv", mime="text/csv")

    # --- 4. DASHBOARD LAYOUT ---
    st.title("ANALYTIC_DASHBOARD")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RECORDS", len(df))
    m2.metric("NUMERIC", len(num_cols))
    m3.metric("CATEGORIES", len(cat_cols))
    m4.metric("STATUS", "READY")

    tab_data, tab_corr, tab_pca, tab_bio = st.tabs(["DATA_HEALTH", "CORRELATION", "PROJECTION", "BIO_PROFILING"])

    with tab_data:
        st.subheader("STRUCTURED_DATA_VIEW")
        st.dataframe(df, use_container_width=True)

    with tab_corr:
        st.subheader("PEARSON_RELATIONSHIP_MATRIX")
        if len(num_cols) > 1:
            fig_corr, ax_c = plt.subplots(figsize=(8, 5))
            fig_corr.patch.set_facecolor('#0A0C10')
            sns.heatmap(df[num_cols].corr(), annot=True, cmap='RdBu', center=0, ax=ax_c)
            st.pyplot(fig_corr)
        else:
            st.warning("INSUFFICIENT_NUMERIC_VECTORS_FOR_CORRELATION")

    with tab_pca:
        st.subheader("DIMENSIONAL_PROJECTION")
        if len(num_cols) >= 2:
            # 1. Prepare Data
            pca_data = df[num_cols].dropna() # Safety drop
            if scaling:
                pca_data = StandardScaler().fit_transform(pca_data)
            
            # 2. Run PCA
            pca = PCA(n_components=2)
            components = pca.fit_transform(pca_data)
            
            # 3. Create matched dataframe (FIXES THE INDEX MISMATCH ERROR)
            pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'], index=df.index)
            
            # 4. User Selection
            hue_col = st.selectbox("COLOR_DIMENSION:", cat_cols, key="pca_hue")
            
            # 5. Plotting (Passing the full DF ensures Column Name alignment)
            pca_df[hue_col] = df[hue_col]
            fig_pca, ax_p = plt.subplots(figsize=(10, 6))
            fig_pca.patch.set_facecolor('#0A0C10')
            sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=hue_col, ax=ax_p, palette="viridis")
            st.pyplot(fig_pca)
        else:
            st.error("PCA_FAILURE: 2+ NUMERIC COLUMNS REQUIRED")

    with tab_bio:
        st.subheader("DISTRIBUTION_ANALYSIS")
        if num_cols and cat_cols:
            # Added safe selection to prevent KeyError
            x_ax = st.selectbox("CATEGORICAL_AXIS:", cat_cols, key="bio_x")
            y_ax = st.selectbox("MEASUREMENT_AXIS:", num_cols, key="bio_y")
            
            fig_bio, ax_b = plt.subplots(figsize=(10, 4))
            fig_bio.patch.set_facecolor('#0A0C10')
            # Using dataframe reference instead of direct series for robustness
            sns.violinplot(data=df, x=x_ax, y=y_ax, ax=ax_b, palette="muted")
            st.pyplot(fig_bio)
        else:
            st.warning("INSUFFICIENT_DATA_FOR_PROFILING")
else:
    st.info("Awaiting secure stream from Terminal...")
