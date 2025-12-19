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
    return [re.sub(r'[^\w]', '_', col).strip('_') for col in columns]

# --- 2. EXECUTIVE SIDEBAR ---
with st.sidebar:
    st.title("ENGINE_CORE")
    st.caption("VERSION 4.2.1-STABLE")
    
    uploaded_file = st.file_uploader("UPLOAD SANITIZED STREAM (.CSV)", type="csv")
    
    if uploaded_file:
        st.divider()
        st.subheader("PIPELINE_CONTROLS")
        clean_headers = st.toggle("PURIFY_HEADERS", value=True)
        impute_strategy = st.selectbox("IMPUTATION_LOGIC", ["Mean", "Median", "Zero Fill"])
        scaling = st.toggle("NORMALIZE_VECTORS", value=True)
        
        st.divider()
        st.subheader("EXPORT_INTERFACE")
        # The Download Feature is placed here for easy access after configuration
        st.caption("DOWNLOAD THE REFINED DATASET AFTER IMPUTATION")
        
# --- 3. DATA ARCHITECTURE & PROCESSING ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if clean_headers:
        df.columns = purify_headers(df.columns)

    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].replace('.', np.nan)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Apply Imputation Logic
    if impute_strategy == "Mean":
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    elif impute_strategy == "Median":
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    else:
        df[num_cols] = df[num_cols].fillna(0)

    # --- NEW: DOWNLOAD LOGIC ---
    # Convert dataframe to CSV in memory
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    
    with st.sidebar:
        st.download_button(
            label="DOWNLOAD_REFINED_DATA",
            data=csv_buffer.getvalue(),
            file_name="refined_neural_stream.csv",
            mime="text/csv"
        )
        st.success("STATUS: READY_FOR_EXPORT")

    # --- 4. DASHBOARD LAYOUT ---
    st.title("ANALYTIC_DASHBOARD")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RECORDS", len(df))
    m2.metric("NUMERIC", len(num_cols))
    m3.metric("CATEGORIES", len(cat_cols))
    m4.metric("CLEAN_INDEX", "100%" if df.isnull().sum().sum() == 0 else "PENDING")

    tab_data, tab_corr, tab_pca, tab_bio = st.tabs([
        "DATA_HEALTH", "CORRELATION", "PROJECTION", "BIO_PROFILING"
    ])

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

    with tab_pca:
        st.subheader("DIMENSIONAL_PROJECTION")
        if len(num_cols) >= 2:
            pca_data = df[num_cols]
            if scaling:
                pca_data = StandardScaler().fit_transform(pca_data)
            
            pca = PCA(n_components=2)
            components = pca.fit_transform(pca_data)
            pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
            
            hue_col = st.selectbox("COLOR_DIMENSION:", cat_cols)
            fig_pca, ax_p = plt.subplots(figsize=(10, 6))
            fig_pca.patch.set_facecolor('#0A0C10')
            sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=df[hue_col], ax=ax_p)
            st.pyplot(fig_pca)

    with tab_bio:
        st.subheader("DISTRIBUTION_ANALYSIS")
        if num_cols and cat_cols:
            feat = st.selectbox("SELECT_MEASUREMENT:", num_cols)
            fig_bio, ax_b = plt.subplots(figsize=(10, 4))
            fig_bio.patch.set_facecolor('#0A0C10')
            sns.violinplot(data=df, x=cat_cols[0], y=feat, ax=ax_b)
            st.pyplot(fig_bio)
else:
    st.info("Awaiting secure stream from Terminal...")
