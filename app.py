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

# Enhanced CSS for Visibility (Fixed the "Black on Black" issue)
st.markdown("""
<style>
    .main { background-color: #0A0C10; }
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #161B22 !important;
        border: 1px solid #30363D !important;
        padding: 20px !important;
        border-radius: 8px !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #8B949E !important; /* Muted Grey Label */
        font-family: 'Roboto Mono' !important;
        letter-spacing: 1px !important;
    }
    div[data-testid="stMetricValue"] {
        color: #58A6FF !important; /* Bright Electric Blue Value */
        font-family: 'Roboto Mono' !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        color: #8B949E;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

def purify_headers(columns):
    return [re.sub(r'[^\w]', '_', col).strip('_') for col in columns]

# --- 2. EXECUTIVE SIDEBAR ---
with st.sidebar:
    st.title("ENGINE_CORE")
    st.caption("VERSION 4.2.4-STABLE")
    uploaded_file = st.file_uploader("UPLOAD DATA STREAM (.CSV)", type="csv")
    
    if uploaded_file:
        st.divider()
        st.subheader("PIPELINE_CONTROLS")
        clean_headers = st.toggle("PURIFY_HEADERS", value=True)
        impute_strategy = st.selectbox("IMPUTATION_LOGIC", ["Mean", "Median", "Zero Fill"])
        scaling = st.toggle("NORMALIZE_VECTORS", value=True)

# --- 3. DATA ARCHITECTURE ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if clean_headers:
        df.columns = purify_headers(df.columns)

    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].replace('.', np.nan)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if num_cols:
        if impute_strategy == "Mean":
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        elif impute_strategy == "Median":
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        else:
            df[num_cols] = df[num_cols].fillna(0)

    # --- 4. DASHBOARD LAYOUT ---
    st.title("ANALYTIC_DASHBOARD")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RECORDS", len(df))
    m2.metric("NUMERIC", len(num_cols))
    m3.metric("CATEGORIES", len(cat_cols))
    m4.metric("STATUS", "SYNC_COMPLETE")

    tab_data, tab_corr, tab_pca, tab_bio = st.tabs(["DATA_HEALTH", "CORRELATION", "PROJECTION", "BIO_PROFILING"])

    with tab_data:
        st.subheader("STRUCTURED_DATA_VIEW")
        st.dataframe(df, use_container_width=True)

    with tab_corr:
        st.subheader("RELATIONSHIP_MATRIX")
        if len(num_cols) > 1:
            fig_corr, ax_c = plt.subplots(figsize=(8, 5))
            fig_corr.patch.set_facecolor('#0A0C10')
            sns.heatmap(df[num_cols].corr(), annot=True, cmap='RdBu', center=0, ax=ax_c, annot_kws={"size": 8})
            st.pyplot(fig_corr)
        else:
            st.warning("INSUFFICIENT_NUMERIC_VECTORS")

    with tab_pca:
        st.subheader("DIMENSIONAL_PROJECTION")
        if len(num_cols) >= 2:
            pca_data = df[num_cols].copy()
            if scaling:
                pca_data = StandardScaler().fit_transform(pca_data)
            
            pca = PCA(n_components=2)
            components = pca.fit_transform(pca_data)
            pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'], index=df.index)
            
            # Use "None" if no categories exist
            options = ["None"] + cat_cols
            hue_col = st.selectbox("COLOR_DIMENSION:", options, key="pca_hue")
            
            fig_pca, ax_p = plt.subplots(figsize=(10, 6))
            fig_pca.patch.set_facecolor('#0A0C10')
            ax_p.set_facecolor('#0A0C10')
            
            if hue_col != "None" and hue_col in df.columns:
                sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=df[hue_col], ax=ax_p, palette="viridis")
            else:
                sns.scatterplot(data=pca_df, x='PC1', y='PC2', ax=ax_p, color='#007BFF')
            
            st.pyplot(fig_pca)
        else:
            st.error("PROJECTION_UNAVAILABLE: REQUIRE 2+ NUMERIC COLUMNS")

    with tab_bio:
        st.subheader("DISTRIBUTION_ANALYSIS")
        if num_cols and cat_cols:
            x_ax = st.selectbox("CATEGORICAL_AXIS:", cat_cols, key="bio_x")
            y_ax = st.selectbox("MEASUREMENT_AXIS:", num_cols, key="bio_y")
            
            fig_bio, ax_b = plt.subplots(figsize=(10, 4))
            fig_bio.patch.set_facecolor('#0A0C10')
            sns.violinplot(data=df, x=x_ax, y=y_ax, ax=ax_b, palette="muted")
            st.pyplot(fig_bio)
        else:
            # Display clearly that profiling is impossible without categories
            st.warning("BIO_PROFILING_UNAVAILABLE: NO_CATEGORICAL_DATA_DETECTED")
            st.info("Ingest a dataset containing labels (e.g., Species, Sex) to enable profiling.")
else:
    st.info("Awaiting secure stream from Terminal...")
