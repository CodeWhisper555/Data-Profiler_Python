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

# VIBRANT ANTI-LAYER CSS OVERRIDE
st.markdown("""
<style>
    /* FORCE TRANSPARENCY ON ALL POTENTIAL OVERLAYS */
    .stApp, .main, .st-emotion-cache-10trblm, .st-emotion-cache-6qob1r {
        background: linear-gradient(135deg, #050505 0%, #0a0e29 50%, #1a0b2e 100%) !important;
    }
    
    /* REMOVE THE WHITE GHOST LAYER */
    [data-testid="stAppViewContainer"] {
        background-color: transparent !important;
    }
    
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
        color: white !important;
    }

    /* GLASSMORPHISM CARDS */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(0, 242, 255, 0.3) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        backdrop-filter: blur(10px);
    }

    /* TEXT VIBRANCY */
    h1, h2, h3, p, span, label, .stMarkdown {
        color: #FFFFFF !important;
    }

    [data-testid="stMetricLabel"] { color: #00F2FF !important; }
    [data-testid="stMetricValue"] { color: #FF00E5 !important; }

    /* BUTTONS */
    .stButton>button {
        background: linear-gradient(45deg, #00F2FF, #FF00E5) !important;
        color: white !important;
        border-radius: 20px !important;
    }
</style>
""", unsafe_allow_html=True)

def purify_headers(columns):
    return [re.sub(r'[^\w]', '_', col).strip('_') for col in columns]

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("NEON_CORE_v5.4")
    uploaded_file = st.file_uploader("INGEST DATA STREAM", type="csv")
    
    if uploaded_file:
        st.divider()
        clean_headers = st.toggle("PURIFY_HEADERS", value=True)
        scaling = st.toggle("NORMALIZE_VECTORS", value=True)
        impute_strategy = st.selectbox("IMPUTATION", ["Mean", "Median", "Zero Fill"])

# --- 3. DATA ENGINE ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if clean_headers:
        df.columns = purify_headers(df.columns)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Impute
    if num_cols:
        if impute_strategy == "Mean": df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        elif impute_strategy == "Median": df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        else: df[num_cols] = df[num_cols].fillna(0)

    # --- DOWNLOAD FUNCTION ---
    with st.sidebar:
        st.divider()
        csv_ready = convert_df(df)
        st.download_button(
            label="DOWNLOAD_PROCESSED_DATA",
            data=csv_ready,
            file_name="processed_neon_stream.csv",
            mime="text/csv",
        )

    # --- 4. DASHBOARD ---
    st.title("VIBRANT_ANALYTIC_HUB")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RECORDS", len(df))
    m2.metric("NUMERIC", len(num_cols))
    m3.metric("CATEGORIES", len(cat_cols))
    m4.metric("STATUS", "VIBRANT")

    tab_data, tab_corr, tab_pca, tab_bio = st.tabs(["DATA_VIEW", "CORRELATION", "PROJECTION", "BIO_PROFILING"])

    with tab_data:
        st.dataframe(df, use_container_width=True)

    with tab_corr:
        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#0a0e29')
            ax.set_facecolor('#0a0e29')
            sns.heatmap(df[num_cols].corr(), annot=True, cmap='magma', ax=ax)
            plt.xticks(color='#00F2FF')
            plt.yticks(color='#00F2FF')
            st.pyplot(fig)

    with tab_pca:
        if len(num_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                hue_sel = st.selectbox("COLOR_DIMENSION", ["None"] + cat_cols)
            with col2:
                run_pca = st.button("RUN_NEON_PCA")

            if run_pca:
                pca_data = df[num_cols].copy()
                if scaling: pca_data = StandardScaler().fit_transform(pca_data)
                pca = PCA(n_components=2)
                coords = pca.fit_transform(pca_data)
                pca_df = pd.DataFrame(coords, columns=['PC1', 'PC2'])
                
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_facecolor('#0a0e29')
                ax.set_facecolor('#0a0e29')
                sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=(df[hue_sel] if hue_sel != "None" else None), palette="cool", ax=ax)
                ax.tick_params(colors='#00F2FF')
                st.pyplot(fig)

    with tab_bio:
        if cat_cols and num_cols:
            x_ax = st.selectbox("LABEL_AXIS", cat_cols)
            y_ax = st.selectbox("VALUE_AXIS", num_cols)
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#0a0e29')
            ax.set_facecolor('#0a0e29')
            sns.violinplot(data=df, x=x_ax, y=y_ax, palette="husl", ax=ax)
            plt.xticks(color='#00F2FF', rotation=45)
            plt.yticks(color='#00F2FF')
            st.pyplot(fig)
else:
    st.info("System Ready: Please upload a data stream.")
