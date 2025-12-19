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

# High-Visibility MNC CSS (Fixed Metric Colors)
st.markdown("""
<style>
    .main { background-color: #0A0C10; }
    [data-testid="stMetric"] {
        background-color: #161B22 !important;
        border: 1px solid #30363D !important;
        padding: 15px !important;
        border-radius: 5px !important;
    }
    [data-testid="stMetricLabel"] { color: #8B949E !important; font-family: 'Roboto Mono'; font-size: 12px !important; }
    [data-testid="stMetricValue"] { color: #58A6FF !important; font-family: 'Roboto Mono'; font-weight: bold !important; }
    .stTabs [data-baseweb="tab"] { color: #8B949E; border-bottom: 2px solid transparent; }
    .stTabs [aria-selected="true"] { color: #58A6FF !important; border-bottom: 2px solid #58A6FF !important; }
</style>
""", unsafe_allow_html=True)

def purify_headers(columns):
    return [re.sub(r'[^\w]', '_', col).strip('_') for col in columns]

# --- 2. EXECUTIVE SIDEBAR ---
with st.sidebar:
    st.title("ENGINE_CORE")
    st.caption("V5.0.1-GOLD_STANDARD")
    uploaded_file = st.file_uploader("UPLOAD CONNECTED_SANITIZED_STREAM.CSV", type="csv")
    
    if uploaded_file:
        st.divider()
        st.subheader("PIPELINE_CONTROLS")
        clean_headers = st.toggle("PURIFY_HEADERS", value=True)
        impute_strategy = st.selectbox("IMPUTATION_LOGIC", ["Mean", "Median", "Zero Fill"])
        scaling = st.toggle("NORMALIZE_VECTORS", value=True)
        
        st.divider()
        # Export feature
        st.subheader("EXPORT_INTERFACE")

# --- 3. DATA ARCHITECTURE ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if clean_headers:
        df.columns = purify_headers(df.columns)

    # Specific fix for Penguin dataset artifacts
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].replace('.', np.nan)

    # Detect Dimensions
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Imputation Engine
    if num_cols:
        if impute_strategy == "Mean": df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        elif impute_strategy == "Median": df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        else: df[num_cols] = df[num_cols].fillna(0)

    # Refined Download Button
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    with st.sidebar:
        st.download_button("DOWNLOAD_REFINED_DATA", csv_buf.getvalue(), "refined_output.csv", "text/csv")

    # --- 4. DASHBOARD LAYOUT ---
    st.title("ANALYTIC_DASHBOARD")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RECORDS", f"{len(df)}")
    m2.metric("NUMERIC", f"{len(num_cols)}")
    m3.metric("CATEGORIES", f"{len(cat_cols)}")
    m4.metric("STATUS", "SYNC_ACTIVE")

    tab_data, tab_corr, tab_pca, tab_bio = st.tabs(["DATA_HEALTH", "CORRELATION", "PROJECTION", "BIO_PROFILING"])

    with tab_data:
        st.subheader("STRUCTURED_DATA_VIEW")
        st.dataframe(df, use_container_width=True)

    with tab_corr:
        st.subheader("RELATIONSHIP_MATRIX")
        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#0A0C10')
            sns.heatmap(df[num_cols].corr(), annot=True, cmap='RdBu', center=0, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("INSUFFICIENT_NUMERIC_DATA_FOR_MATRIX")

    with tab_pca:
        st.subheader("DIMENSIONAL_PROJECTION")
        if len(num_cols) >= 2:
            pca_data = df[num_cols].copy()
            if scaling: pca_data = StandardScaler().fit_transform(pca_data)
            
            pca = PCA(n_components=2)
            coords = pca.fit_transform(pca_data)
            pca_df = pd.DataFrame(coords, columns=['PC1', 'PC2'], index=df.index)
            
            hue_opt = ["None"] + cat_cols
            hue_sel = st.selectbox("COLOR_ENCODING:", hue_opt, key="pca_hue")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0A0C10')
            ax.set_facecolor('#0A0C10')
            
            if hue_sel != "None":
                sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=df[hue_sel], palette="viridis", ax=ax)
            else:
                sns.scatterplot(data=pca_df, x='PC1', y='PC2', color='#58A6FF', ax=ax)
            st.pyplot(fig)
        else:
            st.error("PROJECTION_UNAVAILABLE: DATASET_REQUIRES_2_NUMERIC_COLUMNS")

    with tab_bio:
        st.subheader("DISTRIBUTION_ANALYSIS")
        if num_cols and cat_cols:
            x_ax = st.selectbox("CATEGORICAL_AXIS:", cat_cols, key="bio_x")
            y_ax = st.selectbox("MEASUREMENT_AXIS:", num_cols, key="bio_y")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('#0A0C10')
            sns.violinplot(data=df, x=x_ax, y=y_ax, palette="muted", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("BIO_PROFILING_UNAVAILABLE: NO_CATEGORICAL_LABELS_DETECTED")
            st.info("Check Terminal settings to ensure labels are not being stripped during export.")
else:
    st.info("Awaiting high-integrity stream from Data Terminal...")
