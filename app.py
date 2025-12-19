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

# MCN-GRADE UI OVERRIDE (Fixes the Black-on-Black issue)
st.markdown("""
<style>
    .main { background-color: #0A0C10; }
    
    /* FORCE METRIC VISIBILITY */
    [data-testid="stMetric"] {
        background-color: #161B22 !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
        padding: 15px !important;
    }
    [data-testid="stMetricLabel"] {
        color: #FFFFFF !important; /* Force Label to White */
        font-family: 'Roboto Mono' !important;
        font-size: 14px !important;
        opacity: 0.8;
    }
    [data-testid="stMetricValue"] {
        color: #58A6FF !important; /* Force Value to Bright Blue */
        font-family: 'Roboto Mono' !important;
        font-weight: bold !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab"] { color: #8B949E; }
    .stTabs [aria-selected="true"] { color: #58A6FF !important; border-bottom-color: #58A6FF !important; }
</style>
""", unsafe_allow_html=True)

def purify_headers(columns):
    return [re.sub(r'[^\w]', '_', col).strip('_') for col in columns]

# --- 2. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("ENGINE_CORE_v5.1")
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

    # Detect Features
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Apply Imputation
    if num_cols:
        if impute_strategy == "Mean": df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        elif impute_strategy == "Median": df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        else: df[num_cols] = df[num_cols].fillna(0)

    # --- 4. DASHBOARD ---
    st.title("ANALYTIC_DASHBOARD")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RECORDS", len(df))
    m2.metric("NUMERIC", len(num_cols))
    m3.metric("CATEGORIES", len(cat_cols))
    m4.metric("INTEGRITY", "STABLE")

    tab_data, tab_corr, tab_pca, tab_bio = st.tabs(["DATA_VIEW", "CORRELATION", "PROJECTION", "BIO_PROFILING"])

    with tab_data:
        st.dataframe(df, use_container_width=True)

    with tab_corr:
        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#0A0C10')
            sns.heatmap(df[num_cols].corr(), annot=True, cmap='RdBu', center=0, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Insufficient numeric data for correlation.")

    with tab_pca:
        st.subheader("DIMENSIONAL_PROJECTION")
        if len(num_cols) >= 2:
            st.info("Select parameters and trigger the compute engine.")
            col1, col2 = st.columns(2)
            with col1:
                hue_sel = st.selectbox("COLOR_DIMENSION", ["None"] + cat_cols)
            with col2:
                # NEW FEATURE: Logic Gate Button
                run_pca = st.button("GENERATE_PROJECTION_MAP")

            if run_pca:
                pca_data = df[num_cols].copy()
                if scaling: pca_data = StandardScaler().fit_transform(pca_data)
                
                pca = PCA(n_components=2)
                coords = pca.fit_transform(pca_data)
                pca_df = pd.DataFrame(coords, columns=['PC1', 'PC2'])
                
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_facecolor('#0A0C10')
                ax.set_facecolor('#0A0C10')
                
                if hue_sel != "None":
                    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=df[hue_sel].values, palette="viridis", ax=ax)
                else:
                    sns.scatterplot(data=pca_df, x='PC1', y='PC2', color='#58A6FF', ax=ax)
                
                ax.spines['bottom'].set_color('#30363D')
                ax.spines['left'].set_color('#30363D')
                ax.tick_params(colors='#8B949E')
                st.pyplot(fig)
            else:
                st.write("---")
                st.caption("Awaiting Manual Trigger: Click 'GENERATE_PROJECTION_MAP' to begin calculation.")
        else:
            st.error("Error: PCA requires at least 2 numeric columns.")

    with tab_bio:
        if cat_cols and num_cols:
            x_ax = st.selectbox("LABEL_AXIS", cat_cols)
            y_ax = st.selectbox("VALUE_AXIS", num_cols)
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#0A0C10')
            sns.violinplot(data=df, x=x_ax, y=y_ax, palette="muted", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Bio-Profiling requires both Categorical and Numeric data.")

else:
    st.info("System Standby: Please upload a sanitized data stream.")
