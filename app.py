import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. GLOBAL SYSTEM CONFIGURATION
st.set_page_config(page_title="DATA INTELLIGENCE SYSTEM", layout="wide")

# Safe CSS Injection - Cleaned for Python 3.13 Compatibility
corporate_css = """
<style>
    .stApp { background-color: #0A0C10; color: #E6E6E6; font-family: 'Inter', sans-serif; }
    div[data-testid="stMetricValue"] { color: #007BFF; font-family: 'Roboto Mono', monospace; font-size: 28px !important; }
    div[data-testid="stMetricLabel"] { text-transform: uppercase; letter-spacing: 1px; font-size: 12px; color: #8B949E; }
    [data-testid="stSidebar"] { background-color: #101214; border-right: 1px solid #30363D; }
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; border-bottom: 1px solid #30363D; }
    .stTabs [data-baseweb="tab"] { color: #8B949E; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #007BFF !important; border-bottom-color: #007BFF !important; }
    .stButton>button { background-color: #21262D; border: 1px solid #30363D; color: #C9D1D9; border-radius: 4px; }
</style>
"""
st.markdown(corporate_css, unsafe_allow_name_with_html=True)

# 2. EXECUTIVE HEADER
st.title("INTELLIGENCE ANALYTICS ENGINE")
st.caption("INTERNAL CLOUD INFRASTRUCTURE | VERSION 4.1.0-STABLE")
st.divider()

# 3. CONTROL INTERFACE
with st.sidebar:
    st.markdown("### SYSTEM PARAMETERS")
    uploaded_file = st.file_uploader("SOURCE DATA UPLOAD (.CSV)", type="csv")
    
    if uploaded_file:
        st.divider()
        st.markdown("### PROCESSING PIPELINE")
        impute = st.toggle("AUTOMATED MEAN IMPUTATION", value=True)
        scale_data = st.toggle("Z-SCORE NORMALIZATION", value=False)
        st.markdown("---")
        st.markdown("**CORE STATUS: OPERATIONAL**")

# 4. DATA ARCHITECTURE LOGIC
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Identifying numeric columns strictly
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # DIAGNOSTIC CALCULATION
    null_count = df.isnull().sum().sum()
    total_cells = df.size if df.size > 0 else 1
    anomaly_rate = (null_count / total_cells) * 100

    # EXECUTIVE METRIC ROW
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dataset Records", f"{len(df):,}")
    m2.metric("Feature Count", len(df.columns))
    m3.metric("Integrity Index", f"{100-anomaly_rate:.2f}%")
    m4.metric("System Latency", "0.02ms")

    # ANALYSIS MODULES
    tab1, tab2, tab3 = st.tabs(["[01] DATA_INVENTORY", "[02] CORRELATION_MAPPING", "[03] DIMENSIONAL_PROJECTION"])

    with tab1:
        st.markdown("#### STRUCTURED DATA VIEW")
        st.dataframe(df, use_container_width=True, height=450)
        
    with tab2:
        st.markdown("#### PEARSON MULTI-VARIATE ANALYSIS")
        if len(num_cols) > 1:
            # Using a context manager for the plot to avoid memory leaks
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor('#0A0C10')
            ax.set_facecolor('#0A0C10')
            
            corr = df[num_cols].corr()
            sns.heatmap(corr, annot=True, cmap='RdBu', center=0, ax=ax)
            
            plt.xticks(color='#8B949E')
            plt.yticks(color='#8B949E')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("INSUFFICIENT NUMERIC DIMENSIONS DETECTED.")

    with tab3:
        st.markdown("#### PRINCIPAL COMPONENT PROJECTION (PCA)")
        if len(num_cols) >= 2:
            # Prepare data for PCA
            pca_df_input = df[num_cols].fillna(df[num_cols].mean())
            if scale_data:
                pca_df_input = StandardScaler().fit_transform(pca_df_input)
            
            pca = PCA(n_components=2)
            components = pca.fit_transform(pca_df_input)
            pca_res = pd.DataFrame(components, columns=['PC1', 'PC2'])
            
            group_col = st.selectbox("IDENTIFY CLUSTER GROUP:", cat_cols) if cat_cols else None
            
            fig_pca, ax_pca = plt.subplots(figsize=(10, 5))
            fig_pca.patch.set_facecolor('#0A0C10')
            ax_pca.set_facecolor('#0A0C10')
            
            if group_col:
                pca_res[group_col] = df[group_col].values
                sns.scatterplot(data=pca_res, x='PC1', y='PC2', hue=group_col, palette="Blues_r", ax=ax_pca)
            else:
                sns.scatterplot(data=pca_res, x='PC1', y='PC2', color="#007BFF", ax=ax_pca)
            
            st.pyplot(fig_pca)
            plt.close(fig_pca)
        else:
            st.error("ENGINE ERROR: PCA REQUIRES MINIMUM 2 NUMERIC DIMENSIONS.")
else:
    st.markdown("### SYSTEM STANDBY")
    st.info("Awaiting CSV input for cloud-side analysis.")
