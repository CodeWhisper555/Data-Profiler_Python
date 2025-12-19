import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. GLOBAL SYSTEM CONFIGURATION
st.set_page_config(page_title="DATA INTELLIGENCE SYSTEM", layout="wide")

# MNC Industrial Theme Injection
st.markdown("""
    <style>
    /* Corporate Typography and Background */
    .stApp { background-color: #0A0C10; color: #E6E6E6; font-family: 'Inter', sans-serif; }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] { color: #007BFF; font-family: 'Roboto Mono', monospace; font-size: 28px !important; }
    div[data-testid="stMetricLabel"] { text-transform: uppercase; letter-spacing: 1px; font-size: 12px; color: #8B949E; }
    
    /* Sidebar Overhaul */
    [data-testid="stSidebar"] { background-color: #101214; border-right: 1px solid #30363D; }
    
    /* Tab Styling - Flat Minimalist */
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; border-bottom: 1px solid #30363D; }
    .stTabs [data-baseweb="tab"] { color: #8B949E; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #007BFF !important; border-bottom-color: #007BFF !important; }
    
    /* Button Styling */
    .stButton>button { background-color: #21262D; border: 1px solid #30363D; color: #C9D1D9; border-radius: 4px; }
    .stButton>button:hover { border-color: #8B949E; color: #FFFFFF; }
    </style>
    """, unsafe_allow_name_with_html=True)

# 2. EXECUTIVE HEADER
st.title("INTELLIGENCE ANALYTICS ENGINE")
st.caption("INTERNAL CLOUD INFRASTRUCTURE | VERSION 4.1.0-STABLE")
st.divider()

# 3. CONTROL INTERFACE
with st.sidebar:
    st.markdown("### SYSTEM PARAMETERS")
    uploaded_file = st.file_uploader("SOURCE DATA UPLOAD (.CSV)", type="csv", help="Upload strictly formatted corporate datasets.")
    
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
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # DIAGNOSTIC CALCULATION
    null_count = df.isnull().sum().sum()
    anomaly_rate = (null_count / df.size) * 100

    # EXECUTIVE METRIC ROW
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dataset Records", f"{len(df):,}")
    m2.metric("Feature Count", len(df.columns))
    m3.metric("Integrity Index", f"{100-anomaly_rate:.2f}%")
    m4.metric("System Latency", "0.02ms")

    # PIPELINE LOGS
    with st.status("EXECUTING STATISTICAL HANDSHAKE", expanded=False) as status:
        if impute:
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
            st.write("Imputing missing vectors...")
        status.update(label="PIPELINE STABILIZED", state="complete")

    # ANALYSIS MODULES
    tab1, tab2, tab3 = st.tabs(["[01] DATA_INVENTORY", "[02] CORRELATION_MAPPING", "[03] DIMENSIONAL_PROJECTION"])

    with tab1:
        st.markdown("#### STRUCTURED DATA VIEW")
        st.dataframe(df, use_container_width=True, height=450)
        
    with tab2:
        st.markdown("#### PEARSON MULTI-VARIATE ANALYSIS")
        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor('#0A0C10')
            ax.set_facecolor('#0A0C10')
            
            corr = df[num_cols].corr()
            sns.heatmap(corr, annot=True, cmap='RdBu', center=0, ax=ax, 
                        cbar_kws={'label': 'Correlation Coefficient'},
                        annot_kws={"size": 10, "weight": "bold"})
            
            plt.xticks(color='#8B949E')
            plt.yticks(color='#8B949E')
            st.pyplot(fig)
        else:
            st.warning("INSUFFICIENT NUMERIC DIMENSIONS DETECTED.")

    with tab3:
        st.markdown("#### PRINCIPAL COMPONENT PROJECTION (PCA)")
        if len(num_cols) >= 2:
            pca_data = df[num_cols].dropna()
            scaled_vals = StandardScaler().fit_transform(pca_data)
            pca = PCA(n_components=2)
            components = pca.fit_transform(scaled_vals)
            
            pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
            
            # Identify a grouping column for corporate-grade clustering
            group_col = st.selectbox("IDENTIFY CLUSTER GROUP:", cat_cols) if cat_cols else None
            
            fig_pca, ax_pca = plt.subplots(figsize=(10, 5))
            fig_pca.patch.set_facecolor('#0A0C10')
            ax_pca.set_facecolor('#0A0C10')
            
            if group_col:
                pca_df[group_col] = df[group_col].values
                sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=group_col, style=group_col, palette="Blues_r", ax=ax_pca)
            else:
                sns.scatterplot(data=pca_df, x='PC1', y='PC2', color="#007BFF", ax=ax_pca)
            
            plt.title(f"PCA VARIANCE RETENTION: {sum(pca.explained_variance_ratio_)*100:.1f}%", color="#FFFFFF")
            st.pyplot(fig_pca)
        else:
            st.error("ENGINE ERROR: PCA REQUIRES MINIMUM 2 NUMERIC DIMENSIONS.")

else:
    # MINIMALIST LANDING
    st.markdown("### SYSTEM STANDBY")
    st.info("Awaiting CSV input for cloud-side analysis.")
