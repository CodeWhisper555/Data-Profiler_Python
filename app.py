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

# VIBRANT ANTI-LAYER CSS OVERRIDE (AGGRESSIVE RESET)
st.markdown("""
<style>
    /* 1. UNIVERSAL TRANSPARENCY RESET */
    /* This targets the white layers that sit on top of the background */
    [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"], 
    [data-testid="stMain"], 
    [data-testid="stAppViewBlockContainer"],
    .main, .block-container, .st-emotion-cache-10trblm, .st-emotion-cache-6qob1r {
        background-color: transparent !important;
        background: transparent !important;
    }

    /* 2. THE BASE GRADIENT LAYER */
    .stApp {
        background: linear-gradient(135deg, #020205 0%, #050a24 50%, #12061d 100%) !important;
        background-attachment: fixed !important;
    }

    /* 3. NEON GLASSMORPHISM CARDS */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(0, 242, 255, 0.4) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        backdrop-filter: blur(15px) !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5) !important;
    }

    /* 4. TEXT & UI VIBRANCY */
    h1, h2, h3, p, span, label, .stMarkdown {
        color: #FFFFFF !important;
        text-shadow: 0px 0px 8px rgba(0, 242, 255, 0.3);
    }

    [data-testid="stMetricLabel"] { 
        color: #00F2FF !important; 
        font-family: 'Roboto Mono' !important;
        text-transform: uppercase;
        font-size: 0.8rem !important;
    }
    
    [data-testid="stMetricValue"] { 
        color: #FF00E5 !important; 
        font-family: 'Roboto Mono' !important;
        text-shadow: 0px 0px 12px rgba(255, 0, 229, 0.6) !important;
    }

    /* 5. TAB STYLING */
    .stTabs [data-baseweb="tab-list"] { background-color: transparent !important; }
    .stTabs [data-baseweb="tab"] { color: rgba(255,255,255,0.5) !important; }
    .stTabs [aria-selected="true"] { 
        color: #00F2FF !important; 
        border-bottom: 2px solid #00F2FF !important;
    }

    /* 6. NEON BUTTONS */
    .stButton>button {
        background: linear-gradient(90deg, #00F2FF, #FF00E5) !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        font-weight: 800 !important;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 5px 15px rgba(0, 242, 255, 0.5);
    }
    
    /* 7. DATAFRAME VISIBILITY */
    .stDataFrame {
        border: 1px solid rgba(0, 242, 255, 0.2) !important;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

def purify_headers(columns):
    return [re.sub(r'[^\w]', '_', col).strip('_') for col in columns]

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# --- 2. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("NEON_CORE_v5.5")
    uploaded_file = st.file_uploader("INGEST DATA STREAM", type="csv")
    
    if uploaded_file:
        st.divider()
        # EXPORT BUTTON AT THE TOP OF SIDEBAR
        st.subheader("TERMINAL_EXPORT")
        # Logic is handled below after data processing
        
        st.divider()
        st.subheader("ENGINE_CONFIG")
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

    # Apply Imputation
    if num_cols:
        if impute_strategy == "Mean": df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        elif impute_strategy == "Median": df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        else: df[num_cols] = df[num_cols].fillna(0)

    # SIDEBAR DOWNLOAD BUTTON (Active state)
    with st.sidebar:
        csv_ready = convert_df(df)
        st.download_button(
            label="⚡ DOWNLOAD PROCESSED STREAM",
            data=csv_ready,
            file_name="neural_processed_output.csv",
            mime="text/csv",
        )

    # --- 4. MAIN DASHBOARD ---
    st.title("VIBRANT_ANALYTIC_HUB")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RECORDS", len(df))
    m2.metric("NUMERIC", len(num_cols))
    m3.metric("CATEGORIES", len(cat_cols))
    m4.metric("SYSTEM", "STABLE")

    tab_data, tab_corr, tab_pca, tab_bio = st.tabs(["DATA_VIEW", "CORRELATION", "PROJECTION", "BIO_PROFILING"])

    with tab_data:
        st.dataframe(df, use_container_width=True)

    with tab_corr:
        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#020205')
            ax.set_facecolor('#020205')
            corr = df[num_cols].corr()
            sns.heatmap(corr, annot=True, cmap='magma', ax=ax, 
                        annot_kws={"color": "white", "size": 9})
            plt.xticks(color='#00F2FF', weight='bold')
            plt.yticks(color='#00F2FF', weight='bold')
            st.pyplot(fig)
        else:
            st.warning("Insufficient numeric data for correlation.")

    with tab_pca:
        st.subheader("DIMENSIONAL_PROJECTION")
        if len(num_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                hue_sel = st.selectbox("COLOR_DIMENSION", ["None"] + cat_cols)
            with col2:
                run_pca = st.button("EXECUTE NEON PCA")

            if run_pca:
                pca_data = df[num_cols].copy()
                if scaling: pca_data = StandardScaler().fit_transform(pca_data)
                
                pca = PCA(n_components=2)
                coords = pca.fit_transform(pca_data)
                pca_df = pd.DataFrame(coords, columns=['PC1', 'PC2'])
                
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_facecolor('#020205')
                ax.set_facecolor('#020205')
                
                if hue_sel != "None":
                    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=df[hue_sel].values, palette="husl", ax=ax, s=100)
                else:
                    sns.scatterplot(data=pca_df, x='PC1', y='PC2', color='#00F2FF', ax=ax, s=100)
                
                ax.tick_params(colors='#FFFFFF')
                ax.xaxis.label.set_color('#00F2FF')
                ax.yaxis.label.set_color('#00F2FF')
                st.pyplot(fig)

    with tab_bio:
        if cat_cols and num_cols:
            x_ax = st.selectbox("LABEL_AXIS", cat_cols)
            y_ax = st.selectbox("VALUE_AXIS", num_cols)
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#020205')
            ax.set_facecolor('#020205')
            sns.violinplot(data=df, x=x_ax, y=y_ax, palette="cool", ax=ax)
            plt.xticks(color='#00F2FF', rotation=45)
            plt.yticks(color='#00F2FF')
            st.pyplot(fig)
else:
    st.info("System Ready: Upload a CSV to initialize the Neon Grid.")
