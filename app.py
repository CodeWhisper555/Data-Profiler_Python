import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- 1. CORE UTILITIES ---
def purify_headers(columns):
    """Standardizes column names by removing special characters."""
    return [re.sub(r'[^\w]', '_', col).strip('_') for col in columns]

@st.cache_data
def convert_df(df):
    """Encodes the processed dataframe for the download function."""
    return df.to_csv(index=False).encode('utf-8')

# --- 2. APP SETUP ---
st.set_page_config(page_title="Data Logic Node", layout="wide")
st.title("Operational Analytics Terminal")

# --- 3. SIDEBAR: INGESTION & GLOBAL LOGIC ---
with st.sidebar:
    st.header("1. Data Ingress")
    uploaded_file = st.file_uploader("Upload CSV Stream", type="csv")
    
    if uploaded_file:
        st.divider()
        st.header("2. Logic Configuration")
        clean_headers = st.toggle("Purify Headers", value=True)
        scaling = st.toggle("Normalize Vectors (PCA)", value=True)
        impute_strategy = st.selectbox("Imputation Engine", ["Mean", "Median", "Zero Fill"])

# --- 4. PROCESSING ENGINE ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if clean_headers:
        df.columns = purify_headers(df.columns)

    # Logic: Separate Numeric and Categorical Features
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Logic: Imputation Execution
    if num_cols:
        if impute_strategy == "Mean": 
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        elif impute_strategy == "Median": 
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        else: 
            df[num_cols] = df[num_cols].fillna(0)

    # SIDEBAR: Terminal Export
    with st.sidebar:
        st.divider()
        st.header("3. Terminal Export")
        csv_ready = convert_df(df)
        st.download_button(
            label="Download Processed CSV",
            data=csv_ready,
            file_name="processed_data_output.csv",
            mime="text/csv",
        )

    # --- 5. DASHBOARD UI ---
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Records", len(df))
    col_b.metric("Numeric Features", len(num_cols))
    col_c.metric("Categorical Features", len(cat_cols))

    tabs = st.tabs(["Inspector", "Correlation", "PCA Projection", "Bio-Distribution"])

    with tabs[0]:
        st.dataframe(df, use_container_width=True)

    with tabs[1]:
        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap='viridis', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Logic Error: Correlation requires >1 numeric column.")

    with tabs[2]:
        if len(num_cols) >= 2:
            c1, c2 = st.columns(2)
            hue_col = c1.selectbox("Color By", ["None"] + cat_cols)
            do_pca = c2.button("Execute PCA Analysis")

            if do_pca:
                x = df[num_cols].values
                if scaling:
                    x = StandardScaler().fit_transform(x)
                
                pca_engine = PCA(n_components=2)
                components = pca_engine.fit_transform(x)
                pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
                
                fig, ax = plt.subplots()
                sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=(df[hue_col] if hue_col != "None" else None), ax=ax)
                st.pyplot(fig)
        else:
            st.error("Logic Error: PCA requires minimum 2 numeric features.")

    with tabs[3]:
        # FIXED: Persistent message logic for Bio-Distribution
        if cat_cols and num_cols:
            sel_cat = st.selectbox("Categorical Axis", cat_cols)
            sel_num = st.selectbox("Numerical Axis", num_cols)
            
            # Check if there is data to plot
            if not df[sel_num].dropna().empty:
                fig, ax = plt.subplots()
                sns.violinplot(data=df, x=sel_cat, y=sel_num, ax=ax)
                st.pyplot(fig)
            else:
                st.info("No distribution detected for the selected feature.")
        else:
            # Persistent message when requirements aren't met
            st.warning("Bio-Profiling Standby: This analysis requires at least one Categorical column and one Numeric column to generate a distribution.")

else:
    st.info("System Standby. Please upload a dataset to begin logic execution.")
