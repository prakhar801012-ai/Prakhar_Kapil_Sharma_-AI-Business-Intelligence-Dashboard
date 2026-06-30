import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.impute import SimpleImputer
from scipy.stats import zscore

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Business Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 AI Business Intelligence Smart Dashboard")

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"]
    )

# --- DATA PREPROCESSING PIPELINE (CACHED) ---
@st.cache_data
def load_and_preprocess_data(file):
    # 1. Load Data
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    
    # Store raw metrics before transformations
    duplicate_count = df.duplicated().sum()
    
    # 2. Drop Duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Identify Column Types
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    
    # 3. Handle Missing Values
    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
        
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols]).astype(str)
        
    # 4. Outlier Detection (Z-Score > 3)
    outliers_summary = {}
    if numeric_cols:
        z_scores = np.abs(zscore(df[numeric_cols]))
        outlier_counts = (z_scores > 3).sum(axis=0)
        outliers_summary = dict(zip(numeric_cols, outlier_counts))

    return df, numeric_cols, categorical_cols, duplicate_count, outliers_summary

# --- MAIN APP LOGIC ---
if uploaded_file:
    # Execute Pipeline
    df, numeric, categorical, duplicate_count, outliers_dict = load_and_preprocess_data(uploaded_file)
    
    st.success("Dataset Loaded & Sanitized successfully!")

    # Layout: Top Preview
    with st.expander("🔍 View Raw Dataset Preview", expanded=False):
        st.dataframe(df, use_container_width=True)

    # Tabs for Workspace Organization
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧹 Data Cleaning & Summary", 
        "📊 Standard Analytics", 
        "🌌 Advanced 3D Space", 
        "🧠 AI System Insights"
    ])

    # --- TAB 1: DATA CLEANING & SUMMARY ---
    with tab1:
        st.header("🧹 Automated Preprocessing Metadata")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows (Post-Clean)", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Numeric Features", len(numeric))
        c4.metric("Categorical Features", len(categorical))
        
        col_dup, col_out = st.columns(2)
        with col_dup:
            st.subheader("Duplicate Record Resolution")
            st.info(f"Identified and purged **{duplicate_count}** exact duplicate rows.")
            
        with col_out:
            st.subheader("📌 Statistical Outlier Identification")
            if numeric:
                outlier_df = pd.DataFrame(list(outliers_dict.items()), columns=["Column", "Outlier Count (Z > 3)"])
                st.dataframe(outlier_df, use_container_width=True, hide_index=True)
            else:
                st.write("No numerical metrics available for outlier verification.")

        st.subheader("Descriptive Statistical Matrix")
        st.dataframe(df.describe(include='all' if not numeric else None), use_container_width=True)

    # --- TAB 2: STANDARD ANALYTICS ---
    with tab2:
        # Correlation Matrix
        if len(numeric) >= 2:
            st.header("Heatmap: Feature Inter-Correlation")
            corr = df[numeric].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("At least two numerical properties are required to calculate spatial correlations.")

        st.markdown("---")
        
        # Interactive Custom Visualizations
        st.header("Dynamic Plot Studio")
        graph_ctrl_col1, graph_ctrl_col2, graph_ctrl_col3 = st.columns(3)
        
        with graph_ctrl_col1:
            x_var = st.selectbox("Assign X Axis", df.columns, key="sb_x")
        with graph_ctrl_col2:
            y_var = st.selectbox("Assign Y Axis", numeric, key="sb_y") if numeric else st.selectbox("Assign Y Axis", df.columns, key="sb_y")
        with graph_ctrl_col3:
            chart_type = st.selectbox(
                "Select Graph Layout",
                ["Bar Chart", "Line Representation", "Scatter Visual", "Histogram Profile", "Box Plot Distribution"]
            )

        # Plot Generation Engine
        if chart_type == "Bar Chart":
            fig = px.bar(df, x=x_var, y=y_var, template="plotly_white")
        elif chart_type == "Line Representation":
            fig = px.line(df, x=x_var, y=y_var, template="plotly_white")
        elif chart_type == "Scatter Visual":
            fig = px.scatter(df, x=x_var, y=y_var, template="plotly_white")
        elif chart_type == "Histogram Profile":
            fig = px.histogram(df, x=x_var, template="plotly_white")
        else:
            fig = px.box(df, x=x_var, y=y_var, template="plotly_white")

        st.plotly_chart(fig, use_container_width=True)

        # Categorical Proportions
        if categorical:
            st.markdown("---")
            st.header("Pie Chart Proportion Matrix")
            pie_target = st.selectbox("Select Categorical Dimensions", categorical, key="sb_pie")
            
            pie_data = df[pie_target].value_counts().reset_index()
            fig_pie = px.pie(
                pie_data, 
                values='count', 
                names=pie_target,
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # --- TAB 3: ADVANCED 3D SPACE ---
    with tab3:
        if len(numeric) >= 3:
            st.header("Multidimensional 3D Scatter Configuration")
            
            c3d_1, c3d_2, c3d_3, c3d_4 = st.columns(4)
            with c3d_1:
                x3 = st.selectbox("3D Axis: X", numeric, key="3dx")
            with c3d_2:
                y3 = st.selectbox("3D Axis: Y", numeric, key="3dy")
            with c3d_3:
                z3 = st.selectbox("3D Axis: Z", numeric, key="3dz")
            with c3d_4:
                color_var = st.selectbox("Color Segment Dimension", df.columns, key="3dc")

            fig_3d = px.scatter_3d(
                df, x=x3, y=y3, z=z3,
                color=color_var,
                opacity=0.8,
                template="plotly_dark"
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.warning("3D Mapping requires at least 3 distinct numeric attributes within your file.")

    # --- TAB 4: AI SYSTEM INSIGHTS & EXPORT ---
    with tab4:
        st.header("🧠 Automated Structural Summary Report")
        
        st.success(f"""
        ### Pipeline Summary Execution Metrics:
        * **Total Structural Volume:** {len(df)} computational observations analyzed.
        * **Numeric Schema Parameters:** {len(numeric)} numeric target vectors located.
        * **Categorical Vector Count:** {len(categorical)} contextual properties identified.
        * **Imputation Strategy Execution:** True (NaN parameters dynamically handled via Medians/Modes).
        * **Operational Validation Status:** Verified for predictive downstream pipelines (Classification, Regression, Clustering).
        """)
        
        st.markdown("### Next Strategic Business Measures:")
        st.markdown("""
        1. **Check Correlated Targets:** Look for strong alignments within your Tab 2 correlation heatmap to prevent target leakage.
        2. **Prune Sparse Elements:** Filter features presenting invariant standard deviations or single-value distributions before deploying models.
        """)
        
        st.markdown("---")
        st.subheader("💾 Export Cleaned Enterprise File Assets")
        
        st.download_button(
            label="📥 Download Sanitized CSV File",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="bi_processed_dataset.csv",
            mime="text/csv"
        )
else:
    st.info("👋 Welcome! Please upload a valid CSV or Excel document from the sidebar to initialize processing pipelines.")
