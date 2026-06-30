import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from scipy.stats import zscore

st.set_page_config(
    page_title="AI Business Intelligence",
    layout="wide"
)

st.title("🤖 AI Business Intelligence Smart Dashboard")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel",
    type=["csv","xlsx"]
)

if uploaded_file:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_excel(uploaded_file)

    st.success("Dataset Loaded")

    st.subheader("Dataset Preview")

    st.dataframe(df)

    ####################################
    # DATA CLEANING
    ####################################

    st.header("🧹 AI Preprocessing")

    duplicate_count = df.duplicated().sum()

    st.write("Duplicates:", duplicate_count)

    df.drop_duplicates(inplace=True)

    numeric = df.select_dtypes(include=np.number).columns

    categorical = df.select_dtypes(exclude=np.number).columns

    if len(numeric)>0:

        imputer = SimpleImputer(strategy="median")

        df[numeric]=imputer.fit_transform(df[numeric])

    if len(categorical)>0:

        imputer2 = SimpleImputer(strategy="most_frequent")

        df[categorical]=imputer2.fit_transform(df[categorical])

    st.success("Missing Values Filled")

    ####################################
    # OUTLIERS
    ####################################

    st.header("📌 Outlier Detection")

    if len(numeric)>0:

        z=np.abs(zscore(df[numeric]))

        outliers=(z>3).sum()

        st.write(outliers)

    ####################################
    # DATA SUMMARY
    ####################################

    st.header("📊 Dataset Summary")

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Rows",len(df))

    c2.metric("Columns",len(df.columns))

    c3.metric("Numeric",len(numeric))

    c4.metric("Categorical",len(categorical))

    st.write(df.describe())

    ####################################
    # CORRELATION
    ####################################

    if len(numeric)>=2:

        st.header("Correlation Heatmap")

        corr=df[numeric].corr()

        fig=px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu"
        )

        st.plotly_chart(fig,use_container_width=True)

    ####################################
    # BAR CHART
    ####################################

    st.header("Interactive Graph")

    col1,col2=st.columns(2)

    x=col1.selectbox("X",df.columns)

    y=col2.selectbox("Y",numeric)

    chart=st.selectbox(
        "Chart",
        [
            "Bar",
            "Line",
            "Scatter",
            "Histogram",
            "Box"
        ]
    )

    if chart=="Bar":

        fig=px.bar(df,x=x,y=y)

    elif chart=="Line":

        fig=px.line(df,x=x,y=y)

    elif chart=="Scatter":

        fig=px.scatter(df,x=x,y=y)

    elif chart=="Histogram":

        fig=px.histogram(df,x=x)

    else:

        fig=px.box(df,x=x,y=y)

    st.plotly_chart(fig,use_container_width=True)

    ####################################
    # PIE
    ####################################

    if len(categorical)>0:

        st.header("Pie Chart")

        pie=st.selectbox("Category",categorical)

        fig=px.pie(
            df,
            names=pie
        )

        st.plotly_chart(fig,use_container_width=True)

    ####################################
    # 3D GRAPH
    ####################################

    if len(numeric)>=3:

        st.header("3D Scatter")

        x3=st.selectbox("3D X",numeric,key=1)

        y3=st.selectbox("3D Y",numeric,key=2)

        z3=st.selectbox("3D Z",numeric,key=3)

        color=st.selectbox(
            "Color",
            df.columns
        )

        fig=px.scatter_3d(
            df,
            x=x3,
            y=y3,
            z=z3,
            color=color
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    ####################################
    # AI INSIGHTS
    ####################################

    st.header("🧠 AI Business Insights")

    st.success(f"""
Dataset contains **{len(df)} records**.

• Numeric Features : {len(numeric)}

• Categorical Features : {len(categorical)}

• Missing Values Automatically Filled

• Duplicates Removed

• Ready for Machine Learning

• Correlation Matrix Generated

• Interactive Dashboard Created

Recommendation:

✔ Clean Dataset

✔ Check highly correlated features

✔ Remove unnecessary columns

✔ Perform prediction or clustering
""")

    ####################################
    # DOWNLOAD
    ####################################

    st.download_button(
        "Download Clean Dataset",
        df.to_csv(index=False),
        "clean_dataset.csv"
    )
