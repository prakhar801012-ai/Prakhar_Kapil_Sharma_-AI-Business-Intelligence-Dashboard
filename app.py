import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="AI Business Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Business Intelligence Dashboard")
st.markdown("---")

# Sidebar

st.sidebar.header("Upload Dataset")

file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if file is None:

    st.info("Upload a business dataset to begin.")
    st.stop()

df = pd.read_csv(file)

st.success("Dataset Loaded Successfully")

st.write(df.head())

###########################################################
# KPI
###########################################################

st.subheader("Business KPIs")

col1,col2,col3,col4=st.columns(4)

numeric=df.select_dtypes(include=np.number)

with col1:
    st.metric(
        "Rows",
        len(df)
    )

with col2:
    st.metric(
        "Columns",
        len(df.columns)
    )

with col3:
    st.metric(
        "Missing Values",
        df.isnull().sum().sum()
    )

with col4:
    st.metric(
        "Numeric Features",
        len(numeric.columns)
    )

st.markdown("---")

###########################################################
# Statistics
###########################################################

st.subheader("Dataset Statistics")

st.dataframe(df.describe())

###########################################################
# Charts
###########################################################

st.subheader("Interactive Charts")

numeric_columns=numeric.columns.tolist()

if len(numeric_columns)>=1:

    column=st.selectbox(
        "Select Numeric Column",
        numeric_columns
    )

    fig=px.histogram(
        df,
        x=column,
        nbins=40,
        color_discrete_sequence=["royalblue"]
    )

    st.plotly_chart(fig,use_container_width=True)

###########################################################
# Scatter Plot
###########################################################

if len(numeric_columns)>=2:

    x=st.selectbox(
        "X Axis",
        numeric_columns,
        key=1
    )

    y=st.selectbox(
        "Y Axis",
        numeric_columns,
        index=1,
        key=2
    )

    fig=px.scatter(
        df,
        x=x,
        y=y,
        color=y,
        size=y
    )

    st.plotly_chart(fig,use_container_width=True)

###########################################################
# Correlation
###########################################################

st.subheader("Correlation Heatmap")

if len(numeric_columns)>=2:

    corr=numeric.corr()

    fig,ax=plt.subplots(figsize=(10,6))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

###########################################################
# Top Values
###########################################################

st.subheader("Top Records")

st.dataframe(df.head(20))

###########################################################
# Machine Learning Forecast
###########################################################

st.subheader("AI Forecast")

if len(numeric_columns)>=2:

    target=st.selectbox(
        "Target Variable",
        numeric_columns,
        key=10
    )

    features=[i for i in numeric_columns if i!=target]

    X=df[features]

    y=df[target]

    X=X.fillna(X.mean())

    y=y.fillna(y.mean())

    X_train,X_test,y_train,y_test=train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model=LinearRegression()

    model.fit(X_train,y_train)

    score=model.score(X_test,y_test)

    st.metric(
        "Prediction Accuracy (R²)",
        round(score,3)
    )

###########################################################
# Business Insights
###########################################################

st.subheader("AI Business Insights")

insights=[]

if df.isnull().sum().sum()>0:

    insights.append("Dataset contains missing values.")

else:

    insights.append("No missing values detected.")

if len(numeric_columns)>0:

    for col in numeric_columns:

        if df[col].std()>df[col].mean():

            insights.append(
                f"{col} has high variability."
            )

        if df[col].mean()>df[col].median():

            insights.append(
                f"{col} is positively skewed."
            )

for i in insights:

    st.success(i)

###########################################################
# Recommendation Engine
###########################################################

st.subheader("Business Recommendations")

recommend=[]

if df.isnull().sum().sum()>0:

    recommend.append(
        "Clean missing values."
    )

recommend.append(
    "Monitor high variance features."
)

recommend.append(
    "Improve forecasting using larger datasets."
)

recommend.append(
    "Create customer segmentation."
)

recommend.append(
    "Monitor monthly KPIs."
)

recommend.append(
    "Automate report generation."
)

for r in recommend:

    st.info(r)

###########################################################
# Download CSV
###########################################################

st.subheader("Download Processed Data")

csv=df.to_csv(index=False)

st.download_button(

    "Download CSV",

    csv,

    file_name="processed_data.csv",

    mime="text/csv"

)

###########################################################
# Footer
###########################################################

st.markdown("---")

st.caption("AI Business Intelligence Dashboard")
