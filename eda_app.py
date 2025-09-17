import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit App Title
st.title("📊 Advanced EDA Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔎 Dataset Preview")
    st.write(df.head())

    st.subheader("📐 Dataset Info")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write("Column Data Types:")
    st.write(df.dtypes)

    st.subheader("❌ Missing Values")
    st.write(df.isnull().sum())

    st.subheader("📊 Summary Statistics")
    st.write(df.describe(include="all").transpose())

    # Numeric columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    # Distribution of numeric columns
    st.subheader("📈 Distributions of Numeric Columns")
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax, color="skyblue")
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    # Count plots for categorical columns
    st.subheader("📊 Categorical Columns Distribution")
    for col in cat_cols:
        fig, ax = plt.subplots()
        sns.countplot(y=df[col], order=df[col].value_counts().index, ax=ax, palette="viridis")
        ax.set_title(f"Count of {col}")
        st.pyplot(fig)

    # Correlation heatmap
    if len(num_cols) > 1:
        st.subheader("🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Time series trend if date column exists
    date_cols = df.select_dtypes(include=["datetime64", "object"]).columns
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col])
            st.subheader(f"⏳ Time Series Trend by {col}")
            ts = df.groupby(col)[num_cols].mean()
            fig, ax = plt.subplots(figsize=(10, 5))
            ts.plot(ax=ax)
            ax.set_title(f"Trend over {col}")
            st.pyplot(fig)
            break
        except Exception:
            continue
