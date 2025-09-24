import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit App Title
st.title("🛒 E-Commerce Advanced EDA Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your E-Commerce CSV file", type=["csv"])

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

    # Convert Invoice Date if present
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["year_month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    # Money spent per customer
    if {"CustomerID", "Country", "AmountSpent"}.issubset(df.columns):
        st.subheader("💰 Money Spent by Customers")
        money_spent = df.groupby(by=['CustomerID','Country'], as_index=False)['AmountSpent'].sum()
        fig, ax = plt.subplots(figsize=(12,5))
        plt.plot(money_spent.CustomerID, money_spent.AmountSpent, color="teal")
        plt.xlabel('Customer ID')
        plt.ylabel('Money Spent ($)')
        plt.title('Money Spent by Different Customers')
        st.pyplot(fig)

    # Number of orders per month
    if {"InvoiceNo","year_month"}.issubset(df.columns):
        st.subheader("🗓️ Number of Orders per Month")
        orders_by_month = df.groupby("year_month")["InvoiceNo"].nunique()
        fig, ax = plt.subplots(figsize=(12,5))
        orders_by_month.plot(kind="bar", color="orange", ax=ax)
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Orders")
        ax.set_title("Orders per Month")
        st.pyplot(fig)

    # Number of orders by country (excluding UK if exists)
    if {"Country","InvoiceNo"}.issubset(df.columns):
        st.subheader("🌍 Orders by Country")
        group_country_orders = df.groupby('Country')["InvoiceNo"].count().sort_values()
        if "United Kingdom" in group_country_orders.index:
            group_country_orders = group_country_orders.drop("United Kingdom")
        fig, ax = plt.subplots(figsize=(12,6))
        group_country_orders.plot(kind="barh", color="purple", ax=ax)
        ax.set_xlabel("Number of Orders")
        ax.set_ylabel("Country")
        ax.set_title("Orders by Country (excluding UK)")
        st.pyplot(fig)

    # Correlation heatmap (if numeric features exist)
    num_cols = df.select_dtypes(include=["int64","float64"]).columns
    if len(num_cols) > 1:
        st.subheader("🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
