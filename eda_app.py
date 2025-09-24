import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Streamlit page config
st.set_page_config(page_title="📦 E-commerce EDA Dashboard", layout="wide")

st.title("📦 Advanced E-commerce EDA Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your ecommerce CSV file", type=["csv"])
if uploaded_file is None:
    st.info("👆 Upload your ecommerce dataset to begin.")
    st.stop()

# Load dataset
df = pd.read_csv(uploaded_file)

# Ensure correct dtypes
if "order_date" in df.columns:
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

# Calculate revenue if columns exist
if {"quantity", "price", "discount"}.issubset(df.columns):
    df["revenue"] = df["quantity"] * df["price"] * (1 - df["discount"])

# ========================
# 🔎 Dataset Overview
# ========================
with st.expander("📋 Dataset Overview"):
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write("Data Types:", df.dtypes)
    st.write("Missing Values:", df.isnull().sum())

# ========================
# 📊 KPIs
# ========================
st.subheader("📊 Key Metrics")
total_revenue = df["revenue"].sum()
total_orders = df["order_id"].nunique() if "order_id" in df.columns else 0
avg_order_value = total_revenue / total_orders if total_orders > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
col2.metric("📦 Total Orders", f"{total_orders:,}")
col3.metric("🛒 Avg Order Value", f"${avg_order_value:,.2f}")

# ========================
# 📈 Revenue Trend
# ========================
if "order_date" in df.columns:
    st.subheader("📈 Revenue Over Time")
    freq = st.radio("Select frequency:", ["D", "M"], index=1, horizontal=True)

    if freq == "D":
        ts = df.groupby(df["order_date"].dt.date)["revenue"].sum().reset_index()
    else:
        ts = df.groupby(df["order_date"].dt.to_period("M"))["revenue"].sum().reset_index()
        ts["order_date"] = ts["order_date"].astype(str)

    fig = px.line(ts, x="order_date", y="revenue", title="Revenue Trend")
    st.plotly_chart(fig, use_container_width=True)

# ========================
# 🏆 Top Products & Categories
# ========================
if "product_id" in df.columns and "category" in df.columns:
    st.subheader("🏆 Top Products & Categories")

    top_n = st.slider("How many top items to show?", 5, 20, 10)

    # Top Products
    top_products = (df.groupby("product_id")["revenue"].sum()
                      .sort_values(ascending=False).head(top_n).reset_index())
    fig_prod = px.bar(top_products, x="product_id", y="revenue",
                      title=f"Top {top_n} Products by Revenue",
                      labels={"product_id": "Product ID", "revenue": "Revenue"})
    st.plotly_chart(fig_prod, use_container_width=True)

    # Top Categories
    top_categories = (df.groupby("category")["revenue"].sum()
                        .sort_values(ascending=False).reset_index())
    fig_cat = px.bar(top_categories, x="category", y="revenue",
                     title="Revenue by Category",
                     labels={"category": "Category", "revenue": "Revenue"})
    st.plotly_chart(fig_cat, use_container_width=True)

# ========================
# 💳 Payment Method Analysis
# ========================
if "payment_method" in df.columns:
    st.subheader("💳 Payment Methods Distribution")
    payment_counts = df["payment_method"].value_counts().reset_index()
    payment_counts.columns = ["payment_method", "count"]

    fig_pay = px.pie(payment_counts, names="payment_method", values="count",
                     title="Payment Method Share", hole=0.4)
    st.plotly_chart(fig_pay, use_container_width=True)

# ========================
# 🌍 Regional Analysis
# ========================
if "region" in df.columns:
    st.subheader("🌍 Regional Sales Analysis")
    region_sales = df.groupby("region")["revenue"].sum().reset_index()

    fig_region = px.bar(region_sales, x="region", y="revenue",
                        title="Revenue by Region",
                        labels={"region": "Region", "revenue": "Revenue"})
    st.plotly_chart(fig_region, use_container_width=True)

# ========================
# 📌 Correlations
# ========================
num_cols = [col for col in ["quantity", "price", "discount", "revenue"] if col in df.columns]
if len(num_cols) > 1:
    st.subheader("📌 Correlation Heatmap")
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ========================
# 📌 Professional Graphs (Your Requirement)
# ========================

# Money spent per customer
if {"customer_id", "country", "revenue"}.issubset(df.columns):
    st.subheader("💰 Money Spent by Customers")
    money_spent = df.groupby(by=['customer_id','country'], as_index=False)['revenue'].sum()
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(money_spent.customer_id, money_spent.revenue, color="teal")
    ax.set_xlabel('Customer ID')
    ax.set_ylabel('Money Spent ($)')
    ax.set_title('Money Spent by Different Customers')
    st.pyplot(fig)

# Number of orders per month
if {"order_id", "order_date"}.issubset(df.columns):
    st.subheader("🗓️ Number of Orders per Month")
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)
    orders_by_month = df.groupby("year_month")["order_id"].nunique()
    fig, ax = plt.subplots(figsize=(12,5))
    orders_by_month.plot(kind="bar", color="orange", ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Orders")
    ax.set_title("Orders per Month")
    st.pyplot(fig)

# Number of orders by country (excluding UK)
if {"country", "order_id"}.issubset(df.columns):
    st.subheader("🌍 Orders by Country")
    group_country_orders = df.groupby('country')["order_id"].count().sort_values()
    if "United Kingdom" in group_country_orders.index:
        group_country_orders = group_country_orders.drop("United Kingdom")
    fig, ax = plt.subplots(figsize=(12,6))
    group_country_orders.plot(kind="barh", color="purple", ax=ax)
    ax.set_xlabel("Number of Orders")
    ax.set_ylabel("Country")
    ax.set_title("Orders by Country (excluding UK)")
    st.pyplot(fig)

# ========================
# 🧮 Raw Data
# ========================
with st.expander("🧮 View Raw Data"):
    st.dataframe(df.head(100))
