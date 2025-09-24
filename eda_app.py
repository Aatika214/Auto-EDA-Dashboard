import streamlit as st
import pandas as pd
import numpy as np
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
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

# Calculate revenue
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
total_orders = df["order_id"].nunique()
avg_order_value = total_revenue / total_orders if total_orders > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
col2.metric("📦 Total Orders", f"{total_orders:,}")
col3.metric("🛒 Avg Order Value", f"${avg_order_value:,.2f}")

# ========================
# 📈 Revenue Trend
# ========================
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
st.subheader("💳 Payment Methods Distribution")
payment_counts = df["payment_method"].value_counts().reset_index()
payment_counts.columns = ["payment_method", "count"]

fig_pay = px.pie(payment_counts, names="payment_method", values="count",
                 title="Payment Method Share", hole=0.4)
st.plotly_chart(fig_pay, use_container_width=True)

# ========================
# 🌍 Regional Analysis
# ========================
st.subheader("🌍 Regional Sales Analysis")
region_sales = df.groupby("region")["revenue"].sum().reset_index()

fig_region = px.bar(region_sales, x="region", y="revenue",
                    title="Revenue by Region",
                    labels={"region": "Region", "revenue": "Revenue"})
st.plotly_chart(fig_region, use_container_width=True)

# ========================
# 📌 Correlations
# ========================
st.subheader("📌 Correlation Heatmap")
num_cols = ["quantity", "price", "discount", "revenue"]
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ========================
# 🔍 Distributions (Professional Visuals)
# ========================
st.subheader("🔍 Distributions of Numeric Features")

# Quantity
fig_q = px.histogram(df, x="quantity", nbins=30, marginal="box",
                     title="Distribution of Quantity", color_discrete_sequence=["#636EFA"])
st.plotly_chart(fig_q, use_container_width=True)

# Price
log_price = st.checkbox("🔄 Log Scale for Price", value=False)
fig_p = px.histogram(df, x="price", nbins=40, title="Distribution of Price",
                     color_discrete_sequence=["#EF553B"])
if log_price:
    fig_p.update_xaxes(type="log")
st.plotly_chart(fig_p, use_container_width=True)

# Discount
fig_d = px.histogram(df, x="discount", nbins=20, histnorm="probability density",
                     title="Distribution of Discount (Density)", color_discrete_sequence=["#00CC96"])
st.plotly_chart(fig_d, use_container_width=True)

# Revenue
fig_r = px.violin(df, y="revenue", box=True, points="all",
                  title="Revenue Spread (Violin + Outliers)", color_discrete_sequence=["#AB63FA"])
st.plotly_chart(fig_r, use_container_width=True)

# ========================
# 🧮 Raw Data
# ========================
with st.expander("🧮 View Raw Data"):
    st.dataframe(df.head(100))
