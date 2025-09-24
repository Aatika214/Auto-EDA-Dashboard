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
# 🧮 Raw Data
# ========================
with st.expander("🧮 View Raw Data"):
    st.dataframe(df.head(100))   

# ========================
# ✨ NEW PROFESSIONAL GRAPHS
# ========================

st.subheader("✨ Advanced Comparative Visuals")

# 1. Top 10 Customers by Spending (Only Axis Labels, No Tick Values)
st.markdown("#### 🏅 Top 10 Customers by Spending")
top_customers = (df.groupby("customer_id")["revenue"].sum()
                   .sort_values(ascending=False).head(10).reset_index())

fig1, ax1 = plt.subplots(figsize=(8,5))
sns.barplot(data=top_customers, x="revenue", y="customer_id", palette="viridis", ax=ax1)

# Titles and axis labels
ax1.set_title("Top 10 Customers by Spending", fontsize=14)
ax1.set_xlabel("Revenue ($)", fontsize=12)
ax1.set_ylabel("Customer ID", fontsize=12)

# ❌ Hide axis tick values
ax1.set_xticks([])
ax1.set_yticks([])

# ✅ Show values on bars
for p in ax1.patches:
    ax1.annotate(f"${p.get_width():,.0f}",
                 (p.get_width(), p.get_y() + p.get_height()/2),
                 xytext=(5,0), textcoords="offset points",
                 ha="left", va="center", fontsize=10, color="black")

st.pyplot(fig1)


# 2. Monthly Orders Trend with values
st.markdown("#### 📅 Monthly Orders Trend")
df["month"] = pd.to_datetime(df["order_date"]).dt.to_period("M")
monthly_orders = df.groupby("month")["order_id"].nunique().reset_index()
monthly_orders["month"] = monthly_orders["month"].astype(str)

fig2, ax2 = plt.subplots(figsize=(10,5))
sns.lineplot(data=monthly_orders, x="month", y="order_id", marker="o", ax=ax2, color="teal")

ax2.set_title("Monthly Orders Trend", fontsize=14)
ax2.set_xlabel("Month")
ax2.set_ylabel("Number of Orders")
ax2.tick_params(axis="x", rotation=45)

# Add values on points
for i, row in monthly_orders.iterrows():
    ax2.text(row["month"], row["order_id"]+0.5, str(row["order_id"]),
             ha="center", fontsize=9, color="black")

st.pyplot(fig2)


# 3. Orders by Region with values
st.markdown("#### 🌍 Orders by Region")
region_orders = df.groupby("region")["order_id"].nunique().reset_index()

fig3, ax3 = plt.subplots(figsize=(8,5))
sns.barplot(data=region_orders, x="region", y="order_id", palette="coolwarm", ax=ax3)

ax3.set_title("Orders by Region", fontsize=14)
ax3.set_xlabel("Region")
ax3.set_ylabel("Number of Orders")
ax3.tick_params(axis="x", rotation=30)

# Add values on bars
for p in ax3.patches:
    ax3.annotate(f"{int(p.get_height()):,}",
                 (p.get_x() + p.get_width()/2., p.get_height()),
                 ha="center", va="bottom", fontsize=10, color="black")

st.pyplot(fig3)
