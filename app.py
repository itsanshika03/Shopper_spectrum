import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# ==========================================
# SHOPPER SPECTRUM GLOBAL THEME
# ==========================================

# Logo colors
PRIMARY_PURPLE = "#5B3DF5"
PRIMARY_VIOLET = "#8B5CF6"
PRIMARY_PINK = "#EC4899"
PRIMARY_BLUE = "#4DA3FF"

# Status Colors
SUCCESS = "#22C55E"
WARNING = "#F59E0B"
DANGER = "#EF4444"

# Dashboard Theme
BG = "#0F1117"
CARD = "#1A1D2E"
TEXT = "#FFFFFF"

# Main Gradient (matches logo)
PRIMARY_GRADIENT = (
    "linear-gradient(90deg,"
    "#4DA3FF 0%,"
    "#5B3DF5 30%,"
    "#8B5CF6 65%,"
    "#EC4899 100%)"
)

# Hover Gradient
SECONDARY_GRADIENT = (
    "linear-gradient(90deg,"
    "#5B3DF5 0%,"
    "#8B5CF6 50%,"
    "#EC4899 100%)"
)

# Chart Colors
CHART_COLORS = [
    "#4DA3FF",
    "#6D7DFF",
    "#8B5CF6",
    "#C056FF",
    "#EC4899"
]

st.markdown("""
<style>

/* Buttons */

div.stButton > button:first-child{

background: linear-gradient(90deg,#5B3DF5,#8B5CF6,#EC4899);

color:white;

border:none;

border-radius:12px;

height:3em;

font-size:16px;

font-weight:600;

transition:0.3s;

box-shadow:0px 6px 20px rgba(139,92,246,.30);

}


div.stButton > button:first-child:hover{

background:linear-gradient(90deg,#4DA3FF,#8B5CF6,#EC4899);

transform:scale(1.02);

}


/* Metric Cards */

div[data-testid="metric-container"]{

background:#1A1D2E;

border:1px solid rgba(139,92,246,.25);

padding:18px;

border-radius:15px;

}


/* Sidebar selected page */

[data-testid="stSidebarNav"] a[aria-current="page"]{

background:linear-gradient(90deg,#5B3DF5,#8B5CF6,#EC4899);

border-radius:12px;

}

</style>
""", unsafe_allow_html=True)

# ============================
# Page Configuration
# ============================

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ---------- Global Font ---------- */

html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", sans-serif;
}

.stButton>button{
    background: linear-gradient(
    90deg,
    #5B3DF5,
    #8B5CF6,
    #EC4899,
    #4DA3FF
    );    
    color:white;
    border:none;
    border-radius:12px;
    height:52px;
    font-size:18px;
    font-weight:bold;
    font-weight:600;
    transition:0.3s;
}

.stButton>button:hover{
    background:linear-gradient(90deg,#5B4EF2,#00C897);
    transform:scale(1.02);
}
            
div[data-testid="stMetric"]{
    background:#223246;
    padding:18px;
    border-radius:14px;
    border:1px solid #31475f;
    transition:0.3s;
}

div[data-testid="stMetric"]:hover{
    transform:translateY(-6px);
    border:1px solid #5B6DFF;
    box-shadow:0px 8px 25px rgba(91,109,255,.35);
}

/* Sidebar Radio Buttons */

section[data-testid="stSidebar"] .stRadio > div{
    gap:10px;
}

section[data-testid="stSidebar"] label{
    background:#223246;
    padding:12px 15px;
    border-radius:12px;
    border:1px solid #31475f;
    transition:0.3s;
}

section[data-testid="stSidebar"] label:hover{
    background:#2E4157;
    border:1px solid #5B6DFF;
    transform:translateX(5px);
}

section[data-testid="stSidebar"] input:checked + div{
    color:#5B6DFF;
    font-weight:700;
}          

</style>                  
""", unsafe_allow_html=True)

# Load Logo
logo = Image.open("assets/logo.png")

# ==========================================
# Load Dataset & Models
# ==========================================

df = pd.read_csv("online_retail.csv", encoding="ISO-8859-1")

# ==========================================
# Create Clean Dataset (Same as Training)
# ==========================================

df_clean = df.copy()

df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])

df_clean = df_clean.dropna(subset=["CustomerID"])

df_clean = df_clean.drop_duplicates()

df_clean = df_clean[
    ~df_clean["InvoiceNo"].astype(str).str.startswith("C")
]

df_clean = df_clean[df_clean["Quantity"] > 0]

df_clean = df_clean[df_clean["UnitPrice"] > 0]

df_clean["TotalAmount"] = (
    df_clean["Quantity"] * df_clean["UnitPrice"]
)

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
rfm = joblib.load("rfm_data.pkl") 

# ==========================================
# Build Product Similarity Matrix
# ==========================================

customer_product = df_clean.pivot_table(
    index="CustomerID",
    columns="StockCode",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

similarity = cosine_similarity(customer_product.T)

similarity_df = pd.DataFrame(
    similarity,
    index=customer_product.columns,
    columns=customer_product.columns
)

# ==========================================
# KPI Metrics
# ==========================================

df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

total_revenue = df["TotalAmount"].sum()

# KPI Counts
customer_count = df["CustomerID"].nunique()
product_count = df["StockCode"].nunique()
transaction_count = df["InvoiceNo"].nunique()
country_count = df["Country"].nunique() 

# ==========================================
# Data Preparation for Dashboard
# ==========================================

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

monthly_sales = (
    df.groupby("Month")["TotalAmount"]
    .sum()
    .reset_index()
)

country_sales = (
    df.groupby("Country")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

top_products = (
    df.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

top_customers = (
    df.groupby("CustomerID")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

# ============================
# Sidebar
# ============================

st.sidebar.image(logo, width=250)

st.sidebar.markdown("""
<h2 style='text-align:center; margin-bottom:0px;'>
🛍 Shopper Spectrum
</h2>

<p style='text-align:center;
color:gray;
font-size:15px;
margin-top:-5px;'>
Customer Analytics Dashboard
</p>
""", unsafe_allow_html=True)

st.sidebar.divider()

st.sidebar.markdown("""
<h3 style="
color:white;
text-align:center;
padding-bottom:10px;
">
📂 Navigation
</h3>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    label="",
    options=[
        "🏠 Home",
        "📊 Analytics Dashboard",
        "👥 Customer Segmentation",
        "🎯 Product Recommendation",
        "📈 Business Insights"
    ],
label_visibility="collapsed"
)

st.sidebar.divider()

st.sidebar.info(
"""
**Machine Learning Models**

✅ K-Means Clustering

✅ Cosine Similarity

✅ RFM Analysis
"""
)

# ============================
# Home Page
# ============================

if page == "🏠 Home":
    
    st.markdown(f"""
    <div style="
    background: {PRIMARY_GRADIENT};    
    padding:32px;
    border-radius:20px;
    text-align:center;
    color:white;
    margin-bottom:30px;
    box-shadow:0 8px 30px rgba(91,61,245,.35);
    ">

    <h1 style="
    "margin-bottom:5px;"
    font-size=44px;
    font-weight=800;
    ">
    🛍 Shopper Spectrum
    </h1>

    <h4 style="
    font-size:24px;
    font-weight:600;
    margin-top:10px;
    ">    
    Customer Segmentation & Product Recommendation System
    </h4>

    <p style="
    font-size:20px;
    font-weight=500;
    line-height=1.8
    ">
    K-Means Clustering • RFM Analysis • Cosine Similarity
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "💰 Revenue",
        f"£{total_revenue/1000000:.2f} M",
        "Total Sales"
    )

    col2.metric(
        "👥 Customers",
        f"{customer_count:,}",
        "Unique Customers"
    )

    col3.metric(
        "📦 Products",
        f"{product_count:,}",
        "Products"
    )

    col4.metric(
         "🧾 Orders",
        f"{transaction_count:,}",
        "Invoices"
    )

    col5.metric(
        "🌍 Countries",
        f"{country_count}",
        "Countries"
    )
    
    st.markdown("---")

    left, right = st.columns(2)

    with left:

        st.markdown("""
            <div style="
            background:#1E2D3D;
            padding:25px;
            border-radius:15px;
            border-left:6px solid #5B6DFF;
            font-family:Inter,Segoe UI,sans-serif;
            min-height:330px;        
            ">

            <h3 style="
            color:white;
            font-size=28px;
            font-weight=700;
            ">
            📂 Dataset Overview
            </h3>

            <hr style="border:1px solid #3E5C76;">

            <div style="
            font-size:20px;
            font-weight:500;
            line-height:2
            color:#E8EDF5;
            ">

            📄 <b>Transactions:</b> {transaction_count:,}<br>

            👥 <b>Customers:</b> {customer_count:,}<br>

            📦 <b>Products:</b> {product_count:,}<br>

            🌍 <b>Countries:</b> {country_count}<br>

            💰 <b>Revenue:</b> £{total_revenue/1000000:.2f} M 
                    
            </div>
            """,
            unsafe_allow_html=True)
        
    with right:

        st.markdown("""
            <div style="
            background:#1F2D27;
            padding:25px;
            border-radius:15px;
            border-left:6px solid #00C853;
            min-height:330px; 
            font-family:Inter,Segoe UI,sans-serif;           
            ">

            <h3 style="
            color:white;
            font-size:28px;
            font-weight:700:
            ">
            🤖 Machine Learning Pipeline
            </h3>

            <hr style="border:1px solid #2D6A4F;">

            <div style="
            font-size:20px;
            font-weight:500;
            line-height:2;
            color:#E8EDF5;
            ">

            ✅ Data Cleaning<br>

            ✅ Feature Engineering<br>

            ✅ RFM Analysis<br>

            ✅ K-Means Clustering<br>

            ✅ Cosine Similarity<br>

            ✅ Product Recommendation Engine

            </p>

            </div>
            """,
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
        <div style="
        background:#1e293b;
        padding:18px;
        border-radius:12px;
        font-size:26px;
        font-weight:700;
        color:white;
        ">
        📊 Sales Performance
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

    # Two charts in first row
    col1,col2=st.columns(2)

    # =====================================================
    # Monthly Sales Trend
    # =====================================================

    with col1:

        fig1 = px.line(
            monthly_sales,
            x="Month",
            y="TotalAmount",
            markers=True,
            color_discrete_sequence=[CHART_COLORS[0]],
            title="📈 Monthly Revenue Trend"
        )
        
        fig1.update_layout(
            template="plotly_dark",
            height=420,
            title_x=0.25,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        fig1.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(
                color="white",
                size=13
            ),            
            title_font=dict(
                size=20
            ),
            hovermode="x unified"
        )
        
        fig1.update_traces(
            line=dict(color=PRIMARY_PURPLE, width=4),
            marker=dict(
                size=8,
                color=PRIMARY_PINK,
                line=dict(color="white", width=1)
            )
        )

        st.plotly_chart(
            fig1,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )
    # =====================================================
    # Top Countries
    # =====================================================

    with col2:
        fig2=px.bar(
            country_sales,
            x="TotalAmount",
            y="Country",
            orientation="h",
            title="🌍 Top 10 Countries by Sales",
            color="TotalAmount",
            color_continuous_scale=CHART_COLORS
        )

        fig2.update_layout(
            template="plotly_dark",
            height=420,
            title_x=0.20,
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(categoryorder="total ascending")
        )

        fig2.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(color="white"),
            title_font_size=22,
            coloraxis_showscale=False,
        )

        st.plotly_chart(
            fig2,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )
    
    st.markdown("<br>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    
    # =====================================================
    # Best Selling Products
    # =====================================================

    with col3:

        fig3 = px.bar(
            top_products,
            x="Quantity",
            y="Description",
            orientation="h",
            title="🛒 Top 10 Selling Products",
            color="Quantity",
            color_continuous_scale=CHART_COLORS
        )

        fig3.update_layout(
            height=430,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20),
            title_x=0.18,
            yaxis={"categoryorder": "total ascending"}
        )

        fig3.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(color="white"),
            coloraxis_showscale=False
        )

        st.plotly_chart(
            fig3,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )
    # =====================================================
    # Top Customers
    # =====================================================

    with col4:

        fig4 = px.bar(
            top_customers,
            x="CustomerID",
            y="TotalAmount",
            color="TotalAmount",
            color_continuous_scale=CHART_COLORS,
            title="👑 Top 10 Customers by Revenue"
        )

        fig4.update_layout(
            height=420,
            template="plotly_dark",
            margin=dict(l=10, r=10, t=50, b=10),
            title_x=0.02,
            yaxis={"categoryorder": "total ascending"},
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            xaxis_title="Customer ID",
            yaxis_title="Revenue (£)",
            font=dict(color="white"),
            coloraxis_showscale=False
        )

        fig4.update_traces(
            texttemplate="£%{y:,.0f}",
            textposition="outside",
            marker_line_color="white",
            marker_line_width=0.6,
            opacity=0.95,
            hovertemplate="<b>Customer ID:</b> %{x}<br><b>Revenue:</b> £%{y:,.2f}<extra></extra>"        
        )

        st.plotly_chart(
            fig4,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )
    
# ============================
# Analytics Dashboard
# ============================

if page == "📊 Analytics Dashboard":

    st.markdown(f"""
    <div style="
    background: {PRIMARY_GRADIENT};    
    padding:32px;
    border-radius:22px;
    text-align:center;
    color:white;
    margin-bottom:30px;
    box-shadow:0 8px 30px rgba(91,61,245,.35);
    ">

    <h1 style="
    font-size:42px;
    font-weight:800;
    ">
    📊 Analytics Dashboard
    </h1>

    <h4 style="
    font-size:24px;
    font-weight:600;
    ">
    Interactive Sales Analytics & Business Performance Insights
    </h4>

    <p>
    Sales Trends • Revenue Analysis • Customer Insights • Product Performance
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        selected_country = st.selectbox(
            "🌍 Select Country",
            ["All"] + sorted(df["Country"].unique().tolist())
        )

    with col2:
        selected_month = st.selectbox(
            "📅 Select Month",
            ["All"] + sorted(df["Month"].unique().tolist())
        )

    filtered_df = df.copy()

    if selected_country != "All":
        filtered_df = filtered_df[
            filtered_df["Country"] == selected_country
        ]

    if selected_month != "All":
        filtered_df = filtered_df[
            filtered_df["Month"] == selected_month
        ]    

    filtered_monthly = (
        filtered_df.groupby("Month")["TotalAmount"]
        .sum()
        .reset_index()   
    )

    filtered_country = (
        filtered_df.groupby("Country")["TotalAmount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    filtered_products = (
        filtered_df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    filtered_customers = (
        filtered_df.groupby("CustomerID")["TotalAmount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )  

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "💰 Revenue",
        f"£{filtered_df['TotalAmount'].sum():,.0f}"
    )

    k2.metric(
        "🧾 Orders",
        filtered_df["InvoiceNo"].nunique()
    )

    k3.metric(
        "👥 Customers",
        filtered_df["CustomerID"].nunique()
    )

    k4.metric(
        "📦 Products",
        filtered_df["StockCode"].nunique()
    )
    
    col1, col2 = st.columns(2)
    
    with col1:

        st.markdown("📅 Revenue Trend")

        fig = px.line(
            monthly_sales,
            x="Month",
            y="TotalAmount",
            color_discrete_sequence=[CHART_COLORS[0]],
            title="Monthly Revenue Trend",
            markers=True
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color=TEXT,
            title_font_size=22,
            title_x=0.02,
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="#1F2937",
                font_size=13,
                font_color="white"
            )
        )

        fig.update_traces(
            line=dict(
                color=PRIMARY_PURPLE, 
                width=4
            ),
            marker=dict(
                size=8,
                color=PRIMARY_PINK,
                line=dict(color="white", width=1)
            ),
           hovertemplate=
                "<b>%{x}</b><br>"
                "Revenue: £%{y:,.0f}<extra></extra>"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.markdown("### 🌍 Country Revenue")    

        fig = px.bar(
            filtered_country,
            x="Country",
            y="TotalAmount",
            title="Top 10 Countries",
            color="TotalAmount",
            color_continuous_scale=CHART_COLORS
        )

        fig.update_layout(
            template="plotly_dark",
            yaxis=dict(categoryorder="total ascending"),
            coloraxis_showscale=False,
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color=TEXT,
            title_font_size=22,
            title_x=0.02,
            margin=dict(l=20, r=20, t=50, b=20),
            hoverlabel=dict(
                bgcolor="#1F2937",
                font_size=13,
                font_color="white"
            )
        )

        fig.update_traces(
            marker_line_width=1,
            marker_line_color="white",
            opacity=0.95,
            texttemplate="£%{y:,.0f}",
            textposition="outside"
        )

        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3: 
        st.markdown("### 🛒 Best Selling Products")

        fig = px.bar(
            filtered_products,
            x="Quantity",
            y="Description",
            orientation="h",
            color="Quantity",
            color_continuous_scale=CHART_COLORS,
            title="Top 10 Selling Products"
        )

        fig.update_layout(
            template="plotly_dark",
            yaxis=dict(categoryorder="total ascending"),
            hoverlabel=dict(
                bgcolor="#1F2937",
                font_size=13,
                font_color="white"
            ),
            coloraxis_showscale=False
        )

        fig.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color=TEXT,
            title_font_size=22,
            title_x=0.02,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        fig.update_traces(
            marker_line_color="white",
            marker_line_width=0.5,
            opacity=0.95,
            texttemplate="%{x:,.0f}",
            textposition="outside"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col4:    
        st.markdown("### 👑 Top Customers")

        fig = px.bar(
            filtered_customers,
            x="CustomerID",
            y="TotalAmount",
            title="Top Customers",
            color="TotalAmount", 
            color_continuous_scale=CHART_COLORS
        )

        fig.update_xaxes(
            tickangle=-45
        )

        fig.update_layout(
            template="plotly_dark",
            yaxis=dict(categoryorder="total ascending"),
            bargap=0.35,
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color=TEXT,
            title_font_size=22,
            title_x=0.02,
            margin=dict(l=20, r=20, t=50, b=20),
            hoverlabel=dict(
                bgcolor="#1F2937",
                font_size=13,
                font_color="white"
            ),
            coloraxis_showscale=False
        )

        fig.update_traces(
            marker_line_color="white",
            marker_line_width=0.8,
            opacity=0.95,
            texttemplate="£%{y:,.0f}",
            textposition="outside"
        )

        st.plotly_chart(fig, use_container_width=True)
 
    st.markdown("---")
    st.subheader("💡 Key Business Insights")

    top_country = filtered_country.iloc[0]["Country"] if not filtered_country.empty else "N/A"
    top_product = filtered_products.iloc[0]["Description"] if not filtered_products.empty else "N/A"

    st.info(f"""
    - 🌍 **Highest Revenue Country:** {top_country}
    - 🛒 **Best Selling Product:** {top_product} 
    - 👥 **Active Customers:** {filtered_df['CustomerID'].nunique()}
    - 💰 **Total Revenue:** £{filtered_df['TotalAmount'].sum():,.2f}
    """)

# ============================
# Customer Segmentation
# ============================

if page == "👥 Customer Segmentation":

    st.markdown(f"""
    <div style="
    background: {PRIMARY_GRADIENT};
    padding:32px;
    border-radius:20px;
    text-align:center;
    color:white;
    margin-bottom:30px;
    box-shadow:0 8px 30px rgba(91,61,245,.35);
    ">

    <h1 style="
    font-size:42px;
    font-weight:800;
    ">
    👥 Customer Segmentation
    </h1>

    <h4 style="
    font-size:24px;
    font-weight:600;
    ">
    Predict customer groups using K-Means Clustering
    </h4>

    <p>
    Recency • Frequency • Monetary (RFM)
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("📝 Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        if "recency" not in st.session_state:
            st.session_state.recency = 30

        recency = st.number_input(
            "📅 Recency (Days)",
            min_value=0,
            value=st.session_state.recency,
            key="recency"
        )

    with col2:
        if "frequency" not in st.session_state:
            st.session_state.frequency = 5

        frequency = st.number_input(
            "🛒 Frequency",
            min_value=1,
            value=st.session_state.frequency,
            key="frequency"
        )

    with col3:
        if "monetary" not in st.session_state:
            st.session_state.monetary = 500.0

        monetary = st.number_input(
            "💰 Monetary (£)",
            min_value=0.0,
            value=st.session_state.monetary,
            key="monetary"
        )
        
    st.write("")

    st.info("""
    ### 📘 Understanding RFM

    📅 **Recency** → Number of days since the customer's last purchase.

    🛒 **Frequency** → Total number of purchases made.

    💰 **Monetary** → Total amount spent by the customer.

    These three values are used by the K-Means Machine Learning model to predict the customer segment.
    """)

    predict = st.button(
        "🎯 Predict Customer Segment",
    use_container_width=True
    )

    # ==========================================
    # Predict Customer Segment
    # ==========================================

    if predict:
        st.write("Predict button clicked!")

        # Create input dataframe
        input_data = pd.DataFrame({
        "Recency": [recency],
        "Frequency": [frequency],
        "Monetary": [monetary]
        })

        # Scale the input
        scaled_input = scaler.transform(input_data)

        # Predict cluster
        cluster = kmeans.predict(scaled_input)[0]
        st.write(cluster) 

        # Cluster names
        cluster_names = {
        0: "🟢 Regular Customers",
        1: "🔴 Inactive Customers",
        2: "💎 Elite Customers",
        3: "⭐ High Value Customers",
        4: "👑 Top Customers"
        }

        cluster_colors = {
        0: "#4CAF50",   # Green
        1: "#F44336",   # Red
        2: "#9C27B0",   # Purple
        3: "#FF9800",   # Orange
        4: "#2196F3"    # Blue
        }

        color = cluster_colors.get(cluster, "#5B6DFF")

        st.markdown("---")

        st.markdown(f"""
        <div style="
        background:linear-gradient(135deg,#1A1D2E,#252B45);
        padding:18px;
        border-radius:15px;
        border-left:6px solid {color};
        box-shadow:0px 5px 18px rgba(0,0,0,0.35);
        font-family:Inter,Segoe UI,sans-serif;
        ">

        <h2 style="
        color:white;
        font-size:32px;
        font-weight:700;
        margin-bottom:10px;
        ">
        🎯 Prediction Result
        </h2>

        <hr style="
        border:1px solid #3D4A59;
        margin-top:10px;
        margin-bottom:18px;
        ">

        <h3 style="
        color:{color};
        font-size:38px;
        font-weight:800;
        margin-bottom:15px;
        ">        
        {cluster_names.get(cluster)}
        </h3>

        <p style="
        font-size:19px;
        font-weight:500;
        line-height:1.8;        
        color:#DCE6F1;
        margin-bottom:8px;
        ">       
        <b>Cluster ID :</b> {cluster}
        </p>

        <p style="
        font-size:19px;
        font-weight:500;
        line-height:1.8;
        color:#DCE6F1;
        margin-bottom:8px;
        ">       
        <b>Prediction Method :</b> K-Means Clustering
        </p>

        <p style="
        font-size:19px;
        font-weight:500;
        line-height:1.8;
        color:#DCE6F1;
        margin-bottom:8px;
        ">      
        <b>Model Status :</b> ✅ Successfully Predicted
        </p>

        </div>
        """, unsafe_allow_html=True)

        # ======================================
        # Customer Health Score
        # ======================================

        health_score = {

            4:100,
            2:90,
            3:75,
            0:60,
            1:20

        }.get(cluster,50)

        fig = go.Figure(go.Indicator(

            mode="gauge+number",

            value=health_score,

            number={'suffix':"%"},

            title={'text':"<b>Customer Health Score</b>"},

            gauge={

                'axis':{'range':[0,100]},

                'bar':{'color':color},

                'steps':[

                    {'range':[0,30],'color':'#5A1E1E'},

                    {'range':[30,60],'color':'#9E6B00'},

                    {'range':[60,85],'color':'#355E3B'},

                    {'range':[85,100],'color':'#1B5E20'}

                ]

            }

        ))

        fig.update_layout(

            height=300,

            margin=dict(l=20,r=20,t=60,b=20),

            paper_bgcolor="#0E1117",

            font=dict(color="white")

        )

        left, spacer, right = st.columns([2.1, 0.08, 1])
        with left:

            st.plotly_chart(
                fig,
                use_container_width=False
            )

        with spacer:
            st.empty()

        with right:
             
            st.markdown(
                "<div style='height:20px'></div>", 
                unsafe_allow_html=True
            )
            
            st.markdown(f"""
            <div style="
            background:#1E2D3D;
            border-radius:18px;
            padding:25px;
            min-height:260px;
            margin-top:25px;
            width:100%;
            display:flex;
            flex-direction:column;
            justify-content:center;
            align-items:center;
            ">

            <h2 style="
            color:white;
            font-size:24px;
            margin-bottom:25px;
            ">
            Customer Segment
            </h2>

            <h1 style="
            color:{color};
            font-size:28px;
            font-weight:800;
            text-align:center;
            line-height:1.3;
            ">
            {cluster_names.get(cluster)}
            </h1>

            <h3 style="
            color:white;
            font-size:22px;
            margin-top:20px;
            ">
            Score : {health_score}%
            </h3>

            </div>
            """, unsafe_allow_html=True)

        segment_description = {

        0:"Regular customers purchase occasionally and respond well to personalized offers.",

        1:"Inactive customers have not purchased recently and should be targeted with win-back campaigns.",

        2:"Elite customers are premium buyers with high purchasing power.",

        3:"High Value customers purchase frequently and spend above average.",

        4:"Top customers are the most valuable customers with exceptional loyalty."
  
        }

        left, right = st.columns([2,1])

        with left:
            st.markdown(f"""
            <div style="
            background:#1E2D3D;
            padding:15px;
            border-radius:12px;
            border-left:5px solid {color};
            margin-top:15px;
            ">
            <p style="
            color:#DCE6F1;
            font-size:17px;
            margin:0;
            ">
            {segment_description[cluster]}
            </p>
            </div>
            """, unsafe_allow_html=True)

        with right:
            st.empty()
            
        st.markdown(f"""
        <div style="
        background:{color};
        padding:15px;
        border-radius:12px;
        text-align:center;
        color:white;
        font-size:24px;
        font-weight:bold;
        margin-top:10px;
        margin-bottom:15px;
        ">

        🏆 {cluster_names[cluster]}

        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("## 📋 Prediction Summary")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div style="
            background:#1E2D3D;
            padding:22px;
            border-radius:15px;
            text-align:center;
            border:1px solid rgba(139,92,246,.25);
            box-shadow:0 6px 18px rgba(0,0,0,.25);
            min-height:145px;
            font-family:Inter,Segoe UI,sans-serif;
            ">
                        
            <h4>📅 Recency</h4>
                        
            <div style="
            font-size:30px;
            font-weight:700;
            color:#4DA3FF;
            margin-top:10px;
            ">
            {recency} Days

            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div style="
            background:#1E2D3D;
            padding:22px;
            border-radius:15px;
            text-align:center;
            border:1px solid rgba(139,92,246,.25);
            box-shadow:0 6px 18px rgba(0,0,0,.25);
            min-height:145px;
            font-family:Inter,Segoe UI,sans-serif;
            ">
                        
            <h4>🛒 Frequency</h4>
            <div style="
            font-size:30px;
            font-weight:700;
            color:#FFB74D;
            margin-top:12px;
            ">
            {frequency}

            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div style="
            background:#1E2D3D;
            padding:22px;
            border-radius:15px;
            text-align:center;
            border:1px solid rgba(139,92,246,.25);
            box-shadow:0 6px 18px rgba(0,0,0,.25);
            min-height:145px;
            font-family:Inter,Segoe UI,sans-serif;
            ">
                        
            <h4>💰 Monetary</h4>
            <div style="
            font-size:30px;
            font-weight:700;
            color:#22C55E;
            margin-top:12px;
            ">
            £{monetary:,.2f}

            </div>            
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div style="
            background:#1E2D3D;
            padding:24px;
            border-radius:18px;
            text-align:center;
            border:1px solid rgba(139,92,246,.25);
            box-shadow:0 6px 18px rgba(0,0,0,.25);
            min-height:145px;
            font-family:Inter,Segoe UI,sans-serif;
            ">
                        
            <h4>🏆 Segment</h4>
            <div style="
            font-size:26px;
            font-weight:700;
            color:{color};
            margin-top:12px;
            line-height:1.4;
            ">
            {cluster_names.get(cluster)}

            </div>            
            """, unsafe_allow_html=True)

        st.markdown("---")

        recommendations = {
            0: """
        <li>🟢 Regular Customers</li>
        <li>🎁 Reward with loyalty points</li>
        <li>📧 Send personalized offers</li>
        <li>💎 Introduce premium membership</li>
        """,

            1: """
        <li>🔴 Inactive Customers</li> 
        <li>📩 Send win-back email campaigns</li>
        <li>🎟 Offer discount coupons</li>
        <li>🔥 Recommend trending products</li>
        """,

            2: """
        <li>💎 Elite Customers</li> 
        <li>👑 Provide VIP benefits</li>
        <li>🚀 Early access to new arrivals</li> 
        <li>☎ Priority customer support</li>
        """,

            3: """
        <li>⭐ High Value Customers</li>
        <li>🛍 Cross-sell premium products</li>
        <li>🎁 Offer bundle discounts</li>
        <li>🎉 Invite to special promotions</li>
        """,

            4: """
        <li>👑 Top Customers</li> 
        <li>👨‍💼 Assign account manager</li> 
        <li>💎 Exclusive membership benefits</li> 
        <li>⭐ Priority customer service</li>
        """
        }

        st.markdown(f"""
        <div style="
        background:linear-gradient(135deg,#243447,#1A2634);
        padding:22px;
        border-radius:15px;
        border-left:8px solid {color};
        margin-top:10px;
        width:100%;
        font-family:Inter,Segoe UI,sans-serif;
        ">

        <h2 style="color:white;">
        💡 Business Recommendation
        </h2>

        <hr>

        <ul style="
        margin-bottom:0;
        padding-left:25px;
        line-height:2;
        font-size:18px;
        color:#E6EEF8;
        ">
        {recommendations.get(cluster)}
        <ul>

        </div>
        """, unsafe_allow_html=True)

# ============================
# Product Recommendation
# ============================
if page == "🎯 Product Recommendation":

    st.markdown(f"""
    <div style="
    background: {PRIMARY_GRADIENT};    
    padding:32px;
    border-radius:20px;
    text-align:center;
    color:white;
    margin-bottom:30px;
    box-shadow:0 8px 30px rgba(91,61,245,.35);
    ">

    <h1 style="
    font-size:42px;
    font-weight:800;
    ">
    🎯 Product Recommendation
    </h1>

    <h4 style="
    font-size:24px;
    font-weight:600;
    ">
    Find Similar Products using Cosine Similarity
    </h4>
    Find Similar Products using Cosine Similarity
    </h4>

    <p>
    Product-to-Product Recommendation Engine
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.info(
        "🔍 Select a product below to discover the Top 5 most similar products using our AI recommendation engine."
    )

    product_list = sorted(
        df_clean["Description"]
        .dropna()
        .unique()
    )

    selected_product = st.selectbox(
        "🛍 Select Product",
        product_list,
        index=(
                product_list.index(st.session_state.selected_product)
                if "selected_product" in st.session_state
                and st.session_state.selected_product in product_list
                else None
            ),
        placeholder="Search a product...",
        key="selected_product"  
    )

    if selected_product:

        selected_code =(
            df_clean.loc[
                df_clean["Description"] == selected_product,
                "StockCode"
            ]    
            .drop_duplicates()
            .iloc[0]
        )

    recommend = st.button(
        "🎯 Recommend Products",
        use_container_width=True
    )

    # ==========================================
    # Generate Recommendation
    # ==========================================

    if recommend:
        if not selected_product:
            st.warning("⚠ Please select a product first.")
            st.stop()

        recommendations = (
            similarity_df[selected_code] \
            .sort_values(ascending=False)[1:6]
        )    

        recommendation_df = recommendations.reset_index()

        recommendation_df.columns = [
            "Recommended Product",
            "Similarity Score"
        ]
         
        recommendation_df["Similarity Score"] = (
                recommendation_df["Similarity Score"] * 100
            ).round(2)
        
        st.session_state.recommendation_df = recommendation_df
        st.session_state.selected_code = selected_code
        st.session_state.selected_product_name = selected_product

        st.markdown("---")

    # =====================================
    # Restore Previous Recommendation
    # =====================================

    if "recommendation_df" in st.session_state:

        recommendation_df = st.session_state.recommendation_df
        selected_code = st.session_state.selected_code
        selected_product = st.session_state.selected_product_name

        st.markdown("---")

        st.markdown(f"""
        <div style="
        background:#243447;
        padding:25px;
        border-radius:15px;
        border-left:8px solid #EC4899;
        font-family:Inter,Segoe UI,sans-serif;        
        ">

        <h2 style="
        color:white;
        font-size:32px;
        font-weight:700;
        ">
        🎯 Recommendation Result
        </h2>

        <hr>

        <h3 style="
        color:#EC4899;
        font-size:30px;
        font-weight:700">
        Selected Product
        </h3>

        <p style="
        font-size:20px;
        font-weight:500;
        line-height:1.8;
        color:white;
        ">
        {selected_product}
        </p>

        <p style="
        font-size:20px;
        font-weight:500;
        line-height:1.8;
        color:white;
        ">
        <b>Product Code :</b> {selected_code}
        </p>

        <p style=
        "font-size:20px;
        font-weight:500;
        line-height:1.8;
        color:white;
        ">
        <b>Recommendation Method :</b> Cosine Similarity
        </p>

        <p style="
        font-size:18px;
        font-height:500;
        line-height:1.8;
        color:white;
        ">
        <b>Status :</b> ✅ Success
        </p>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("## 🏆 Top 5 Recommendations")

        rank_icons = ["🥇", "🥈", "🥉", "🏅", "⭐"]

        for i, row in recommendation_df.iterrows():

            # Get Product Name from StockCode
            product_match = df.loc[
                df["StockCode"] == row["Recommended Product"],
                "Description"
            ].dropna().drop_duplicates()

            if not product_match.empty:
                recommended_name = product_match.iloc[0]
            else:
                recommended_name = "Unknown Product"

            st.markdown(f"""
            <div style="
            background:#1E2D3D;
            padding:18px;
            border-radius:12px;
            margin-bottom:15px;
            border-left:6px solid #8B5CF6;            
            ">

            <h4 style="color:white;">
            {rank_icons[i]} Recommendation {i+1}            
            </h4>

            <p style="font-size:18px;color:white;">
            <b>Product :</b> {recommended_name}
            </p>

            <p style="font-size:16px;color:#CFCFCF;">
            <b>Product Code :</b> {row['Recommended Product']}
            </p>

            <p style="font-size:18px;color:#FFD166;">
            Similarity Score : {row['Similarity Score']}%
            </p>

            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📊 Recommendation Score Analysis")    

        fig = px.bar(
            recommendation_df,
            x="Recommended Product",
            y="Similarity Score",
            color="Similarity Score",
            color_continuous_scale=CHART_COLORS,
            text="Similarity Score",
            title="Top 5 Recommended Products"       
        )

        fig.update_layout(
            height=500,
            template="plotly_dark",
            xaxis_title="Recommended Products",
            yaxis_title="Similarity Score (%)",
            coloraxis_showscale=False
        )

        fig.update_layout(
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            font_color=TEXT,
            title_font_size=22,
            title_x=0.02,
            margin=dict(l=20, r=20, t=50, b=20),
            bargap=0.30
        )

        fig.update_xaxes(
            tickangle=-30
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
            marker_line_color="white",
            marker_line_width=1,
            hovertemplate=
                "<b>%{x}</b><br>"
                "Similarity Score : %{y:.2f}%<br>"
                "Recommendation Rank : %{text}<extra></extra>" 
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        st.markdown("---")

        st.success(f"""
    ### ✅ Recommendation Summary

    - Selected Product : **{selected_product}**
    - Product Code : **{selected_code}**
    - Recommendation Model : **Cosine Similarity**
    - Products Recommended : **5**
    - Recommendation Status : **Successful**
    """)

# ============================
# Business Insights
# ============================

elif page == "📈 Business Insights":

    st.markdown(f"""
    <div style="
    background:{PRIMARY_GRADIENT};
    padding:32px;
    border-radius:20px;
    text-align:center;
    color:white;
    margin-bottom:30px;
    box-shadow:0 8px 30px rgba(91,61,245,.35);
    ">

    <h1>📈 Business Insights</h1>

    <h4>
    Executive Summary & Strategic Recommendations
    </h4>

    <p>
    Customer Intelligence • Business Strategy • Revenue Optimization
    <p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Key Findings")

    st.success("""
    ✅ Premium and VIP customers contribute the highest revenue.

    ✅ Regular customers form the largest customer segment.

    ✅ At Risk customers require targeted retention campaigns.

    ✅ Product recommendations can increase cross-selling opportunities.

    ✅ Understanding customer behavior helps improve marketing effectiveness.
    """)

    st.subheader("Business Recommendations")

    st.info("""
    • Reward Premium and VIP customers with exclusive offers.

    • Launch personalized email campaigns for At Risk customers.

    • Bundle frequently purchased products together.

    • Use customer segments for targeted promotions.

    • Maintain stock for the most recommended products.
    """)

st.markdown("---")

st.markdown("""
<div style="
text-align:center;
padding:18px;
color:#9CA3AF;
font-size:15px;
">

🛍 <b>Shopper Spectrum</b> |
Customer Analytics & Product Recommendation Dashboard

<br><br>

Built with ❤️ using
<b>Python</b> •
<b>Streamlit</b> •
<b>Plotly</b> •
<b>Scikit-Learn</b>

</div>
""", unsafe_allow_html=True)