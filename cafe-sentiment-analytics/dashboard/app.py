import streamlit as st
import pandas as pd
import os

# =====================
# Base directory
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Page configuration
st.set_page_config(page_title="Cafe Sentiment Dashboard", layout="wide")

# Title
st.title("☕ Cafe Sentiment Analysis Dashboard")
st.write("Business insights from customer reviews")

# =====================
# Load main dataset
# =====================
@st.cache_data
def load_data():
    data_path = os.path.join(BASE_DIR, "..", "data", "cafe_reviews_10000.csv")
    return pd.read_csv(data_path)

df = load_data()

# =====================
# Business sentiment from rating
# =====================
df["sentiment_business"] = df["rating"].apply(
    lambda x: "positive" if x >= 4 else "negative_or_neutral"
)

# =====================
# Sidebar Filters
# =====================
st.sidebar.header("Filters")

branch_options = ["All"] + sorted(df["branch_name"].unique().tolist())
selected_branch = st.sidebar.selectbox("Select Cafe Branch", branch_options)

sentiment_options = ["All", "positive", "negative_or_neutral"]
selected_sentiment = st.sidebar.selectbox("Select Sentiment", sentiment_options)

# Apply filters
filtered_df = df.copy()

if selected_branch != "All":
    filtered_df = filtered_df[filtered_df["branch_name"] == selected_branch]

if selected_sentiment != "All":
    filtered_df = filtered_df[
        filtered_df["sentiment_business"] == selected_sentiment
    ]

# =====================
# Key Metrics
# =====================
st.subheader("📊 Key Metrics")

total_reviews = len(filtered_df)
positive_pct = (
    (filtered_df["sentiment_business"] == "positive").mean() * 100
    if total_reviews > 0
    else 0
)

col1, col2 = st.columns(2)
col1.metric("Total Reviews", total_reviews)
col2.metric("Positive Reviews (%)", f"{positive_pct:.1f}%")

# =====================
# Sentiment Distribution
# =====================
st.subheader("Sentiment Distribution")

sentiment_counts = filtered_df["sentiment_business"].value_counts()
st.bar_chart(sentiment_counts)

# =====================
# Branch-wise Comparison
# =====================
st.subheader("Branch-wise Sentiment Comparison")

branch_sentiment = (
    df.groupby(["branch_name", "sentiment_business"])
    .size()
    .unstack(fill_value=0)
)

st.dataframe(branch_sentiment)

# =====================
# Business Insight
# =====================
st.info(
    "Insight: Customer dissatisfaction (negative or neutral sentiment) varies across cafe branches. "
    "Branches with consistently lower positive sentiment may require improvements in service quality, "
    "staff responsiveness, or menu offerings."
)

# =====================
# ML vs Business Sentiment (SAFE COMPARISON)
# =====================
st.subheader("🤖 ML vs Business Sentiment (Comparison)")

ml_path = os.path.join(BASE_DIR, "..", "outputs", "ml_predictions.csv")

if os.path.exists(ml_path):
    ml_df = pd.read_csv(ml_path)

    ml_agreement = (ml_df["ml_correct"].mean()) * 100

    st.metric(
        "ML Agreement with Rating-Based Sentiment",
        f"{ml_agreement:.1f}%"
    )

    st.info(
        "The ML model is used as a comparative tool. "
        "Lower agreement highlights the limitations of text-only sentiment models "
        "and reinforces why ratings are more reliable for business decisions."
    )

    st.dataframe(
        ml_df[["review_text", "true_sentiment", "predicted_sentiment"]].head(10)
    )
else:
    st.warning(
        "ML results not found. Run the ML notebook to generate "
        "`outputs/ml_predictions.csv`."
    )

# =====================
# Dataset Preview
# =====================
st.subheader("Dataset Preview")

st.dataframe(
    df[
        ["review_id", "rating", "review_text", "branch_name", "review_date"]
    ].head()
)