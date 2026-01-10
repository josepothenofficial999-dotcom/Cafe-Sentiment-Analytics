# ☕ Cafe Sentiment Analysis and Business Insights

## 📌 Project Overview
This project analyzes customer reviews from local cafés to extract meaningful business insights.  
By combining data analysis, sentiment logic, and an interactive dashboard, the project helps café owners understand customer satisfaction trends and identify improvement areas.

The focus of this project is **Business & Marketing Analytics**, not just model accuracy.

---

## 🎯 Problem Statement
Cafés receive a large number of customer reviews across platforms, but manually reading and interpreting them is inefficient.  
This project aims to:
- Analyze customer feedback at scale
- Identify sentiment patterns
- Provide actionable insights for business improvement

---

## 🛠️ Tools & Technologies Used
- **Python**
- **Pandas** (Data processing)
- **Streamlit** (Interactive dashboard)
- **VS Code**
- **CSV Dataset (10,000+ reviews)**

---

## 📊 Dataset Description
The dataset contains customer reviews with the following fields:
- `review_id`
- `rating`
- `review_text`
- `review_date`
- `branch_name`

Sentiment for business insights is derived from ratings:
- Ratings ≥ 4 → **Positive**
- Ratings < 4 → **Negative or Neutral**

---

## 🔍 Methodology
1. Loaded and cleaned customer review data  
2. Derived business sentiment using rating-based logic  
3. Built interactive filters (branch, sentiment)  
4. Generated key metrics and visualizations  
5. Interpreted results to provide business insights  

---

## 📈 Key Insights
- Customer satisfaction varies significantly across café branches  
- Some branches show higher negative or neutral sentiment  
- Ratings serve as a reliable proxy for customer sentiment  
- Business owners can prioritize improvements based on branch-level feedback  

---

## 🖥️ Dashboard Features
- Branch-wise sentiment filtering  
- Sentiment distribution visualization  
- Key performance metrics (total reviews, positive percentage)  
- Clean, interactive Streamlit interface  

---

