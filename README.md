# ✈️ Airways Customer Segmentation & Booking Predictor  

> End-to-End Machine Learning Project combining Customer Review Analysis, Segmentation & Booking Prediction

---

## 📌 Project Overview

This project analyzes airline customer reviews and booking behavior to:

- 📊 Understand customer satisfaction patterns  
- 👥 Segment customers using K-Means Clustering  
- 🎯 Predict booking probability using a Machine Learning Model  
- 🌐 Deploy an interactive Streamlit Web Application  

The goal is to simulate how an airline company can:
- Identify high-value customers  
- Detect bargain hunters  
- Improve conversion rates  
- Personalize marketing strategies  

---

## 🚀 Run the Application

This project includes a fully working Streamlit Web App.

### Run Locally:

```bash
streamlit run app.py
```

The app:
- Detects customer segment
- Predicts booking probability
- Displays business-friendly insights

---

## 📂 Project Structure

```
├── Airways_Task1.ipynb        # Web scraping & review analysis
├── Airways_Task2.ipynb        # ML modeling & segmentation
├── app.py                     # Streamlit web application
├── model.pkl                  # Booking prediction model
├── kmeans.pkl                 # Customer segmentation model
├── scaler.pkl                 # Feature scaling object
├── README.md
```

---

## 🧠 Project Components

### 1️⃣ Customer Review Analysis

- Scraped airline reviews
- Cleaned & structured rating features
- Generated WordClouds for:
  - Recommended Reviews
  - Not Recommended Reviews
- Analyzed:
  - Seat Comfort
  - Cabin Staff Service
  - Food & Beverages
  - Inflight Entertainment
  - WiFi & Connectivity
  - Value for Money

### Key Insights:
- First Class & Business Class have highest ratings
- WiFi & Value for Money are major complaint areas
- Strong relationship between service ratings and recommendation

---

### 2️⃣ Customer Segmentation (K-Means)

Used features:
- Number of passengers  
- Purchase lead time  
- Stay duration  
- Flight duration  

Customers grouped into:

| Segment | Description |
|----------|------------|
| Big Spenders | Early planners, longer stays |
| Bargain Hunters | Short lead time, price sensitive |
| Occasional Shoppers | Moderate booking behavior |

Business Use:
- Targeted promotions
- Premium upsell offers
- Personalized campaigns

---

### 3️⃣ Booking Prediction Model

Built a classification model to predict:

> Will the customer complete the booking?

- Trained on structured booking dataset
- Uses predict_proba() for probability output
- Integrated into Streamlit app

Example Output:

Segment: Big Spenders  
Booking Probability: 78%  
Status: High chance of conversion  

---

## 📊 Exploratory Data Analysis

Visualizations include:

- Rating Distribution
- Recommended vs Not Recommended
- Seat Type vs Average Rating
- WordCloud Analysis
- Service Feature Distribution

These insights explain:
- What drives satisfaction
- What affects recommendation
- Where improvement is needed

---

## 🛠️ Tech Stack

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- WordCloud
- Scikit-Learn
- XGBoost (or ML Classifier)
- Streamlit
- Joblib

---

## 📈 Business Impact

This system helps airlines:

- Identify high-conversion customers  
- Improve marketing ROI  
- Reduce abandoned bookings  
- Personalize customer experience  
- Make data-driven decisions  

---

## 🔧 Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Ashutoshrana434/Airways-Customer-Segmentation-Booking-Predictor.git
cd Airways-Customer-Segmentation-Booking-Predictor
```

### 2️⃣ Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit wordcloud joblib
```

### 3️⃣ Run Application

```bash
streamlit run app.py
```

---

## 📌 Future Improvements

- Deploy on Streamlit Cloud / Render
- Add SHAP explainability
- Add interactive dashboard
- Add NLP sentiment analysis
- Improve feature engineering

---

## 👨‍💻 Author

Ashutosh Rana  
MCA | Data Science Enthusiast  
Focused on ML, Customer Analytics & End-to-End AI Systems  

---

## ⭐ Why This Project Is Resume-Strong

This project demonstrates:

- Real-world business problem solving
- Unsupervised Learning (Clustering)
- Supervised Learning (Classification)
- Model Deployment
- Feature Engineering
- Business Interpretation
- End-to-End ML Pipeline
