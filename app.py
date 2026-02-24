import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the "Brains" of the project
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans.pkl')

# 2. Set up the Website Interface
st.set_page_config(page_title="BA Booking Predictor", layout="wide")
st.title("✈️ Airways: Customer Segmentation & Booking Predictor")
st.markdown("Enter customer details below to see which **Segment** they belong to and their **Booking Probability**.")

# 3. Create Input Fields for the user
col1, col2 = st.columns(2)

with col1:
    st.subheader("Travel Details")
    passengers = st.slider("Number of Passengers", 1, 10, 1)
    lead_time = st.number_input("Purchase Lead (Days before flight)", 0, 600, 30)
    stay_duration = st.number_input("Length of Stay (Days)", 0, 100, 7)

with col2:
    st.subheader("Flight Context")
    flight_dur = st.slider("Flight Duration (Hours)", 4.0, 15.0, 7.5)
    hour = st.slider("Flight Hour (0-23)", 0, 23, 12)
    extra_baggage = st.selectbox("Wants Extra Baggage?", ["No", "Yes"])

# 4. The Prediction Logic
if st.button("Analyze This Customer"):
    # A. Process inputs for Segmentation (using the 4 features used in K-Means)
    seg_input = np.array([[passengers, lead_time, stay_duration, flight_dur]])
    scaled_input = scaler.transform(seg_input)
    segment_id = kmeans.predict(scaled_input)[0]
    
    # Map Segment IDs to Names
    segment_names = {0: "Big Spenders", 1: "Bargain Hunters", 2: "Occasional Shoppers"}
    current_segment = segment_names.get(segment_id, "Standard Traveler")

    # B. Prepare data for the XGBoost Predictor
    # FIX: Changed from 13 to 14 to match the model's training shape
    final_features = np.zeros((1, 14)) 
    
    final_features[0, 0] = passengers      # num_passengers
    final_features[0, 1] = 1               # sales_channel (assuming 'Internet' as default)
    final_features[0, 2] = lead_time       # purchase_lead
    final_features[0, 3] = stay_duration   # length_of_stay
    final_features[0, 4] = hour            # flight_hour
    final_features[0, 5] = 0               # flight_day (Monday = 0)
    final_features[0, 6] = 0               # route (placeholder index)
    final_features[0, 7] = 0               # booking_origin (placeholder index)
    final_features[0, 8] = 0               # trip_type (placeholder index)
    final_features[0, 9] = 1 if extra_baggage == "Yes" else 0
    final_features[0, 10] = 0              # wants_preferred_seat (Missing in your current input!)
    final_features[0, 11] = 0              # wants_in_flight_meals (Missing in your current input!)
    final_features[0, 12] = flight_dur     # flight_duration
    final_features[0, 13] = 0              # extra feature from your Task2 notebook
    
    # Run the Prediction
    prediction_prob = model.predict_proba(final_features)[0][1]
    
    # 5. Display the Results
    st.divider()
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric("Detected Segment", current_segment)
        st.info(f"This customer behaves like a '{current_segment}' based on their booking patterns.")

    with res_col2:
        st.metric("Booking Probability", f"{prediction_prob:.1%}")
        if prediction_prob > 0.5:
            st.success("High Probability: This customer is likely to complete the booking!")
        else:
            st.warning("Low Probability: This customer is just browsing.")