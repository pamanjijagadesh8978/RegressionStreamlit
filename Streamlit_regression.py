import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# Load saved encoders and model
label_encoder_gender = pickle.load(open("label_encoder_gender.pkl", "rb"))
onehot_encoder_geo = pickle.load(open("onehot_encoder_geo.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
model = tf.keras.models.load_model("regression_model.h5")

st.title("Customer Salary Prediction App 💰")
st.write("Enter customer details below to predict their **Estimated Salary**.")

# Input fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.slider("Age", min_value=18, max_value=100, value=35)
tenure = st.slider("Tenure (Years with Bank)", 0, 10, 3)
balance = st.number_input("Balance", min_value=0.0, max_value=250000.0, value=10000.0)
num_of_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card?", ['Yes', 'No'])
is_active_member = st.selectbox("Is Active Member?", ['Yes', 'No'])

# Convert categorical to numeric
gender_encoded = label_encoder_gender.transform([gender])[0]
geography_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

# Other features
has_cr_card = 1 if has_cr_card == 'Yes' else 0
is_active_member = 1 if is_active_member == 'Yes' else 0

# Combine all features
features = np.array([[credit_score, gender_encoded, age, tenure, balance, num_of_products,
                      has_cr_card, is_active_member]])
features = np.concatenate([features, geography_encoded], axis=1)

# Scale features
features_scaled = scaler.transform(features)

# Predict
if st.button("Predict Salary"):
    predicted_salary = model.predict(features_scaled)[0][0]
    st.success(f"💸 Predicted Estimated Salary: ₹{predicted_salary:,.2f}")