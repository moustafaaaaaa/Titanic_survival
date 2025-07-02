import streamlit as st
import numpy as np
import pickle

# Load model and scaling values
model = pickle.load(open('titanic_model.pkl', 'rb'))
age_max, fare_min, fare_max = pickle.load(open('scaling_values.pkl', 'rb'))

st.title("🚢 Titanic Survival Predictor")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ['male', 'female'])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare Paid", 0.0, 600.0, 30.0)
embarked = st.selectbox("Port of Embarkation", ['S', 'C', 'Q'])

# Encode features
pclass_2 = 1 if pclass == 2 else 0
pclass_3 = 1 if pclass == 3 else 0
sex_male = 1 if sex == 'male' else 0
embarked_q = 1 if embarked == 'Q' else 0
embarked_s = 1 if embarked == 'S' else 0

# Scale Age and Fare
age_scaled = age / age_max
fare_scaled = (fare - fare_min) / (fare_max - fare_min)

# Prepare final input
input_data = (pclass_2, pclass_3, age_scaled, sibsp, parch,
              fare_scaled, sex_male, embarked_q, embarked_s)

input_array = np.asarray(input_data).reshape(1, -1)

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_array)
    if prediction[0] == 0:
        st.error("🚫 The passenger did NOT survive.")
    else:
        st.success("✅ The passenger SURVIVED!")
