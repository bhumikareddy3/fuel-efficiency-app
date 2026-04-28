import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model


# Page config
st.set_page_config(page_title="Fuel Efficiency Predictor", page_icon="🚗")

# Load model
model = load_model("model.keras", compile=False)
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title
st.title("🚗 Fuel Efficiency Prediction App")
st.markdown("Predict **Miles Per Gallon (MPG)** based on car features using a Machine Learning model.")

# Sidebar inputs
st.sidebar.header("🔧 Enter Car Details")

cylinders = st.sidebar.slider("Cylinders", 3, 8, 4)
horsepower = st.sidebar.slider("Horsepower", 40, 250, 100)
weight = st.sidebar.slider("Weight (lbs)", 1500, 5000, 3000)
acceleration = st.sidebar.slider("Acceleration", 8.0, 25.0, 15.0)
model_year = st.sidebar.slider("Model Year", 70, 82, 76)
origin = st.sidebar.selectbox("Origin", ["USA(1)", "Europe(2)", "Japan(3)"])

# Convert origin to numeric
origin_map = {"USA(1)": 1, "Europe(2)": 2, "Japan(3)": 3}
origin_val = origin_map[origin]

# Input array
input_data = np.array([[cylinders, horsepower, weight, acceleration, model_year, origin_val]])
input_data = scaler.transform(input_data)

# Predict
if st.button("🔍 Predict MPG"):
    prediction = model.predict(input_data)[0][0]

    st.success(f"🚀 Estimated Fuel Efficiency: **{prediction:.2f} MPG**")

    # Interpretation
    if prediction > 25:
        st.info("✅ This car is fuel efficient.")
    elif prediction > 15:
        st.warning("⚖️ This car has moderate efficiency.")
    else:
        st.error("❌ This car consumes more fuel.")

# Footer
st.markdown("---")
st.markdown("💡 Built using TensorFlow & Streamlit")