import streamlit as st
import pandas as pd
import pickle

cars = pd.read_csv('cleaned_cars.csv')

with open("carPricePredictionModel.pkl", "rb") as file:
    lr_model = pickle.load(file)

st.title("🚗 Old Car Reselling Price Prediction")

st.header("Input Features")

company = st.selectbox("Select Company", sorted(cars['company'].unique()))
model = st.selectbox("Select Model", cars[cars['company'] == company]['name'].unique())
year = st.selectbox("Select Year", sorted(cars['year'].unique(), reverse=True))
fuel = st.selectbox("Select Fuel Type", sorted(cars['fuel_type'].unique()))
kms = st.number_input("Enter Kilometers Driven", min_value=0, step=1000)

if st.button("Predict Price"):
    try:
        # Preparing the input for prediction
        input_data = pd.DataFrame([[company, model, year, fuel, kms]], 
                          columns=['company', 'name', 'year', 'fuel_type', 'kms_driven'])

        # Predicting the price
        predicted_price = lr_model.predict(input_data)[0]

        st.write(f"### Predicted Price: INR {predicted_price:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")