import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import pickle #convert python objects into bytestream and back
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import os
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Load models and scalers
def load_models():
    with open(r'D:\Projects\Project1\jupyter\random_forest_crop_recommendation_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    
    with open(r'D:\Projects\Project1\jupyter\weather_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    lstm_model = load_model(r'D:\Projects\Project1\jupyter\crop.h5')
    return rf_model, scaler, lstm_model

def predict_future(user_date, scaler, model, original_data, time_step=30):
    user_date = pd.to_datetime(user_date)
    last_date = original_data['date'].max() 
    last_date = last_date.tz_localize(None)

    days_to_predict = (user_date - last_date).days

    if days_to_predict <= 0:
        raise ValueError("The date you are trying to predict is before or equal to the last date in the dataset.")

    latest_data = original_data.tail(time_step)

    scaled_latest_data = scaler.transform(latest_data[['temperature_2m', 'relative_humidity_2m', 'rain']])

    X_input = scaled_latest_data.reshape(1, time_step, 3)

    predicted_values = []
    for i in range(days_to_predict):
        prediction = model.predict(X_input)
        predicted_values.append(prediction)

        X_input = np.append(X_input[:, 1:, :], prediction.reshape(1, 1, 3), axis=1)

    final_prediction = predicted_values[-1]

    final_prediction_inverse = scaler.inverse_transform(final_prediction)

    future_temperature = final_prediction_inverse[0][0]
    future_humidity = final_prediction_inverse[0][1]
    future_rain = final_prediction_inverse[0][2]

    return future_temperature, future_humidity, future_rain

def recommend_crop(n, p, k, ph, date, scaler, lstm_model, original_data, rf_model, time_step=30):
    temperature, humidity, rainfall = predict_future(date, scaler, lstm_model, original_data, time_step)
    input_features = np.array([[n, p, k, temperature, humidity, ph]])
    
    # Get probabilities for all crops
    probabilities = rf_model.predict_proba(input_features)[0]
    crop_classes = rf_model.classes_
    
    # Find top 3 crops
    top_indices = np.argsort(probabilities)[-3:][::-1]
    top_crops = [(crop_classes[i], probabilities[i]) for i in top_indices]

    return top_crops

crop_image_paths = {
    'apple': 'D:\Projects\Project1\crop img\apple.jpg',
    'banana': 'D:\Projects\Project1\crop img\banana.jpg',
    'blackgram': 'D:\Projects\Project1\crop img\blackgram.jpg',
    'chickpea': 'D:\Projects\Project1\crop img\chickpea.jpg',
    'coconut': 'D:\Projects\Project1\crop img\coconut.jpg',
    'coffee': 'D:\Projects\Project1\crop img\coffee.jpg',
    'cotton': 'D:\Projects\Project1\crop img\cotton.jpg',
    'grapes': 'D:\Projects\Project1\crop img\grapes.jpg',
    'jute': 'D:\Projects\Project1\crop img\jute.jpg',
    'kidneybeans': 'D:\Projects\Project1\crop img\kidneybean.jpeg',
    'mango': 'D:\Projects\Project1\crop img\mango.jpg',
    'lentil': 'D:\Projects\Project1\crop img\lentil.jpeg',
    'maize': 'D:\Projects\Project1\crop img\maize.jpeg',
    'mothbeans': 'D:\Projects\Project1\crop img\mothbeans.jpg',
    'mungbean': 'D:\Projects\Project1\crop img\mungbean.jpg',
    'muskmelon': 'D:\Projects\Project1\crop img\muskmelon.jpg',
    'orange': 'D:\Projects\Project1\crop img\orange.jpg',
    'papaya': 'D:\Projects\Project1\crop img\papaya.jpeg',
    'pigeonpeas': 'D:\Projects\Project1\crop img\pigeonpeas.jpg',
    'rice': 'D:\Projects\Project1\crop img\rice.jpg',
    'watermelon': 'D:\Projects\Project1\crop img\watermelon.jpg'
}

# Streamlit UI
def main():
    st.title("Smart Crop Recommendation System")

    st.sidebar.header("Input Parameters")
    n = st.sidebar.slider("Nitrogen (N)", 0, 140, 50)
    p = st.sidebar.slider("Phosphorus (P)", 0, 145, 40)
    k = st.sidebar.slider("Potassium (K)", 0, 205, 60)
    ph = st.sidebar.slider("pH Value", 0.0, 14.0, 6.5)
    date = st.sidebar.date_input("Select a date for prediction", datetime(2024, 8, 8))

    # Load models
    rf_model, scaler, lstm_model = load_models()
    
    # Load historical data (replace 'data.csv' with your dataset path)
    data = pd.read_csv('D:\Projects\Project1\dataset\Aligarh.csv', parse_dates=['date'])
    
    # Predict weather conditions
    try:
        temperature, humidity, rainfall = predict_future(date, scaler, lstm_model, data)
        st.write(f"Predicted Weather on {date}:")
        st.write(f"Temperature: {temperature:.2f}°C")
        st.write(f"Humidity: {humidity:.2f}%")
        if rainfall < 0:
            st.write(f"Rainfall: {00:.2f} mm")
        else:
            st.write(f"Rainfall: {rainfall:.2f} mm")
    except ValueError as e:
        st.error(str(e))
        return

    # Recommend crops
    top_crops = recommend_crop(n, p, k, ph, date, scaler, lstm_model, data, rf_model)
    
    st.write("Top 3 Recommended Crops for the given conditions:")
    for rank, (crop, prob) in enumerate(top_crops, 1):
        st.write(f"{rank}. {crop.capitalize()} (Probability: {prob:.2f})")
        
        # Display crop image
        image_path = crop_image_paths.get(crop.lower(), None)
        if image_path:
            image = Image.open(image_path)
            st.image(image, caption=crop.capitalize(), use_column_width=True)
        else:
            st.write("Image not available for this crop.")

if __name__ == "__main__":
    try:
        main()  # Or the name of your Streamlit app's main function
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")

