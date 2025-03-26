import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import pickle
from datetime import datetime,timedelta
from tensorflow.keras.models import load_model
import os
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'


# Load models and scalers
# @st.cache(allow_output_mutation=True)
def load_models():
    with open('C:/Users/Priyanshu/Desktop/Project1/model/random_forest_crop_recommendation_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    
    with open('C:/Users/Priyanshu/Desktop/Project1/model\weather_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    lstm_model = load_model('C:/Users/Priyanshu/Desktop/Project1/model/crop.h5')
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
    temperature, humidity,rainfall = predict_future(date, scaler, lstm_model, original_data, time_step)
    input_features = np.array([[n, p, k, temperature, humidity, ph]])
    predicted_crop = rf_model.predict(input_features)[0]

    return predicted_crop

crop_image_paths = {
    'apple': 'C:/Users/Priyanshu/Desktop/Project1/crop img/apple.jpg',
    'banana': 'C:/Users/Priyanshu/Desktop/Project1/crop img/banana.jpg',
    'blackgram': 'C:/Users/Priyanshu/Desktop/Project1/crop img/blackgram.jpg',
    'chickpea': 'C:/Users/Priyanshu/Desktop/Project1/crop img/chickpea.jpg',
    'coconut': 'C:/Users/Priyanshu/Desktop/Project1/crop img/coconut.jpg',
    'coffee': 'C:/Users/Priyanshu/Desktop/Project1/crop img/coffee.jpg',
    'cotton': 'C:/Users/Priyanshu/Desktop/Project1/crop img/cotton.jpg',
    'grapes': 'C:/Users/Priyanshu/Desktop/Project1/crop img/grapes.jpg',
    'jute': 'C:/Users/Priyanshu/Desktop/Project1/crop img/jute.jpg',
    'kidneybeans': 'C:/Users/Priyanshu/Desktop/Project1/crop img/kidneybean.jpeg',
    'mango': 'C:/Users/Priyanshu/Desktop/Project1/crop img/mango.jpg',
    'lentil': 'C:/Users/Priyanshu/Desktop/Project1/crop img/lentil.jpeg',
    'maize': 'C:/Users/Priyanshu/Desktop/Project1/crop img/maize.jpeg',
    'mothbeans': 'C:/Users/Priyanshu/Desktop/Project1/crop img/mothbeans.jpg',
    'mungbean': 'C:/Users/Priyanshu/Desktop/Project1/crop img/mungbean.jpg',
    'muskmelon': 'C:/Users/Priyanshu/Desktop/Project1/crop img/muskmelon.jpg',
    'orange': 'C:/Users/Priyanshu/Desktop/Project1/crop img/orange.jpg',
    'papaya': 'C:/Users/Priyanshu/Desktop/Project1/crop img/papaya.jpeg',
    'pigeonpeas': 'C:/Users/Priyanshu/Desktop/Project1/crop img/pigeonbeans.jpg',
    'rice': 'C:/Users/Priyanshu/Desktop/Project1/crop img/rice.jpg',
    'watermelon': 'C:/Users/Priyanshu/Desktop/Project1/crop img/watermelon.jpg'
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
    data = pd.read_csv('C:/Users/Priyanshu/Desktop/Project1/dataset/Aligarh.csv', parse_dates=['date'])
    
    # Predict weather conditions
    try:
        temperature, humidity, rainfall = predict_future(date, scaler, lstm_model, data)
        st.write(f"Predicted Weather on {date}:")
        st.write(f"Temperature: {temperature:.2f}°C")
        st.write(f"Humidity: {humidity:.2f}%")
        if(rainfall<0):
            st.write(f"Rainfall: {00:.2f} mm")
        else:
            st.write(f"Rainfall: {rainfall:.2f} mm")
    except ValueError as e:
        st.error(str(e))
        return

    # Recommend crop
    predicted_crop = recommend_crop(n, p, k, ph, date, scaler, lstm_model, data, rf_model)
    
    st.write(f"Recommended Crop for the given conditions: {predicted_crop.capitalize()}")
    
    # Display crop image
    image_path = crop_image_paths.get(predicted_crop.lower(), None)
    if image_path:
        image = Image.open(image_path)
        st.image(image, caption=predicted_crop.capitalize(), use_column_width=True)
    else:
        st.write("Image not available for this crop.")

if __name__ == '__main__':
    main()
