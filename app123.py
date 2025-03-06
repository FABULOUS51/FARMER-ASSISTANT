# Define categories (soil types)
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os

# Load trained model
model = load_model(r"C:\Users\edwin\Desktop\alfred\Dataset\soil_classifier.keras")

# Define categories (soil types)
categories = ["Alluvial soil", "Black soil", "Clay soil","Red soil"]


# Load crop data
crop_data = pd.read_csv(r"C:\Users\edwin\Desktop\alfred\soil_to_crop.csv")

# Function to predict soil type from an image
def predict_soil(image):
    img_size = 155  # Same as used during training
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)  # Read image from file buffer
    img = cv2.resize(img, (img_size, img_size)) / 255.0  # Resize and normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    soil_type = categories[np.argmax(prediction)]
    return soil_type


def suggest_crops(soil_type):
    soil_type = soil_type.strip()  # Remove leading/trailing spaces
    print(f"🔍 Debug: Predicted Soil Type -> '{soil_type}'")  # Debugging output
    print(f"🔍 Debug: Available Soil Types in CSV -> {crop_data['soil_type'].unique()}")  # Print unique values from CSV

    # Ensure soil type comparison is consistent
    crops = crop_data[crop_data['soil_type'].str.strip().str.lower() == soil_type.lower()]['crops']
    
    if crops.empty:
        print("⚠️ No match found in CSV!")
        return "No crops found for this soil type"

    return crops.iloc[0].split(', ') if isinstance(crops.iloc[0], str) else []





# Streamlit Web App

st.title("🌱 FARMER BUDDY SYSTEM")
col1, col2, col3 = st.columns(3)
with col1:
    st.image(r"C:\Users\edwin\Desktop\alfred\farmers\5e7c07a78fb76a9066bbfa410458b849.jpg",use_container_width=True)
with col2:
    st.image(r"C:\Users\edwin\Desktop\alfred\farmers\360_F_123708977_X8lHoZ3iSb6rRjsmFb2mxGNp2dngJrjh.jpg",use_container_width=True)
with col3:
    st.image(r"C:\Users\edwin\Desktop\alfred\farmers\lovepik-farmer-farming-in-wheat-field-picture_501611486.jpg",use_container_width=True)
st.write("**UPLOAD SOIL IMAGES FOR CROP RECOMMENDATION**.")

# Upload image
uploaded_file = st.file_uploader("Choose a soil image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
   st.image(uploaded_file, caption="Uploaded Soil Image", use_container_width=True)

    # Predict soil type
   predicted_soil = predict_soil(uploaded_file)
   st.success(f"**Predicted Soil Type:** {predicted_soil}")

    # Suggest crops
   recommended_crops = suggest_crops(predicted_soil)
   st.info(f"**Recommended Crops:** {', '.join(recommended_crops)}")
   st.image(r"C:\Users\edwin\Desktop\alfred\farmers\farmers-7457046_1280.jpg", caption="Support Farmers for a Better Future", use_container_width=True)
