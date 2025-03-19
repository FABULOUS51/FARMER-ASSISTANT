import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import gdown
from tensorflow.keras.models import load_model
import gdown

url = "https://drive.google.com/uc?id=1-_fheugEQeUInNDXTnZIZTyr1okhN1r8"
output = "soil_classifier.keras"

if not os.path.exists(output):  # Download only if not exists
    gdown.download(url, output, quiet=False)

model = load_model(output)



# Define categories (soil types)
categories = ["Alluvial soil", "Black soil", "Clay soil","Red soil"]


# Load crop data
crop_data = pd.read_csv(r"https://raw.githubusercontent.com/FABULOUS51/FARMER-ASSISTANT/refs/heads/main/soil_to_crop.csv")

# Function to predict soil type from an image
def predict_soil(image):
    img_size = 155
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    img = cv2.resize(img, (img_size, img_size)) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    
    soil_type = categories[np.argmax(prediction)]
    return soil_type

# Function to suggest crops based on soil type
def suggest_crops(soil_type):
    soil_type = soil_type.strip()
    crops = crop_data[crop_data['soil_type'].str.strip().str.lower() == soil_type.lower()]['crops']
    
    if crops.empty:
        return ["No crops found for this soil type"]
    
    return crops.iloc[0].split(', ') if isinstance(crops.iloc[0], str) else []

# Streamlit Web App

st.title("FARMER BUDDY SYSTEM")
col1, col2, col3 = st.columns(3)
with col1:
    st.image(r"360_F_123708977_X8lHoZ3iSb6rRjsmFb2mxGNp2dngJrjh.jpg",width=400)
with col2:
    st.image(r"5e7c07a78fb76a9066bbfa410458b849.jpg",width=400)
with col3:
    st.image(r"lovepik-farmer-farming-in-wheat-field-picture_501611486.jpg",use_container_width=400)
st.write("**UPLOAD SOIL IMAGES FOR CROP RECOMMENDATION**.")
st.subheader("🖼️ Sample Soil Types")
soil_col1, soil_col2, soil_col3, soil_col4 = st.columns(4)
with soil_col1:
    st.image(r"Alluvial_3.jpg", caption="Alluvial Soil",use_container_width=200)
with soil_col2:
    st.image(r"Black_10.jpg", caption="Black Soil",use_container_width=200)
with soil_col3:
    st.image(r"Clay_5.jpg", caption="Clay Soil",use_container_width=200)
with soil_col4:
    st.image(r"Copy of image5.jpeg", caption="Red Soil",use_container_width=200)
# Upload image
uploaded_file = st.file_uploader("Choose a soil image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Soil Image", use_container_width=400)

    predicted_soil = predict_soil(uploaded_file)
    st.success(f"✅ **Predicted Soil Type:** {predicted_soil}")

    # Display recommended crops in a large column
    st.subheader("🌾 Recommended Crops:")
    recommended_crops = suggest_crops(predicted_soil)
    
    crop_col = st.columns(1)[0]  
    with crop_col:
        for crop in recommended_crops:
            st.markdown(f"<div class='crop-box'>✅ {crop}</div>", unsafe_allow_html=True)
 
    st.image(r"farmers-7457046_1280.jpg", caption="Support Farmers for a Better Future", use_container_width=True)


