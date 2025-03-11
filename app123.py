import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import gdown
from tensorflow.keras.models import load_model

# Download the model if not already present
url = "https://drive.google.com/uc?id=1-_fheugEQeUInNDXTnZIZTyr1okhN1r8"
output = "soil_classifier.keras"

if not os.path.exists(output):  # Download only if not exists
    gdown.download(url, output, quiet=False)
model = load_model(output)

# Define categories (soil types)
categories = ["Alluvial soil", "Black soil", "Clay soil", "Red soil"]

# Load crop data
crop_data = pd.read_csv(r"https://raw.githubusercontent.com/FABULOUS51/FARMER-ASSISTANT/refs/heads/main/soil_to_crop.csv")

# Create a dictionary with crop names and their Wikipedia links
crop_wikipedia_links = {
"Rice":"https://en.wikipedia.org/wiki/Rice"
 "Wheat":"https://en.wikipedia.org/wiki/Wheat"
 "Sugercane":"https://en.wikipedia.org/wiki/Sugarcane"
  "Maize":"https://en.wikipedia.org/wiki/Maize"
   
    # Add more crops and their Wikipedia links as needed
}

# Function to predict soil type from an image
def predict_soil(image):
    img_size = 155  # Same as used during training
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)  # Read image from file buffer
    img = cv2.resize(img, (img_size, img_size)) / 255.0  # Resize and normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    confidence = np.max(prediction)  # Get confidence (probability) of the prediction
    soil_type = categories[np.argmax(prediction)]
    return soil_type, confidence

# Function to suggest crops based on soil type and provide Wikipedia links
def suggest_crops(soil_type):
    soil_type = soil_type.strip()  # Remove leading/trailing spaces
    crops = crop_data[crop_data['soil_type'].str.strip().str.lower() == soil_type.lower()]['crops']
    
    if crops.empty:
        return "No crops found for this soil type"
    
    # Get the crops for the soil type and split by comma
    crop_list = crops.iloc[0].split(', ') if isinstance(crops.iloc[0], str) else []
    
    # Create a list of crops with their Wikipedia links
    crop_with_links = []
    for crop in crop_list:
        # If crop is found in the dictionary, create a hyperlink
        if crop in crop_wikipedia_links:
            crop_with_links.append(f"[{crop}]({crop_wikipedia_links[crop]})")
        else:
            crop_with_links.append(crop)  # If no Wikipedia link found, just display the crop name
    
    return crop_with_links

# Streamlit Web App
st.title("🌱 FARMER BUDDY SYSTEM")

col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the proportions of the columns

with col2:
    st.image(r"LOGO.png", width=300)

st.write("**UPLOAD SOIL IMAGES FOR CROP RECOMMENDATION**.")

# Upload image
uploaded_file = st.file_uploader("Choose a soil image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Soil Image", use_container_width=True)

    # Predict soil type and confidence
    predicted_soil, confidence = predict_soil(uploaded_file)
    
    # If confidence is below a threshold, consider the image invalid
    confidence_threshold = 0.6  # You can adjust this threshold
    if confidence < confidence_threshold:
        st.error(f"❌ Invalid Input: The uploaded image does not appear to be a valid soil image.")
    else:
        st.success(f"**Predicted Soil Type:** {predicted_soil} (Confidence: {confidence*100:.2f}%)")
        
        # Suggest crops with Wikipedia links
        recommended_crops = suggest_crops(predicted_soil)
        if isinstance(recommended_crops, list):
            st.markdown("**Recommended Crops:**")
            for crop in recommended_crops:
                st.markdown(f"- {crop}")
        else:
            st.info(recommended_crops)

    # Support message
    st.image(r"farmers-7457046_1280.jpg", caption="Support Farmers for a Better Future", use_container_width=True)


