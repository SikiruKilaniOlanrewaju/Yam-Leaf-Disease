import streamlit as st
import cv2 as cv
import numpy as np
import keras

# Set up page title and layout
st.set_page_config(page_title="Advanced Leaf Disease Detection", page_icon="🌿", layout="wide")

# Add custom CSS for styling
st.markdown("""
    <style>
        .css-1d391kg {
            background-color: #f4f9f4;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 12px 30px;
            border-radius: 8px;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
        }
        .stImage {
            border: 3px solid #4CAF50;
            border-radius: 8px;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.2);
        }
        .header {
            font-size: 36px;
            color: #2e8b57;
            text-align: center;
            font-weight: bold;
        }
        .result-text {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }
        .warning-text {
            font-size: 18px;
            color: #ff9800;
        }
        .prediction-text {
            font-size: 18px;
            color: #2e8b57;
        }
        .sidebar .sidebar-content {
            font-size: 18px;
            color: #333333;
        }
    </style>
""", unsafe_allow_html=True)

# Header section
st.markdown("<h1 class='header'>🌿 Advanced Leaf Disease Detection</h1>", unsafe_allow_html=True)

# Sidebar with brief introduction to the app
st.sidebar.header("About the App")
st.sidebar.markdown("""
    This application uses **deep learning** techniques to detect various leaf diseases. 
    It utilizes **transfer learning**, leveraging a pre-trained base model to identify diseases in different plant species, including:
    
    - Apple
    - Cherry
    - Corn
    - Grape
    - Peach
    - Pepper
    - Potato
    - Strawberry
    - Tomato

    **How it works:**
    - Upload a leaf image.
    - The model predicts whether the leaf is **healthy** or **affected** by a disease.
    - The app will display the confidence level of the prediction.
""")

# Main content
st.markdown("""
This tool helps farmers, researchers, and plant enthusiasts to identify and diagnose diseases affecting different plant species based on their leaf images. 
Upload a leaf image, and get quick results with a high confidence level!
""")

# Load the model
model = keras.models.load_model('Training/model/Leaf Deases(96,88).h5')

# Define the disease labels (for model reference)
label_name = ['Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Cherry Powdery mildew',
              'Cherry healthy', 'Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust', 'Corn Northern Leaf Blight', 'Corn healthy', 
              'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy', 'Peach Bacterial spot', 'Peach healthy', 
              'Pepper bell Bacterial spot', 'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 
              'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 
              'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites', 'Tomato Target Spot', 
              'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']

# Image upload interface
uploaded_file = st.file_uploader("Upload a Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Process the uploaded image
    image_bytes = uploaded_file.read()
    img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
    normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)), axis=0)

    # Show a progress spinner while predicting
    with st.spinner("Analyzing the image..."):
        # Predict disease
        predictions = model.predict(normalized_image)
        
        # Display the uploaded image
        st.image(image_bytes, caption="Uploaded Leaf Image", use_column_width=True, channels="RGB")

        # Process prediction result
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class] * 100
        
        # Determine health status based on prediction (healthy vs affected)
        if 'healthy' in label_name[predicted_class]:
            health_status = 'Healthy'
        else:
            health_status = 'Affected'

        # Provide a professional and interactive result display
        if confidence >= 80:
            st.markdown(f"### 🌿 **Prediction Result**")
            st.markdown(f"**Health Status:** {health_status}")
            st.markdown(f"**Confidence Level:** {confidence:.2f}%")
            st.success("The model is confident about this prediction.")
        else:
            st.warning("⚠️ **Low Confidence**")
            st.markdown("The model's confidence is below 80%. Please upload a clearer image or try again.")
            st.markdown(f"**Confidence:** {confidence:.2f}%")
        
    # Optionally, add a button to upload another image
    if st.button('Upload Another Image'):
        st.experimental_rerun()
