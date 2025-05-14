import streamlit as st
import cv2 as cv
import numpy as np
import keras

# Set up page title and layout with enhanced UI
st.set_page_config(page_title="Leaf Disease Detection", page_icon="🌿", layout="centered")

# Add custom CSS to enhance design (e.g., font, background colors)
st.markdown("""
    <style>
        .css-1d391kg {
            background-color: #f0f8ff;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton > button {
            background-color: #008000;
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
    </style>
""", unsafe_allow_html=True)

# Display the title and description in a modern style
st.markdown('<div class="header">🌿 Leaf Disease Detection</div>', unsafe_allow_html=True)

st.markdown("""
This application uses **deep learning** to detect various leaf diseases. 
Upload a leaf image of **Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry**, or **Tomato** to get predictions.
""")

# Load the pre-trained model
model = keras.models.load_model('Training/model/Leaf Deases(96,88).h5')

# Disease labels
label_name = ['Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Cherry Powdery mildew',
              'Cherry healthy', 'Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust', 'Corn Northern Leaf Blight', 'Corn healthy', 
              'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy', 'Peach Bacterial spot', 'Peach healthy', 
              'Pepper bell Bacterial spot', 'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 
              'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 
              'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites', 'Tomato Target Spot', 
              'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']

# Sidebar for navigation and instructions
st.sidebar.title("Instructions")
st.sidebar.write("""
1. **Upload** a leaf image (JPG, PNG, or JPEG).
2. **Wait for prediction**: The model will analyze the image.
3. **Get results**: See the disease prediction with confidence levels.
""")

# Image upload functionality
uploaded_file = st.file_uploader("Upload Leaf Image (Max size 200MB)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Displaying a progress bar while processing
    with st.spinner("Analyzing the image..."):
        image_bytes = uploaded_file.read()
        img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
        # Image preprocessing
        normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)), axis=0)

        # Predict disease
        predictions = model.predict(normalized_image)

        # Show the uploaded image
        st.image(image_bytes, caption="Uploaded Leaf Image", use_column_width=True, channels="RGB", clamped=True)

        # Process prediction
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class] * 100
        result = label_name[predicted_class]

        # Show prediction results with dynamic messages
        if confidence >= 80:
            st.success(f"🌿 Prediction: **{result}**")
            st.write(f"Confidence Level: **{confidence:.2f}%**")
            st.balloons()  # Celebrate if prediction is successful
        else:
            st.warning("⚠️ Confidence is low! Please upload a clearer image or try again.")
            st.markdown(f"Confidence Level: **{confidence:.2f}%**", unsafe_allow_html=True)

        # Adding a button for reset or upload new image
        if st.button('Upload Another Image'):
            st.experimental_rerun()
