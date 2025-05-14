import streamlit as st
import cv2 as cv
import numpy as np
import keras

# Set up page title and layout
st.set_page_config(page_title="Yam Leaf Disease Detection", page_icon="🌿", layout="centered")

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
    </style>
""", unsafe_allow_html=True)

# Display the welcome screen
def show_welcome_screen():
    st.markdown("<h1 class='header'>🌿 YAM LEAF DISEASE DETECTION SYSTEM</h1>", unsafe_allow_html=True)
    st.markdown("""
    **Welcome to the Yam Leaf Disease Detection application!**

    This app uses **deep learning** techniques to detect various diseases in plants by analyzing uploaded leaf images. 
    It leverages **transfer learning** and a pre-trained model to give accurate results. 
    The app supports detection of Yam Anthracnose disease and others.

    #### How to Use:
    1. Upload a leaf image of the plant you want to test.
    2. The app will predict whether the leaf is affected by disease or is healthy.
    3. You will see the confidence level of the prediction.

    Press the button below to begin testing your leaf images.
    """)
    
    # Button to proceed to the main detection functionality
    if st.button("Proceed to Test a Leaf"):
        # Set a session state variable to track progress
        st.session_state.proceeded_to_test = True
        st.experimental_rerun()

# Main leaf disease detection page
def show_leaf_disease_detection():
    st.title("🌿 Yam Leaf Disease Detection")
    st.markdown("""
    This application uses **deep learning** to detect various leaf diseases. 
    The model is built using **transfer learning**, leveraging a pre-trained base model to identify diseases in different types of plants.

    **Please upload a leaf image of any Yam** for accurate predictions.
    """)

    # Load the model
    model = keras.models.load_model('Training/model/Leaf Deases(96,88).h5')

    # Define the disease labels
    label_name = ['cab', 'Black rot', 'rust', 'healthy', 'mildew',
                  'healthy', 'leaf spot Gray leaf spot', 'Common rust', 'Leaf Blight', 'healthy', 
                  'Black rot', 'Grape Esca', 'Grape Leaf blight', 'healthy', 'Peach Bacterial spot', 'healthy', 
                  'Bacterial spot', 'healthy', 'Early blight', 'Late blight', 'healthy', 
                  'Anthracnose Leaf scorch', 'healthy', 'Bacterial spot', 'Early blight', 'Late blight', 
                  'Leaf Mold', 'Septoria leaf spot', 'Spider mites', 'Spot', 
                  'Yellow Leaf Curl Virus', 'virus', 'healthy']

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
            result = label_name[predicted_class]

            # Provide a professional and interactive result display
            if confidence >= 80:
                st.markdown(f"### 🌿 **Prediction Result**")
                st.markdown(f"**Disease:** {result}")
                st.markdown(f"**Confidence Level:** {confidence:.2f}%")
                st.success("The model is confident about this prediction.")
            else:
                st.warning("⚠️ **Low Confidence**")
                st.markdown("The model's confidence is below 80%. Please upload a clearer image or try again.")
                st.markdown(f"**Confidence:** {confidence:.2f}%")
            
        # Optionally, add a button to upload another image
        if st.button('Upload Another Image'):
            st.experimental_rerun()

    # Add a button to go back to the welcome screen
    if st.button("Go Back to Welcome Screen"):
        # Reset the session state to go back to the welcome page
        del st.session_state.proceeded_to_test
        st.experimental_rerun()

# Check if the user has proceeded to the testing page
if 'proceeded_to_test' not in st.session_state:
    # If the user has not clicked the proceed button, show the welcome screen
    show_welcome_screen()
else:
    # If the user has clicked proceed, show the leaf disease detection page
    show_leaf_disease_detection()
