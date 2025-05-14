import streamlit as st
import cv2 as cv
import numpy as np
import keras

# Set up page title and layout
st.set_page_config(page_title="Leaf Disease Detection", page_icon="🌿", layout="centered")

# Display the title and description in a more structured way
st.title("🌿 Leaf Disease Detection")
st.markdown("""
This application uses deep learning techniques to detect various leaf diseases based on uploaded images. 
The model is built using transfer learning, leveraging a pre-trained base model to identify diseases in different types of plants.

Please upload a leaf image of **Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry**, or **Tomato** for accurate predictions.
""")

# Load the model
model = keras.models.load_model('Training/model/Leaf Deases(96,88).h5')

# Define the disease labels
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
        st.markdown(f"### Prediction Result: 🌿 **{result}**")
        st.markdown(f"Confidence: **{confidence:.2f}%**")
        st.success("The model is confident about this prediction.")
    else:
        st.warning("The model is not confident. Please try another image for a better result.")
        st.markdown("The confidence is below 80%. We recommend you upload a clearer image.")
        
    # Optionally, you can add the option to display the processed image or debug info
    # st.image(normalized_image[0], caption="Processed Image", use_column_width=True, channels="RGB")
