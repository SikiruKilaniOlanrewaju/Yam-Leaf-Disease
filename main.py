import streamlit as st
import cv2 as cv
import numpy as np
import keras

# Page config
st.set_page_config(page_title="Yam Leaf Disease Detection -- Ewe isu yin ni arun Anthracnose", page_icon="🌿", layout="centered")

# Custom CSS
st.markdown("""
    <style>
        .css-1d391kg {
            background-color: #f4f9f4;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton > button {
            background-color: red;
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
        footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: black;
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.1);
        }
        .about-developer {
            font-size: 16px;
            font-weight: bold;
            color: #2e8b57;
            text-align: center;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Language options
lang = st.sidebar.selectbox("🌍 Select Language / Yan Ede:", ["English", "Yoruba"])

# Translation dictionary
def t(en_text):
    translations = {
        "Welcome to the Yam Leaf Disease Detection application!": "Kaabo si Eto Idanimọ Arun Ewe Isu!",
        "This app uses **deep learning** techniques to detect various diseases in plants by analyzing uploaded leaf images.": "App yii nlo **ẹ̀kọ́ jinlẹ̀** lati mọ awọn arun orisirisi ninu awọn ohun ọgbin nipa ayewo aworan ewe ti a gbe soke.",
        "It leverages **transfer learning** and a pre-trained model to give accurate results.": "O lo **ẹ̀kọ́ gbigbe** ati awoṣe ti a ti kọ́ tẹlẹ lati fun ni abajade deede.",
        "The app supports detection of Yam Anthracnose disease and others.": "App naa n ṣe idanimọ Arun Anthracnose ati awọn miiran lori ewe isu.",
        "How to Use:": "Bá a ṣe n Lo:",
        "Upload a leaf image of the plant you want to test.": "Gbe aworan ewe ti o fẹ ṣe ayewo si oke.",
        "The app will predict whether the leaf is affected by disease or is healthy.": "App naa yoo sọ boya ewe naa ni arun tabi o ni ilera.",
        "You will see the confidence level of the prediction.": "Ipele igbẹkẹle esi naa yoo han.",
        "Proceed to Test a Leaf": "Tẹsiwaju lati Ṣayẹwo Ewe",
        "Go Back to Welcome Screen": "Pada si Iboju Ifihan",
        "Yam Leaf Disease Detection": "Idanimọ Arun Ewe Isu",
        "Please upload a leaf image of any Yam": "Jowo gbe aworan ewe isu kan sii",
        "Upload a Leaf Image": "Gbe Aworan Ewe sii",
        "Analyzing the image...": "N ṣe itupalẹ aworan naa...",
        "Prediction Result": "Abajade Asọtẹlẹ",
        "Disease:": "Arun:",
        "Confidence Level:": "Ipele Igbekele:",
        "The model is confident about this prediction.": "Awoṣe naa ni igbẹkẹle lori asọtẹlẹ yii.",
        "Low Confidence": "Igbekele Kekere",
        "The model's confidence is below 80%. Please upload a clearer image or try again.": "Igbekele awoṣe naa kere ju 80%. Jọwọ gbe aworan ti o ye diẹ sii tabi gbiyanju lẹẹkansi.",
        "Confidence:": "Igbekele:",
        "Upload Another Image": "Gbe Aworan Miran sii",
        "Developed by Mubaraq Salaudeen": "A ṣe apẹrẹ nipasẹ Mubaraq Salaudeen",
        "Project Student - OYSCATECH": "Akẹkọ iṣẹ akanṣe - OYSCATECH",
        "Contact: .......@gmail.com": "Pe: .......@gmail.com",
    }
    return translations.get(en_text, en_text) if lang == "Yoruba" else en_text

# Welcome Screen
def show_welcome_screen():
    st.markdown("<h1 class='header'>🌿 YAM LEAF DISEASE DETECTION SYSTEM</h1>", unsafe_allow_html=True)
    st.markdown(f"**{t('Welcome to the Yam Leaf Disease Detection application!')}**\n\n" +
                f"{t('This app uses **deep learning** techniques to detect various diseases in plants by analyzing uploaded leaf images.')}\n\n" +
                f"{t('It leverages **transfer learning** and a pre-trained model to give accurate results.')}\n\n" +
                f"{t('The app supports detection of Yam Anthracnose disease and others.')}")
    
    st.markdown(f"#### {t('How to Use:')}")
    st.markdown(f"1. {t('Upload a leaf image of the plant you want to test.')}")
    st.markdown(f"2. {t('The app will predict whether the leaf is affected by disease or is healthy.')}")
    st.markdown(f"3. {t('You will see the confidence level of the prediction.')}")
    
    if st.button(t("Proceed to Test a Leaf")):
        st.session_state.proceeded_to_test = True
        st.experimental_rerun()

    st.markdown(f"""
    <div class="about-developer">
        <p>{t('Developed by Mubaraq Salaudeen')}</p>
        <p>{t('Project Student - OYSCATECH')}</p>
        <p>{t('Contact: .......@gmail.com')}</p>
    </div>
    """, unsafe_allow_html=True)

# Detection Page
def show_leaf_disease_detection():
    st.title(f"🌿 {t('Yam Leaf Disease Detection')}")
    st.markdown(f"""
    {t("This application uses **deep learning** to detect various leaf diseases.")}  
    {t("The model is built using **transfer learning**, leveraging a pre-trained base model to identify diseases in different types of plants.")}

    **{t('Please upload a leaf image of any Yam')}**
    """)

    model = keras.models.load_model('Training/model/Leaf Deases(96,88).h5')

    label_name = ['cab', 'Black rot', 'rust', 'healthy', 'mildew',
                  'healthy', 'leaf spot Gray leaf spot', 'Common rust', 'Leaf Blight', 'healthy', 
                  'Black rot', 'Grape Esca', 'Grape Leaf blight', 'healthy', 'Peach Bacterial spot', 'healthy', 
                  'Bacterial spot', 'healthy', 'Early blight', 'Late blight', 'healthy', 
                  'Anthracnose Leaf scorch', 'healthy', 'Bacterial spot', 'Early blight', 'Late blight', 
                  'Leaf Mold', 'Septoria leaf spot', 'Spider mites', 'Spot', 
                  'Yellow Leaf Curl Virus', 'virus', 'healthy']

    uploaded_file = st.file_uploader(t("Upload a Leaf Image"), type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
        normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)), axis=0)

        with st.spinner(t("Analyzing the image...")):
            predictions = model.predict(normalized_image)
            st.image(image_bytes, caption=t("Uploaded Leaf Image"), use_column_width=True, channels="RGB")

            predicted_class = np.argmax(predictions)
            confidence = predictions[0][predicted_class] * 100
            result = label_name[predicted_class]

            if confidence >= 80:
                st.markdown(f"### 🌿 **{t('Prediction Result')}**")
                st.markdown(f"**{t('Disease:')} {result}**")
                st.markdown(f"**{t('Confidence Level:')} {confidence:.2f}%**")
                st.success(t("The model is confident about this prediction."))
            else:
                st.warning(f"⚠️ **{t('Low Confidence')}**")
                st.markdown(t("The model's confidence is below 80%. Please upload a clearer image or try again."))
                st.markdown(f"**{t('Confidence:')} {confidence:.2f}%**")

        if st.button(t("Upload Another Image")):
            st.experimental_rerun()

    if st.button(t("Go Back to Welcome Screen")):
        del st.session_state.proceeded_to_test
        st.experimental_rerun()

# App Routing
if 'proceeded_to_test' not in st.session_state:
    show_welcome_screen()
else:
    show_leaf_disease_detection()
