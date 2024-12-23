import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2 

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model(r"C:\Users\nishu\OneDrive\Desktop\plant dataset\trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Plant Identifier"])

#homepage 
if(app_mode=="Home"):
    st.header("**MEDICINAL PLANT IDENTIFIER**")
    img = Image.open("home_pg.jpeg")
    st.image(img,width=500,channels='RGB')
    st.markdown("""
    Welcome to Medicinal Plant Identifier System! 🌿🔍
    
    Our mission is to help in identifying medicinal plant efficiently. Just Upload an image of a plant, and our system will identify what kind of medicinal plant it is.!

    ### How It Works
    1. **Upload Image:** Go to the **Plant Identifier** page and upload an image of a plant you want to identify.
    2. **Analysis:** Our system will process the image using algorithms to identify the possible plant.
    3. **Results:** View the results and recommendations of the plant.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes machine learning techniques for accurate Identification.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Plant Identifier** page in the sidebar to upload an image and know different kinds of medicinal plants.!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                ### Our Team 
                "Hello and welcome to Plant Identifier! 
                We are students from the AI-ML department who have create an application that can recognize medicinal plants from images.
                Meet our awesome team: """)

    img1 = Image.open("bhav.jpeg")
    st.image(img1,width=180,caption = 'kasala Bhavana', channels='RGB')
    img2 = Image.open("lak.jpeg")
    st.image(img2,width=250,caption = 'Lakshya U Reddy', channels='RGB')
    img3 = Image.open("nish.jpeg")
    st.image(img3,width=220,caption = 'Nishat N Shahu', channels='RGB')
    img4 = Image.open("tan.jpeg")
    st.image(img4,width=310,caption = 'Tanya Gopal', channels='RGB')
    st.markdown("""We hope you enjoy using our application and learning about different plants. Thank you for checking out Plant Identifier!""")
    st.markdown("""         
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found from ncbi library.
                This dataset consists of about 6K plus rgb images of medicinal leaves which is categorized into 80 different classes.The total dataset is divided into training and validation set preserving the directory structure.
                A new directory containing 2K plus test images is created later for prediction purpose.
                #### Content
                1. train (6899 images)
                2. test (2068 images)
                3. validation (6899 images)
                """)

#Prediction Page
elif(app_mode=="Plant Identifier"):
    st.header("Identifier")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name =  ['Aloevera',
 'Amla',
 'Amruthaballi',
 'Arali',
 'Astma_weed',
 'Badipala',
 'Balloon_Vine',
 'Bamboo',
 'Beans',
 'Betel',
 'Bhrami',
 'Bringaraja',
 'Caricature',
 'Castor',
 'Catharanthus',
 'Chakte',
 'Chilly',
 'Citron lime (herelikai)',
 'Coffee',
 'Common rue(naagdalli)',
 'Coriender',
 'Curry',
 'Doddpathre',
 'Drumstick',
 'Ekka',
 'Eucalyptus',
 'Ganigale',
 'Ganike',
 'Gasagase',
 'Ginger',
 'Globe Amarnath',
 'Guava',
 'Henna',
 'Hibiscus',
 'Honge',
 'Insulin',
 'Jackfruit',
 'Jasmine',
 'Kambajala',
 'Kasambruga',
 'Kohlrabi',
 'Lantana',
 'Lemon',
 'Lemongrass',
 'Malabar_Nut',
 'Malabar_Spinach',
 'Mango',
 'Marigold',
 'Mint',
 'Neem',
 'Nelavembu',
 'Nerale',
 'Nooni',
 'Onion',
 'Padri',
 'Palak(Spinach)',
 'Papaya',
 'Parijatha',
 'Pea',
 'Pepper',
 'Pomoegranate',
 'Pumpkin',
 'Raddish',
 'Rose',
 'Sampige',
 'Sapota',
 'Seethaashoka',
 'Seethapala',
 'Spinach1',
 'Tamarind',
 'Taro',
 'Tecoma',
 'Thumbe',
 'Tomato',
 'Tulsi',
 'Turmeric',
 'ashoka',
 'camphor',
 'kamakasturi',
 'kepala']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))


