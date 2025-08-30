import streamlit as st
# import tensorflow as tf
import keras 
import numpy as np 
from keras.applications import mobilenet,imagenet_utils
from keras.preprocessing import image


#Load MobileNet model 
@st.cache_resource
def load_model():
    model=mobilenet.MobileNet()
    return model  

mobile=load_model()

# Title 
st.title("🖼️ MobileNet Image Classifier")

# file uploader 
uploaded_file=st.file_uploader("Upload an Image...",type=['jpg','png','jpeg'])

if uploaded_file is not None:
    img=image.load_img(uploaded_file,target_size=(224,224))

    #Show image in streamlit 

    st.image(img,caption="Uploaded Image",use_column_width=True)

    #Convert image into array

    img_array=image.img_to_array(img)
    img_array_expanded_dims=np.expand_dims(img_array,axis=0)

    # Preprocessing image 
    preprocessing_img=keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
    
    # Predicitons 
    predictions=mobile.predict(preprocessing_img)
    results=imagenet_utils.decode_predictions(predictions,top=5)[0]

    # show top predictions 
    st.subheader("Prediction")
    for imgenet_id,label,score in results:
        st.write(f"**{label}**:{score:.2f}")
        