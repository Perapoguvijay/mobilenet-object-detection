import streamlit as st 
import tensorflow as tf 
import keras 
from keras.applications import mobilenet,imagenet_utils
from keras.preprocessing import image 
import numpy as np 

#1.Load the MobileNet 
@st.cache_resource
def load_model(): 
    model=keras.applications.mobilenet.MobileNet()
    return model
mobile=load_model()

# Load ImageNet class names 
@st.cache_data
def load_classes():
    class_names_path = tf.keras.utils.get_file('ImageNetLabels.txt',
                                           'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    with open(class_names_path, 'r') as f:
        class_names = f.read().splitlines()
    return class_names
class_names=load_classes()

# 3.Streamlit App UI

st.title("MobileNet Image classifier (streamlit)")
st.write("Upload an image and let MobileNet classify it into one of the 1000 ImageNet categories.")

#upload file 
upload_file=st.file_uploader("Upload an image...",type=["jpg","jpeg","png"])

## 4. Process Upload Image 
if upload_file is not None:
    img=image.load_img(upload_file,target_size=(224,224))
    st.image(img,caption="Upload Image",use_column_width=True)

    # convert image to array
    img_array=image.img_to_array(img)
    expand_dims=np.expand_dims(img_array,axis=0)
    processing_img=keras.applications.mobilenet.preprocess_input(expand_dims)

    # 5.Make predicitons 

    predictions=mobile.predict(processing_img)
    predicted_id=np.argmax(predictions)

    st.subheader("Prediction Result:")
    st.write(f"**predicted:** {class_names[predicted_id+1]}")
    st.write(f"**Confidence:**{np.max(predictions):.2f}")

    #6.Show Top-5 Predicitons 

    st.subheader(" Top-5 Predicitons:")
    top_indices=predictions[0].argsort()[-5:][::-1]
    for i in top_indices:
        st.write(f"{class_names[i+1]}: {predictions[0][i]:.2f}")
