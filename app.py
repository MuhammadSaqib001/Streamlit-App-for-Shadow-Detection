import streamlit as st
from PIL import Image
import numpy as np
import torch
import os
import sys
import cv2
import io
from base64 import b64encode  # Import b64encode function from base64 module


# Ensure the scripts directory is in the import path
scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

# Import the UNet model from the scripts directory
from unet_model import UNet

# Load the trained model
model = UNet()
model.load_state_dict(torch.load(os.path.join('models', 'unet_model.pth'), map_location=torch.device('cpu')))
model.eval()

# Function to preprocess image
def preprocess_image(image):
    image = cv2.resize(image, (256, 256))
    # Normalize
    image = image.astype(np.float32) / 255.0
    # Transpose the image dimensions to match the input requirements of the model (C, H, W)
    image = image.transpose((2, 0, 1))
    # Convert to torch tensor and add a batch dimension
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    return image

# Function to postprocess the prediction
def postprocess_image(prediction):
    # Convert model prediction to a displayable format
    prediction = prediction.squeeze().detach().numpy()
    # Apply a threshold to create a binary mask
    prediction = (prediction > 0.5).astype(np.uint8) * 255
    return prediction

# Function to predict shadow
def predict_shadow(image):
    processed_image = preprocess_image(image)
    with torch.no_grad():
        prediction = model(processed_image)
    shadow_mask = postprocess_image(prediction)
    return shadow_mask


# Function to create a download link
def get_binary_file_downloader_html(link_text, file, file_name):
    # Convert numpy array to PIL image
    pil_img = Image.fromarray(file.astype('uint8'))
    file_name = f"shadows_{file_name}"
    # Get original dimensions
    with io.BytesIO() as buffer:
        pil_img.save(buffer, format='PNG')
        buffer.seek(0)
        bin_str = buffer.getvalue()

    href = f'data:application/octet-stream;base64,{b64encode(bin_str).decode()}'
    return f'<div style="text-align: right;"><a href="{href}" download="{file_name}" style="font-weight: bold; color:white;">{link_text}</a></div>'


st.set_page_config(page_title='Shadow Detector', page_icon='🔍', layout="wide")
st.title("Shadow Detector App")

# Initialize session state for button_pressed
if 'button_pressed' not in st.session_state:
    st.session_state.button_pressed = False

uploaded_file = st.file_uploader(":orange-background[Instantly identify and analyze shadows in your photos with advanced AI technology with click of a button]", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # Decode the byte stream into an image using OpenCV
    image = cv2.imdecode(file_bytes, 1)
    
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the uploaded image
    with st.expander('Preview the images uploaded'):
        st.image(image_rgb, caption='Uploaded Image', use_column_width=True)

    _, button_display = st.columns([5, 1])
    with button_display:
        if st.button('Detect Shadows', use_container_width=True):
            st.session_state.button_pressed = True

    if st.session_state.button_pressed:
        with st.spinner('Processing the image ...'):
            shadow_mask = predict_shadow(image)

            new_size = (256, 256)
            image_rgb = cv2.resize(image_rgb, new_size)
            gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
            shadowed_image_array = (gray_image * (1 - shadow_mask)).astype(np.uint8)

            stitched_image = np.hstack((gray_image, shadow_mask, shadowed_image_array))

        with st.expander('Preview the shadows ...'):
            st.image(stitched_image, caption='Shadow Detected', use_column_width=True)
            if shadow_mask is not None:
                st.markdown(get_binary_file_downloader_html('Download Shadow Mask', shadow_mask, file_name), unsafe_allow_html=True)
        st.session_state.button_pressed = False
