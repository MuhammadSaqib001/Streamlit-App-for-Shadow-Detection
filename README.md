# Shadow Detector App

## Overview

The **Shadow Detector App** is a web application built using Streamlit that utilizes a trained UNet model to identify and analyze shadows in uploaded images. This app allows users to upload images and detect shadows using advanced AI technology with just a click of a button.

## Features

- **Image Upload**: Users can upload images in JPG, PNG, or JPEG format.
- **Shadow Detection**: The app processes the image to detect shadows and provides a binary mask representing the detected shadows.
- **Image Preview**: Users can preview the uploaded image and the detected shadows side by side.
- **Download Option**: Users can download the shadow mask as a PNG file.

## Technologies Used

- **Streamlit**: For building the web application.
- **PyTorch**: For loading and using the UNet model for shadow detection.
- **OpenCV**: For image processing tasks.
- **NumPy**: For numerical operations on images.
- **PIL**: For image saving and manipulation.
