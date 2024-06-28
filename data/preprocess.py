import os
from PIL import Image

def resize_images(source_dir, target_dir, size=(256, 256)):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for filename in os.listdir(source_dir):
        img = Image.open(os.path.join(source_dir, filename))
        img = img.resize(size)
        img.save(os.path.join(target_dir, filename))

# Directories
train_image_dir = 'C:/Users/aimyn/test_model2/data/SBU/SBU-Train/ShadowImages'
train_mask_dir = 'C:/Users/aimyn/test_model2/data/SBU/SBU-Train/ShadowMasks'
val_image_dir = 'C:/Users/aimyn/test_model2/data/SBU/SBU-Val/ShadowImages'
val_mask_dir = 'C:/Users/aimyn/test_model2/data/SBU/SBU-Val/ShadowMasks'
test_image_dir = 'C:/Users/aimyn/test_model2/data/SBU/SBU-Test/ShadowImages'
test_mask_dir = 'C:/Users/aimyn/test_model2/data/SBU/SBU-Test/ShadowMasks'

# Resize images and masks
resize_images(train_image_dir, train_image_dir + '_resized')
resize_images(train_mask_dir, train_mask_dir + '_resized')
resize_images(val_image_dir, val_image_dir + '_resized')
resize_images(val_mask_dir, val_mask_dir + '_resized')
resize_images(test_image_dir, test_image_dir + '_resized')
resize_images(test_mask_dir, test_mask_dir + '_resized')
