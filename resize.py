# coding: utf-8

import tensorflow as tf
import os
import model
import numpy as np
slim = tf.contrib.slim


gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

#config = tf.ConfigProto(device_count={'GPU': 0})

isess = tf.InteractiveSession(config=config)

# Define tf operations
# Here test tf.image.resize_images
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
image = tf.image.convert_image_dtype(img_input, dtype=tf.float32)
image.set_shape(shape=(544, 960, 3))
images = tf.expand_dims(image, 0)
resized_images = tf.image.resize_images(images, (544, 960), method=0)

import cv2
cap = cv2.VideoCapture(r'a.mp4') 


while cap.isOpened():  
    # get a frame  
    rval, frame = cap.read()
    # save a frame  
    if rval==True:
        frame= frame[:,:, (2,1,0)]
        resized_frame = isess.run(resized_images, feed_dict={img_input: frame})  
    else:
        break
    # show a frame
    cv2.imshow("Capture",frame[:,:, (2,1,0)])
    resized_frame = resized_frame[0,:,:,:]
    cv2.imshow("Resized", resized_frame[:,:, (2,1,0)])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()  
cv2.destroyAllWindows()


# In[5]:




