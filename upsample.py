# coding: utf-8

import tensorflow as tf
import os
import model
import numpy as np
slim = tf.contrib.slim


gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Define tf operations
# Here test tf.image.resize_images
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
image = tf.image.convert_image_dtype(img_input, dtype=tf.float32)
image.set_shape(shape=(544, 960, 3))
images = tf.expand_dims(image, 0)
resized_images = tf.image.resize_images(images, (1088, 1920), method=0)
#probabilities=model.train(images,None,shape=[544,960],numclasses=27,reuse=None)
#variables_to_restore = slim.get_variables_to_restore()
#saver = tf.train.Saver(variables_to_restore)
#saver.restore(isess, checkpoint)



import cv2
cap = cv2.VideoCapture(r'a.mp4') 


while cap.isOpened():  
    # get a frame  
    rval, frame = cap.read()
    # save a frame  
    if rval==True:
        frame= frame[:,:, (2,1,0)]
        s = isess.run(feed_dict={img_input: frame})  
    else:
        break
    # show a frame
    #mm= np.array(s[0,:,:])
    #colour_codes = np.array(label_to_colours)
    #x = colour_codes[mm.astype(int)]
    cv2.imshow("Resized",frame[:,:, (2,1,0)])
    cv2.imshow("Capture", x)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()  
cv2.destroyAllWindows()


# In[5]:




