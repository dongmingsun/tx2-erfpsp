import os
import sys
import numpy as np
import cv2
import pyzed.camera
import pyzed.types
import pyzed.core
import pyzed.defines
import color_definitions
import model
import tensorflow as tf
slim = tf.contrib.slim


def main():
  zed_init = pyzed.camera.PyInitParameters()
  # Use VGA video mode
  zed_init.camera_resolution = pyzed.defines.PyRESOLUTION.PyRESOLUTION_VGA
  zed_init.camera_fps = 30  # Set fps at 30
  zed_camera = pyzed.camera.PyZEDCamera()
  if not zed_camera.is_opened():
    print("Opening ZED Camera...")
  zed_status = zed_camera.open(zed_init)
  if zed_status != pyzed.types.PyERROR_CODE.PySUCCESS:
    print(repr(zed_status))
    exit()

  zed_runtime = pyzed.camera.PyRuntimeParameters()
  zed_mat = pyzed.core.PyMat()
  print_camera_information(zed_camera)

  checkpoint = 'ckpt/model.ckpt-27537'
  gpu_options = tf.GPUOptions(allow_growth=True)
  config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
  isess = tf.InteractiveSession(config=config)
  img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
  image = tf.image.convert_image_dtype(img_input, dtype=tf.float32)
  image.set_shape(shape=(376, 672, 3))
  images = tf.expand_dims(image, 0)
  probabilities = model.train(
      images, None, shape=[376, 672], numclasses=27, reuse=None)
  variables_to_restore = slim.get_variables_to_restore()
  saver = tf.train.Saver(variables_to_restore)
  saver.restore(isess, checkpoint)
  predictions = tf.argmax(probabilities, -1)
  color_mat = tf.constant(label_to_colours, dtype=tf.float32)
  onehot_output = tf.one_hot(predictions, depth=27)
  onehot_output = tf.reshape(onehot_output, (-1, 27))
  pred = tf.matmul(onehot_output, color_mat)
  pred = tf.reshape(pred, (1, 376, 672, 3))


  while True:
    # Retrieve a frame
    zed_err = zed_camera.grab(zed_runtime)
    if zed_err == pyzed.types.PyERROR_CODE.PySUCCESS:
      zed_camera.retrieve_image(zed_mat, pyzed.defines.PyVIEW.PyVIEW_LEFT)
      rval = True
      frame = zed_mat.get_data()
      frame = frame[:, :, (2, 1, 0)]
      s = isess.run(pred, feed_dict={img_input: frame})
      key = cv2.waitKey(5)
    else:
      rval = False
      print('zed err retriving an image...')
      break
      
    # Show a frame
    cv2.imshow("ZED-M", frame[:, :, (2, 1, 0)])
    cv2.imshow("Prediction", s[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  cv2.destroyAllWindows()


def print_camera_information(cam):
  print("Resolution: {0}, {1}.".format(
      round(cam.get_resolution().width, 2), cam.get_resolution().height))
  print("Camera FPS: {0}.".format(cam.get_camera_fps()))
  print("Firmware: {0}.".format(cam.get_camera_information().firmware_version))
  print("Serial number: {0}.\n".format(
      cam.get_camera_information().serial_number))


if __name__ == "__main__":
  main()
