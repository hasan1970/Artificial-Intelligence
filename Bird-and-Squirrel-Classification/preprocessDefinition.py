import tensorflow as tf
from tensorflow import keras
import numpy as np

def myPreprocess(image, label):
    #Resize
    resized_image = tf.image.resize_with_pad(image, 299, 299)
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label
 