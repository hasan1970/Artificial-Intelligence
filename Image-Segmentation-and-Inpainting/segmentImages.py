import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
from PIL import Image
import os
from scipy.signal import convolve
import matplotlib.pyplot as plt
from subSelectImages import subSelectImages

def segmentImages(image_paths):
    model = hub.load('https://tfhub.dev/google/HRNet/coco-hrnetv2-w48/1')

    threshold = 0.45
    bird_class = 17
    for fil in image_paths:
        inIm = tf.io.decode_jpeg(tf.io.read_file(fil), channels = 3)
        resp = model.predict( [tf.cast(inIm, tf.float32)/255.])
        maskIm = np.zeros_like(inIm)
        mask = tf.squeeze(resp)[:,:,bird_class] > threshold
        maskIm[mask] = inIm[mask]
    
        #Try 0.25, thats also great
        pixToEnlarge = int(0.15 * np.sqrt(float(len(np.where(mask)[0]))))
    
        if pixToEnlarge < 10:
            continue
    
        enlargenedMask = np.abs(convolve(mask.numpy(), np.ones((pixToEnlarge,pixToEnlarge)), mode='same')) > 1e-8
    
        Image.fromarray(mask.numpy()).save(fil.rpartition('.')[0]+'-mask.png')
        Image.fromarray(enlargenedMask).save(fil.rpartition('.')[0]+'-larger-mask.png')


def main():
    # Path to the directory containing the images
    image_dir = 'images'
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    print(image_paths)

    selected_images = subSelectImages(image_paths)

    segmentImages(selected_images)



