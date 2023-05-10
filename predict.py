import os
import cv2
import glob
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import img_to_array, load_img


IMAGE_SIZE = 256
IMAGE_PATH = "1.jpg"
MODEL_PATH = "saved_models/unet.pkl"


def create_figure(image, result, i):
    n_cols = 3

    # Plot Image 
    plt.subplot(1, n_cols, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    # Plot result Map
    plt.subplot(1, n_cols, 2)
    plt.imshow(result, cmap='gray')
    plt.title('Result Map')
    plt.axis('off')

    # Plot Mixed Overlap
    plt.subplot(1, n_cols, 3)
    plt.imshow(image)
    plt.imshow(result, alpha=0.5, cmap='gray')
    plt.title("Overlap")
    plt.axis('off')


    # Final Show
    figure = plt.gcf()
    figure.set_size_inches(32, 12)
    plt.savefig(f"save_video/{i}.png", bbox_inches='tight')
    # plt.show()


def load_image(img_path):
    image = load_img(img_path)
    image = img_to_array(image)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))        
    image = tf.cast(image, tf.float32)
    image = image / 255.
    return tf.expand_dims(image, axis=0)


def load_model(model_path):
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    return model


def create_video():

    for image in os.listdir("save_video/"):
        if len(image) == 5:
            os.rename(f"save_video/{image}", f"save_video/00{image}")
    
    for image in os.listdir("save_video/"):
        if len(image) == 6:
            os.rename(f"save_video/{image}", f"save_video/0{image}")
        
    
    img_array = []
    for filename in sorted(glob.glob('save_video/*.png')):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)


    out = cv2.VideoWriter('island_result.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()



if __name__ == "__main__":
#    img = load_image(IMAGE_PATH)

    model = load_model(MODEL_PATH)
    
    cap = cv2.VideoCapture("/island.mp4")

    frames = []
    _, image = cap.read()
    i = 0

    while _ :
        image = img_to_array(image)
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))        
        image = tf.cast(image, tf.float32)
        image = image / 255.
        
        frames.append(image)

        _, image = cap.read()
        i += 1

    no_frames = len(frames)
    frames = np.array(frames)
    result = model.predict(frames)


    for i in range(no_frames):
        bgr_img = cv2.cvtColor(np.array(frames[i,:,:,:]), cv2.COLOR_RGB2BGR)
        label = result[i,:,:,:]
        create_figure(bgr_img, label, i)

    
    create_video()
 