import pickle
import tensorflow as tf

from flask import Flask
from tensorflow.keras.utils import img_to_array, load_img


IMAGE_SIZE = 256
MODEL_PATH = "saved_models/unet.pkl"


def load_image(img_path):
    image = load_img(img_path)
    image = img_to_array(image)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))        
    image = tf.cast(image, tf.float32)
    image = image / 255.
    return tf.expand_dims(image, axis=0)


# Load model
print("Loading model....")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

print("model loaded.")



app = Flask(__name__)


@app.post("/store/<string:image_name>")
def create_item(image_name):
    
    # Load image
    image = load_image(image_name)

    # predict result
    result = model.predict(image)

    # save predicted image
    with open(f"result_images/{image_name}", "wb") as file:
        file.write(result[0,:,:,:])
    
    return {"result_path": f"result_images/{image_name}"}