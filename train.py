import argparse
import tensorflow as tf

from glob import glob
from utils import ShowProgress, save_model
from models import Unet, DataGenerator


MODELS = {
    "unet": Unet()
}


if __name__ == "__main__":

    # Parsing arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data_path", type=str)

    args = parser.parse_args()


    # Data paths
    images = sorted(glob("/content/Water Bodies Dataset/Images/*.jpg"))
    masks = sorted(glob("/content/Water Bodies Dataset/Masks/*.jpg"))

    data = []
    for image, mask in zip(images, masks):
        data.append((image, mask))

    # Create Data Generator
    generator = DataGenerator(data)

    # Create model
    model = MODELS[args.model]

    # Compile the model
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
    )

    # Train the model
    history = model.fit(
        generator,
        epochs=args.epochs,
        callbacks=[ShowProgress(generator)]
    )

    # Save the model
    save_model(model, save_path=f"saved_models/{args.model}.pkl")