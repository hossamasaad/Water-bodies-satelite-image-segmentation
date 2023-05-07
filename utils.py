import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback


class ShowProgress(Callback):
    def __init__(self, generator):
        self.generator = generator

    def on_epoch_end(self, epochs, logs=None):
        if (epochs+1) % 5 == 0:
            show_maps(data=self.generator, model=self.model, n_images=3)


def save_model(model, save_path):
    """
    Save model
    Args:
        model (keras.Model): model to save
        save_path (str): path to save model
    """
    with open(save_path, "wb") as file:
        pickle.dump(model, file)


def show_maps(data, n_images=10, model=None, SIZE=(20,10), ALPHA=0.5, explain=False):
    
    # plot Configurations
    if model is not None:
        n_cols = 4
    else:
        n_cols = 3
    
    # Select the Data
    images, label_maps = next(iter(data))
    
    if model is None:
        # Create N plots where N = Number of Images
        for image_no in range(n_images):

            # Figure Size
            plt.figure(figsize=SIZE)

            # Select Image and Label Map 
            id = np.random.randint(len(images))
            image, label_map = images[id], label_maps[id]

            # Plot Image 
            plt.subplot(1, n_cols, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')

            # Plot Label Map
            plt.subplot(1, n_cols, 2)
            plt.imshow(label_map)
            plt.title('Original Label Map')
            plt.axis('off')

            # Plot Mixed Overlap
            plt.subplot(1, n_cols, 3)
            plt.imshow(image)
            plt.imshow(label_map, alpha=ALPHA)
            plt.title("Overlap")
            plt.axis('off')

            # Final Show
            plt.show()

    else:
        # Create N plots where N = Number of Images
        for image_no in range(n_images):

            # Figure Size
            plt.figure(figsize=SIZE)

            # Select Image and Label Map 
            id = np.random.randint(len(images))
            image, label_map = images[id], label_maps[id]
            pred_map = model.predict(image[np.newaxis, ...])[0]

            # Plot Image 
            plt.subplot(1, n_cols, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')

            # Plot Original Label Map
            plt.subplot(1, n_cols, 2)
            plt.imshow(label_map)
            plt.title('Original Label Map')
            plt.axis('off')
            
            # Plot Predicted Label Map
            plt.subplot(1, n_cols, 3)
            plt.imshow(pred_map)
            plt.title('Predicted Label Map')
            plt.axis('off')
            
            # Plot Mixed Overlap
            plt.subplot(1, n_cols, 4)
            plt.imshow(image)
            plt.imshow(pred_map, alpha=ALPHA)
            plt.title("Overlap")
            plt.axis('off')

            # Final Show
            plt.show()