import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, data, n_classes=3, shuffle=True, batch_size=32):
        self.data = data
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.indexes = np.arange(len(data))


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


    def __load_img(self, path, IMAGE_SIZE=256):

        image = load_img(path)
        image = img_to_array(image)
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))        
        image = tf.cast(image, tf.float32)
        image = image / 255.
        return image


    def __data_generation(self, indexes):

        images = []
        masks = []
        for idx in indexes:
            image_path, mask_path = self.data[idx]
            image = self.__load_img(image_path)
            mask = self.__load_img(mask_path)
            images.append(image)
            masks.append(mask)

        images = np.array(images)
        masks = np.array(masks)

        return images, masks


    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))


    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        images, masks = self.__data_generation(indexes)
        return images, masks