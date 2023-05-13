import tensorflow as tf
import numpy as np

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_paths, dict_angles, batch_size, dim, n_channels,
                                                     shuffle):
                    'Initialization'
                    self.dim = dim
                    self.batch_size = batch_size
                    self.dict_angles = dict_angles
                    self.list_paths = list_paths
                    self.n_channels = n_channels
                    self.shuffle = shuffle
                    self.on_epoch_end()

    def __len__(self):
                    'Denotes the number of batches per epoch'
                    return int(np.floor(len(self.list_paths) / self.batch_size))

    def __getitem__(self, index):
                    'Generate one batch of data'
                    # Generate indexes of the batch
                    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

                    # Find list of IDs
                    list_paths_temp = [self.list_paths[k] for k in indexes]

                    # Generate data
                    X, y = self.__data_generation(list_paths_temp)

                    return (X, y)

    def on_epoch_end(self):
                    'Updates indexes after each epoch'
                    self.indexes = np.arange(len(self.list_paths))
                    if self.shuffle == True:
                                    np.random.shuffle(self.indexes)

    def __data_generation(self, list_paths_temp):
                    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
                    # Initialization
                    X = np.empty((self.batch_size, *self.dim, self.n_channels))
                    y = np.empty((self.batch_size), dtype=float)

                    # Generate data
                    for i, path in enumerate(list_paths_temp):
                        if path.split('\\')[-1] in self.dict_angles:
                            # Store sample
                            image = tf.keras.preprocessing.image.load_img(path)
                            image_arr = tf.keras.preprocessing.image.img_to_array(image)
                            image_arr = tf.image.resize(image_arr,(self.dim[0], self.dim[1])).numpy()

                            X[i,] = image_arr

                            # Store class
                            y[i] = self.dict_angles[path.split('\\')[-1]]
                            print(path, y[i])

                    return (X, y)