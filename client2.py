# import flwr as fl
# import tensorflow as tf

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# model = tf.keras.Sequential(
#   [
#       tf.keras.Input(shape = (32, 32, 3)), 
#       tf.keras.layers.Conv2D(32, kernel_size = (3, 3), activation = 'relu'),
#       tf.keras.layers.Flatten(), 
#       tf.keras.layers.Dense(10, activation = "softmax")
#   ])
# model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# class CifarClient(fl.client.NumPyClient):
#     def get_parameters(self, config):
#         return model.get_weights()

#     def fit(self, parameters, config):
#         model.set_weights(parameters)
#         model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
#         return model.get_weights(), len(x_train), {}

#     def evaluate(self, parameters, config):
#         model.set_weights(parameters)
#         loss, accuracy = model.evaluate(x_test, y_test)
#         return loss, len(x_test), {"accuracy": float(accuracy)}


# fl.client.start_numpy_client(server_address="100.64.69.60:8080", client=CifarClient())

import argparse
import os
from pathlib import Path

import tensorflow as tf
import flwr as fl
import numpy as np
import keras
from glob import glob
from ast import literal_eval
from dataloader import DataGenerator
import datetime
from keras.optimizers import SGD



# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class SteeringClient(fl.client.NumPyClient):
        def __init__(self, model, traingen, valgen):
                self.model = model
                self.traingen = traingen
                self.valgen = valgen

        def get_properties(self, config):
                """Get properties of client."""
                raise Exception("Not implemented")

        def get_parameters(self, config):
                """Get parameters of the local model."""
                raise Exception("Not implemented (server-side parameter initialization)")

        def fit(self, parameters, config):
                """Train parameters on the locally held training set."""

                # Update local model parameters
                self.model.set_weights(parameters)

                # Get hyperparameters for this round
                batch_size: int = config["batch_size"]
                epochs: int = config["local_epochs"]

                log_dir = "../logs/client2/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

                # Train the model using hyperparameters from config
                history = self.model.fit_generator(self.traingen, batch_size, epochs, validation_data=self.valgen, callbacks = [tensorboard_callback] )

                # Return updated model parameters and results
                parameters_prime = self.model.get_weights()
                num_examples_train = len(self.traingen)
                results = {
                        "loss": history.history["loss"][0],
                        "accuracy": history.history["accuracy"][0],
                        "val_loss": history.history["val_loss"][0],
                        "val_accuracy": history.history["val_accuracy"][0],
                }
                return parameters_prime, num_examples_train, results

        def evaluate(self, parameters, config):
                """Evaluate parameters on the locally held test set."""

                # Update local model with global parameters
                self.model.set_weights(parameters)

                # Get config values
                steps: int = config["val_steps"]

                # Evaluate global model parameters on the local test data and return results
                loss, accuracy = self.model.evaluate(self.valgen)
                num_examples_test = len(self.valgen)
                return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
        # Parse command line argument `partition`
        parser = argparse.ArgumentParser(description="Flower")
        parser.add_argument("--data_path", type=str, required=True)
        args = parser.parse_args()


        # Load image paths and label path to a variable
        list_image_paths = glob(f'{args.data_path}/*.jpg')
        list_train_image_paths = list_image_paths[:90]
        list_val_image_paths = list_image_paths[-10:]


        with open(f'{args.data_path}/angles.txt', 'r') as angle_txt: angles_list = angle_txt.readlines()
        dict_angles = { _.split(' ')[0]: literal_eval(_.split(' ')[1].replace('\n', '')) for _ in angles_list } 

        # Load and compile Keras model
        # model = tf.keras.Sequential(
        #     [
        #         tf.keras.Input(shape = (32, 32, 3)), 
        #         tf.keras.layers.Conv2D(32, kernel_size = (3, 3), activation = 'relu'),
        #         tf.keras.layers.Flatten(), 
        #         tf.keras.layers.Dense(1)
        #     ])

        base_model = tf.keras.applications.ResNet50(
            include_top=True,
            input_shape= (224, 224, 3)
        )

        for layer in base_model.layers[:10]:
            layer.trainable = False

        x = base_model.output
        out_pred = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs = base_model.input, outputs= out_pred)
        sgd = SGD(lr=0.001, momentum=0.8, nesterov=True)

        model.compile(sgd, tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanSquaredError()])

        # Load a subset of CIFAR-10 to simulate the local data partition
        traingen = DataGenerator(list_train_image_paths, 
                                dict_angles, 
                                batch_size = 16, 
                                dim= (224, 224), 
                                n_channels=3, 
                                shuffle=True)

        valgen = DataGenerator(list_train_image_paths, 
                                    dict_angles, 
                                    batch_size = 16, 
                                    dim= (224, 224), 
                                    n_channels=3, 
                                    shuffle=True)


        # Start Flower client
        client = SteeringClient(model, traingen, valgen)

        fl.client.start_numpy_client(
                server_address="192.168.0.101:8080",
                client=client
        )






if __name__ == "__main__":
        main()
