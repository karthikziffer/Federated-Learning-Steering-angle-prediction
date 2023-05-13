# import flwr as fl


# # Start Flower server
# fl.server.start_server(
#     server_address="0.0.0.0:8080",
#     config=fl.server.ServerConfig(num_rounds=10),
# )

import argparse
from typing import Dict, Optional, Tuple
from pathlib import Path
import flwr as fl
import tensorflow as tf
from glob import glob
import numpy as np
from ast import literal_eval
from dataloader import DataGenerator


def main() -> None:

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    # model = tf.keras.Sequential(
    #     [
    #         tf.keras.Input(shape = (32, 32, 3)), 
    #         tf.keras.layers.Conv2D(32, kernel_size = (3, 3), activation = 'relu'),
    #         tf.keras.layers.Flatten(), 
    #         tf.keras.layers.Dense(1)
    #     ])

    base_model = tf.keras.applications.ResNet50(
            include_top=True,
            weights="imagenet",
            input_shape= (224, 224, 3)
        )

    for layer in base_model.layers[:10]:
        layer.trainable = False

    x = base_model.output
    out_pred = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs = base_model.input, outputs= out_pred)

    model.compile("adam", tf.keras.losses.MeanSquaredError(), metrics=["accuracy"])

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
            fraction_fit=1,
            fraction_evaluate=0.001,
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
            evaluate_fn=get_evaluate_fn(model, args.data_path),
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
            initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=100),
            strategy=strategy
    )


def get_evaluate_fn(model, data_path):
    """Return an evaluation function for server-side evaluation."""

    # Load image paths and label path to a variable
    list_image_paths = glob(f'{data_path}/*.jpg')
    list_train_image_paths = list_image_paths[:30000]
    list_val_image_paths = list_image_paths[:400]

    with open(f'{data_path}/angles.txt', 'r') as angle_txt: angles_list = angle_txt.readlines()
    dict_angles = { _.split(' ')[0]: literal_eval(_.split(' ')[1].replace('\n', '')) for _ in angles_list } 

    valgen = DataGenerator(list_val_image_paths, 
                                dict_angles, 
                                batch_size = 16, 
                                dim= (224, 224), 
                                n_channels=3, 
                                shuffle=True)


    # The `evaluate` function will be called after every round
    def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(valgen)
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(server_round: int):
        """Return training configuration dict for each round.

        Keep batch size fixed at 32, perform two rounds of training with one
        local epoch, increase to two local epochs afterwards.
        """
        config = {
                "batch_size": 16,
                "local_epochs": 1 if server_round < 2 else 2,
        }
        return config


def evaluate_config(server_round: int):
        """Return evaluation configuration dict for each round.

        Perform five local evaluation steps on each client (i.e., use five
        batches) during rounds one to three, then increase to ten local
        evaluation steps.
        """
        val_steps = 5 if server_round < 4 else 10
        return {"val_steps": val_steps}




if __name__ == "__main__":
        main()
