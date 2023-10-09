# Flower Framework for Steering Angle Classification

<div align="center">
<img src=https://github.com/karthikziffer/Federated-Learning-Steering-angle-prediction/assets/24503303/658bfaa4-64b2-4748-b25b-b039bc3db6b0" >
</div>


## Overview
This project leverages the Flower Framework for steering angle classification. The Flower Framework is a powerful tool for developing and fine-tuning machine learning models to predict steering angles in autonomous vehicles. Whether you are building a self-driving car or working on a related research project, this framework provides the foundation you need.

## Features
- Flower framework for federated learning.
- Steering angle prediction for autonomous vehicles.
- Efficient model training and evaluation.

## Table of Contents
- Installation
- Data
- Training
- Evaluation
- Contributing
- License


## Installation
The packages are installed using pip package manager and python version is 3.9

```
hydra-core==1.3.2
flwr==1.4.0
keras==2.12.0
numpy==1.23.5
omegaconf==2.3.0
tensorflow==2.12.0
mlflow==2.5.0
```

Install the pip packages from requirements.txt using the below command

```
pip install -r requirements.txt
```

## Data
The [Steering angle classification Dataset](https://www.kaggle.com/datasets/roydatascience/training-car) was obtained from Kaggle. The dataset comprises 24K images and their corresponding steering angle. 


## Training 
The training was done on a federated manner, where the data is divided into two clients. The local model is trained in each of the clients and the trained local weights are sent to the server through the gRPC communication framework. The local weights get averaged in the server using the federated average algorithm and the global weights are sent to the clients. This way the training data remains in the client and a distributed model training happens in a federated manner. 


## Evaluation







