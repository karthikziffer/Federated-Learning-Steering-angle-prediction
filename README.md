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
The training process followed a federated approach, wherein the dataset was partitioned into two separate clients. Within each client, a local model was trained, and the resulting local weights were transmitted to the server via the gRPC communication framework. At the server, the local weights were aggregated using the federated average algorithm to obtain global weights, which were then distributed back to the clients. This approach allowed the training data to stay on the clients while enabling a distributed model training process with a federated approach.


## Evaluation







