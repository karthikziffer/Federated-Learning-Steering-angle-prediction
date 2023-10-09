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

The data was partitioned into training, testing, and validation datasets. The evaluation of the trained model occurred at both the clients and the server using the following steps:

1. Prior to sending their local weights, each client performed its own model evaluation.
2. On the server side, all the local weights were aggregated, and the model was evaluated using the server's local data.
3. The resulting global weights were sent back to the clients, and the evaluation process was repeated using these new global weights.


## License

```
MIT License

Copyright (c) [2023] [Karthik Rajendran]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


