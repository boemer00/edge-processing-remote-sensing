# Edge Processing in Extreme Conditions using Remote Sensing Data

![](docs/renato-boemer-satellite-remote-sensing.jpeg)

This project aims to develop a compact neural network model capable of real-time classification of satellite imagery within cloud-based edge computing contexts. Inspired by the demand for urgent and accurate data in the defence and security sectors, this initiative demonstrates the ability to handle satellite data with limited computational resources, with a particular focus on real-world, high-stakes scenarios. This project aims to augment conventional remote sensing methodologies, assuring prompt identification and analysis of crucial features within the imagery, even under the most extreme conditions, such as the Earth's orbit.

## Objective
The core objective is to develop a lightweight, efficient neural network model for real-time satellite image classification on cloud-based edge computing environments.

## Installation
Clone this repository using git:
```$ git clone https://github.com/boemer00/edge-processing-remote-sensing.git```

The primary dataset for this project is sourced from [Kaggle's Satellite Image Classification dataset](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification). Additional datasets and resources may be incorporated to enrich the model training phase.

## Model Architecture and Training

This is a lightweight model using **MobileNetV3Small** as the backbone, adapted for the classification of satellite imagery. The network is fine-tuned to the nuances of edge computing, with hyperparameters optimized via Optuna to strike a balance between speed and accuracy.

### Core Components
- **Base**: MobileNetV3Small, chosen for its efficiency on edge devices.
- **Regularization**: Dropout to prevent overfitting, rate optimized through trials.
- **Classification Head**: Custom dense layers, ending with a softmax for probability distribution across 4 classes.

### Optimization
Optuna facilitated the hyperparameter tuning, focusing on:
- **Learning Rate**: Ensuring efficient convergence.
- **Dropout Rate & Dense Neurons**: Tailored to the dataset and computational constraints.

### Training
The model underwent training with augmentation techniques applied using OpenCV. I employed a validation split to guide the early stopping and model checkpoint callbacks, ensuring an optimal stopping point.

## Project Update
This project is in development (last updated: 30 Oct 23)
