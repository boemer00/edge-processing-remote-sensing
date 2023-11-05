# Edge Processing in Extreme Conditions using Remote Sensing Data

![Satellite Remote Sensing](docs/renato-boemer-satellite-remote-sensing.jpeg)

This project develops a compact neural network model for real-time satellite imagery classification within cloud-based edge computing contexts, tailored for high-stakes scenarios such as the defense and security sectors.

## Table of Contents
- [Objective](#objective)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture and Training](#model-architecture-and-training)
- [License](#license)
- [Contact](#contact)
- [Project Status](#project-status)

## Objective
The core objective is to develop a lightweight, efficient neural network model for real-time satellite image classification on cloud-based edge computing environments.

## Installation
Clone the repository and set up a virtual environment:

$ git clone https://github.com/boemer00/edge-processing-remote-sensing.git
$ cd edge-processing-remote-sensing
$ python -m venv venv
$ source venv/bin/activate (Linux) or venv\Scripts\activate (Windows)

Install the package using `setup.py`:
`pip install .`

## Usage
Provide clear instructions on how to use the project, including how to run scripts and tests.

## Model Architecture and Training
The project employs a **MobileNetV3Small**-based architecture, known for its lightweight structure and efficacy on edge devices.

### Hyperparameter Optimisation
The model's hyperparameters have been fine-tuned using Optuna, a hyperparameter optimisation framework that systematically searches for the most effective parameter configuration. The focus areas include:

### Training Regimen
The training process adopts data augmentation strategies to enrich the dataset, enhancing the model's ability to generalise from limited samples. Techniques such as rotation, translation, and scaling are applied to simulate a variety of operational scenarios.

### Evaluation and Metrics
Model performance is evaluated using **accuracy** as the training set has equal number of images for each class.

## License
This project is licensed under the [MIT License](LICENSE.txt).

## Contact
For questions or feedback, please reach out to [Renato Boemer](https://www.linkedin.com/in/renatoboemer/).

## Project Status
This project is currently in development.
Last updated on November 5, 2023.
