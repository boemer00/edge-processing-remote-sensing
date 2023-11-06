# Edge Processing in Extreme Conditions using Remote Sensing Data

![Satellite Remote Sensing](docs/renato-boemer-satellite-remote-sensing.jpeg)

This project develops a compact neural network model for real-time satellite imagery classification within cloud-based edge computing contexts. Edge computing refers to the decentralised processing of data, closer to the source of data generation, which facilitates quick response times and reduces the bandwidth needed for data transmission. While edge computing is relevant for the defence and security sectors, its utility extends to other domains including environmental monitoring, disaster response, and urban planning. This project is versatile and can have broad applications.

## Table of Contents
- [Objective](#objective)
- [Installation](#installation)
- [Docker Usage](#docker-usage)
- [Model Architecture and Training](#model-architecture-and-training)
- [License](#license)
- [Contact](#contact)
- [Project Status](#project-status)

## Objective
The core objective is to develop a lightweight, efficient neural network model for real-time satellite image classification on cloud-based edge computing environments.

## Installation
This package uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

### Prerequisites

Ensure you have Poetry installed on your system. If not, install it using:

```curl -sSL https://install.python-poetry.org | python3 -```

### Setup
Clone the repository and navigate into the directory:
```git clone https://github.com/boemer00/edge-processing-remote-sensing.git edge```

```cd edge```

Install the dependencies and setup the project using Poetry:
```poetry install```

### Usage
Activate the Poetry environment shell to run scripts:
```poetry shell```

Run the pipeline:
```python src/pipeline/pipeline.py --n_trials=100```

Exit the shell when done:
```exit```

## Docker Usage
The project includes a Dockerfile for building a containerised environment. Install Docker on your machine following the instructions for your OS from [Docker's official site](https://docs.docker.com/get-docker/).

To build the Docker image:

`docker build -t edge-model .`

To run the container:

`docker run -it --rm --name edge-model-container edge-model`

Within the Docker container, you can execute the pipeline.py script as follows:

`python src/pipeline/pipeline.py --n_trials=100`

## Model Architecture and Training
The project employs a **MobileNetV3Small**-based architecture, known for its lightweight structure and efficacy on edge devices.

### Hyperparameter Optimisation
Hyperparameters are optimised using Optuna integrated with MLflow for tracking. The optimisation process is managed through a `pipeline.py` script that puts together data preparation, model training, and hyperparameter searching.

### Training
The training process adopts data augmentation strategies to enrich the dataset, enhancing the model's ability to generalise from limited samples. Techniques such as rotation, translation, and scaling are applied to simulate a variety of operational scenarios.

### Evaluation and Metrics
Model performance is evaluated using **accuracy** as the training set has equal number of images for each class.

## License
This project is licensed under the [BSD](LICENSE).

## Contact
For questions or feedback, please reach out to [Renato Boemer](https://www.linkedin.com/in/renatoboemer/).

## Project Status
This project is currently in development.
Last updated on 6 Novemeber, 2023.
