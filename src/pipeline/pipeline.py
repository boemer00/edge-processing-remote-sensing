import argparse

import mlflow.tensorflow
import optuna
from optuna.integration.mlflow import MLflowCallback
import logging

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from model.data_preparation import load_data
from model.model import EdgeModel
from model.train import objective


def main(args):
    # Load data
    train_data, val_data = load_data()

    # Set up MLflow callback for Optuna
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(), metric_name="val_accuracy"
    )

    # Create a study object and specify that we aim to maximise the objective function
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial),
        n_trials=args.n_trials,
        callbacks=[mlflow_callback],
    )

    # Retrieve the best parameters
    best_trial = study.best_trial
    logger.info(f"Best trial: {best_trial.number}")
    logger.info(f"Value: {best_trial.value}")
    logger.info("Params: ")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")

    # Retrain the model with the best parameters
    best_params = best_trial.params
    edge_model = EdgeModel(
        learning_rate=best_params["learning_rate"],
        dropout_rate=best_params["dropout_rate"],
        dense_neurons=best_params["dense_neurons"],
    )
    final_model = edge_model.get_model()

    # Fit the model with the training data
    final_model.fit(
        train_data[0],
        train_data[1],
        validation_data=val_data,
        shuffle=True,
        batch_size=best_params["batch_size"],
        epochs=best_params["epochs"],
    )

    # Save the model in MLflow
    mlflow.tensorflow.log_model(final_model, "model")

    # Optional -- Save the model locally as well
    final_model.save("best_model.h5")

    # Load saved model
    # loaded_model = load_model("best_model.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full model training and optimisation pipeline."
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Number of trials for hyperparameter optimisation.",
    )
    args = parser.parse_args()

    main(args)
