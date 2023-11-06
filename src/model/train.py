import argparse

import mlflow
import optuna
from optuna.integration import KerasPruningCallback

from model.data_preparation import load_data
from model.model import EdgeModel


def objective(trial):
    # Load data
    train_data, val_data = load_data()

    # Pass the trial to train_model using Optuna to suggest parameters
    val_accuracy = train_model(train_data, val_data, trial)

    # Return the validation accuracy to be maximised
    return val_accuracy


def train_model(train_data, val_data, trial_or_params):
    # Check if we are retraining with known parameters or using Optuna's trial to suggest them
    if isinstance(trial_or_params, dict):
        # Retraining with provided params
        learning_rate = trial_or_params["learning_rate"]
        dropout_rate = trial_or_params["dropout_rate"]
        dense_neurons = trial_or_params["dense_neurons"]
        batch_size = trial_or_params["batch_size"]
        epochs = trial_or_params["epochs"]
    else:
        # Optuna's trial is suggesting parameters
        learning_rate = trial_or_params.suggest_float(
            "learning_rate", 1e-4, 1e-1, log=True
        )
        dropout_rate = trial_or_params.suggest_float("dropout_rate", 0.1, 0.5)
        dense_neurons = trial_or_params.suggest_int("dense_neurons", 64, 256)
        batch_size = trial_or_params.suggest_categorical("batch_size", [16, 32, 64])
        epochs = trial_or_params.suggest_int("epochs", 5, 50)

    # Initialise the model with either suggested parameters or the best-known parameters
    edge_model = EdgeModel(learning_rate, dropout_rate, dense_neurons)
    model = edge_model.get_model()

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(
            {
                "learning_rate": learning_rate,
                "dropout_rate": dropout_rate,
                "dense_neurons": dense_neurons,
                "batch_size": batch_size,
                "epochs": epochs,
            }
        )

        # Train the model
        history = model.fit(
            train_data[0],
            train_data[1],
            validation_data=val_data,
            shuffle=True,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[KerasPruningCallback(trial_or_params, "val_accuracy")]
            if not isinstance(trial_or_params, dict)
            else [],
        )

        # Log metrics and save the model
        val_accuracy = max(history.history["val_accuracy"])
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.keras.log_model(model, "model")

    return history


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train an Edge Model.")
    args = parser.parse_args()

    # Start Optuna optimisation
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
