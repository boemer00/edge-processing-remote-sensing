import mlflow
import argparse
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.applications import MobileNetV3Small
from keras.optimizers import Adam

# Parameters available to change using the command line
parser = argparse.ArgumentParser(description='Train an Edge Model.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate.')
parser.add_argument('--dense_neurons', type=int, default=128, help='Number of neurons in the dense layer.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
args = parser.parse_args()

NUM_CLASSES = 4

class EdgeModel:
    def __init__(self, learning_rate, dropout_rate, dense_neurons):
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.dense_neurons = dense_neurons
        self.model = self.create_model()

    def create_model(self):
        # Model architecture
        base_model = MobileNetV3Small(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        base_model.trainable = False
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(self.dense_neurons, activation='relu')(x)
        predictions = Dense(NUM_CLASSES, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, train_data, val_data, batch_size, epochs):
        with mlflow.start_run():
            mlflow.log_param('learning_rate', self.learning_rate)
            mlflow.log_param('dropout_rate', self.dropout_rate)
            mlflow.log_param('dense_neurons', self.dense_neurons)
            mlflow.log_param('batch_size', batch_size)
            mlflow.log_param('epochs', epochs)

            train_images, train_labels = train_data
            val_images, val_labels = val_data

            # Assuming KerasPruningCallback is already integrated with MLflow
            from optuna.integration import KerasPruningCallback
            callbacks = [KerasPruningCallback(self.trial, 'val_accuracy')]

            history = self.model.fit(
                train_images, train_labels,
                validation_data=(val_images, val_labels),
                shuffle=True,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
            )

            # Log metrics to MLflow
            mlflow.log_metric('val_accuracy', max(history.history['val_accuracy']))

            # Save the model in MLflow
            mlflow.keras.log_model(self.model, 'model')

        return history

    def evaluate(self, val_data):
        val_images, val_labels = val_data
        val_accuracy = self.model.evaluate(val_images, val_labels)[1]
        return val_accuracy

# Assuming train_data and val_data are loaded properly
train_data = ...
val_data = ...

# Initialise and train the model
edge_model = EdgeModel(args.learning_rate, args.dropout_rate, args.dense_neurons)
edge_model.train(train_data, val_data, args.batch_size, args.epochs)
