# architecture.py
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.applications import MobileNetV3Small
from keras.optimizers import Adam
import config

class EdgeModel:
    def __init__(self, trial):
        self.trial = trial
        self.model = self.create_model()

    def create_model(self):
        # Hyperparameters
        learning_rate = self.trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
        dropout_rate = self.trial.suggest_float('dropout_rate', 0.1, 0.5)
        dense_neurons = self.trial.suggest_int('dense_neurons', 64, 256)

        # Model architecture
        base_model = MobileNetV3Small(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        base_model.trainable = False
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(dropout_rate)(x)
        x = Dense(dense_neurons, activation='relu')(x)
        predictions = Dense(config.NUM_CLASSES, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, train_data, val_data):
        train_images, train_labels = train_data
        val_images, val_labels = val_data

        from optuna.integration import KerasPruningCallback
        callbacks = [KerasPruningCallback(self.trial, 'val_accuracy')]

        history = self.model.fit(
            train_images, train_labels,
            validation_data=(val_images, val_labels),
            shuffle=True,
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            callbacks=callbacks,
        )
        return history

    def evaluate(self, val_data):
        val_images, val_labels = val_data
        val_accuracy = self.model.evaluate(val_images, val_labels)[1]
        return val_accuracy
