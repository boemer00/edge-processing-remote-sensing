from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.applications import MobileNetV3Small

class EdgeModel:
    NUM_CLASSES = 4

    def __init__(self, learning_rate, dropout_rate, dense_neurons):
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.dense_neurons = dense_neurons
        self.model = self._create_model()

    def _create_model(self):
        base_model = MobileNetV3Small(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        base_model.trainable = False
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(self.dense_neurons, activation='relu')(x)
        predictions = Dense(EdgeModel.NUM_CLASSES, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def get_model(self):
        return self.model

if __name__ == "__main__":
    edge_model = EdgeModel(learning_rate=0.001, dropout_rate=0.3, dense_neurons=128)
    edge_model.get_model().summary()
