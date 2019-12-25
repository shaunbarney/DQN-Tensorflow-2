import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, input_shape, hidden_units, num_actions):
        super(Model, self).__init__()
        self.input_layer = tf.keras.Input(shape=(input_shape,))
        self.hidden_layers = []
        for units in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(units, activation='relu'))
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')

    @tf.function
    def call(self, inputs):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def copy_weights(self, model):
        for v1, v2 in zip(self.trainable_variables, model.trainable_variables):
            v1.assign(v2)