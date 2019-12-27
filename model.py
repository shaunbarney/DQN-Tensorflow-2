import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, hidden_units, num_actions):
        super(Model, self).__init__()
        self.hidden_layers = []
        for i, units in enumerate(hidden_units):
            self.hidden_layers.append(tf.keras.layers.Dense(units, activation='relu', name=f"D_{i}"))
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear', name="Output")

    @tf.function
    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def copy_weights(self, model):
        for v1, v2 in zip(self.trainable_variables, model.trainable_variables):
            v1.assign(v2)