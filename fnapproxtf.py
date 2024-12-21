import tensorflow as tf
class fnapproxtf(tf.keras.Model):
    def __init__(self, state_size=5, action_size=1, hidden_size=10, learning_rate=0.01):
        super(fnapproxtf, self).__init__()
        
        # Define layers
        self.fc1 = tf.keras.layers.Dense(hidden_size, activation='relu', name='fc1')
        self.fc2 = tf.keras.layers.Dense(hidden_size, activation='relu', name='fc2')
        self.output_layer = tf.keras.layers.Dense(action_size, activation=None, name='output')  # Linear output layer

        # Define optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def call(self, inputs):
        """Forward pass of the QNetwork"""
        x = self.fc1(inputs)
        x = self.fc2(x)
        return self.output_layer(x)

    def train_step(self, states, actions, targets):
        """Train the network for one step"""
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self(states)

            # Get the Q-values for the taken actions
            action_masks = tf.one_hot(actions, predictions.shape[1])
            fn_values = tf.reduce_sum(predictions * action_masks, axis=1)

            # Compute the loss
            loss = tf.reduce_mean(tf.square(targets - fn_values))

        # Backpropagation
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss