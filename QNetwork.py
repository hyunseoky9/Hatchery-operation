import tensorflow as tf

class QNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size, hidden_size, layer_num, learning_rate):
        super(QNetwork, self).__init__()
        
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
            q_values = tf.reduce_sum(predictions * action_masks, axis=1)

            # Compute the loss
            loss = tf.reduce_mean(tf.square(targets - q_values))

        # Backpropagation
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

# Example usage
if __name__ == "__main__":
    state_size = 4
    action_size = 2
    hidden_size = 10
    learning_rate = 0.01

    # Instantiate the QNetwork
    q_network = QNetwork(state_size=state_size, action_size=action_size, hidden_size=hidden_size, learning_rate=learning_rate)

    # Example input
    example_states = tf.random.uniform((5, state_size))  # Batch size of 5, state size 4
    example_actions = tf.constant([0, 1, 0, 1, 0])  # Example actions taken
    example_targets = tf.random.uniform((5,))  # Example target Q-values

    # Training step
    loss = q_network.train_step(example_states, example_actions, example_targets)
    print("Training loss:", loss.numpy())

    # Forward pass to get Q-values
    output = q_network(example_states)
    print("Q-values for actions:", output.numpy())
