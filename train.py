import tensorflow as tf

def train(model, train_dataset, num_epochs=5):
    # Define the optimizer and loss function
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(num_epochs):
        for images, labels in train_dataset:
            with tf.GradientTape() as tape:
                logits = model(images)  # Forward pass
                loss = loss_fn(labels, logits)  # Compute loss

            # Backward pass
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # Update weights

        print(f'Epoch {epoch + 1}, Loss: {loss.numpy():.4f}')
