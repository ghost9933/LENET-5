import tensorflow as tf

def load_data():
    # Load the MNIST dataset directly from TensorFlow
    mnist = tf.keras.datasets.mnist  # Use TensorFlow's built-in dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize and preprocess the dataset
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # Resize images to 32x32 pixels (LeNet-5 expects this size)
    train_images = tf.image.resize(train_images[..., tf.newaxis], [32, 32])  # Add channel dimension
    test_images = tf.image.resize(test_images[..., tf.newaxis], [32, 32])    # Add channel dimension

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(64).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(64).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset
