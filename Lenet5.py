import tensorflow as tf

class LeNet5(tf.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Define layers
        self.conv1 = tf.keras.layers.Conv2D(6, kernel_size=5, activation='relu', padding='valid')
        self.pool1 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)

        self.conv2 = tf.keras.layers.Conv2D(16, kernel_size=5, activation='relu', padding='valid')
        self.pool2 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(120, activation='relu')
        self.fc2 = tf.keras.layers.Dense(84, activation='relu')
        self.fc3 = tf.keras.layers.Dense(10)  # 10 classes for output

    def __call__(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
