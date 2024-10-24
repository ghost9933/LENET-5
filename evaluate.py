import tensorflow as tf

def evaluate(model, test_dataset):
    correct = 0
    total = 0

    for images, labels in test_dataset:
        logits = model(images)
        predictions = tf.argmax(logits, axis=1)

        # Cast labels to int64 for compatibility
        labels = tf.cast(labels, tf.int64)

        correct += tf.reduce_sum(tf.cast(tf.equal(predictions, labels), tf.int32)).numpy()
        total += labels.shape[0]

    accuracy = correct / total
    print(f'Accuracy on the test set: {accuracy * 100:.2f}%')
