import tensorflow as tf

# Import TensorFlow Datasets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Helper Libraries
import math
import numpy
import matplotlib.pyplot as plt
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Set Training and Testing Datasets
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

class_names = metadata.features['label'].names
print("Class Names: {}".format(class_names))

num_train_examples = metadata.splits['train'].num_examples
num_test_examples  = metadata.splits['test'].num_examples
print("Number of Training Examples: {}".format(num_train_examples))
print("Number of Test Examples:     {}".format(num_test_examples))


def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

    # The Map Function appoies the normalize function to each element in the train and tests datasets
    train_dataset = train_dataset.map(normalize)
    test_dataset  = test_dataset.map(normalize)

    # The first tiem you use the datset, the images will be loaded from disk
    # Caching will keep them in memory, making the training faster
    train_dataset = train_dataset.cache()
    test_dataset  = test_datatset.cache()

# Take a Single Image, and Remove the Color Dimension by Reshaping
    for image, label in test_dataset.take(1):
        break
    image = image.numpy().reshape((28,28))

    # plot the Image
    plt.figure()
    plt.imshow(image, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.show()

plt.figure(figsize=(10,10))
for i, (image, label) in enumerate(test_dataset.take(25)):
    image = image.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
plt.show()
