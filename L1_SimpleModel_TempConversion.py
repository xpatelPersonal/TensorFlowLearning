import tensorflow as tf
import numpy as np
import logging

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Create two numpy lists celsius and fahrenheit to train the model

celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38], dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100], dtype=float)

# Iterate through list and show alignment between list values

for i,c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))
print("\n")


# ML Terminology

# Feature - the inputs to our model, in this case the degrees in Celsius
# Labels  - the output our model predicts, in this case the degrees in Fahrenheit
# Example - a pair of inputs/outputs used during training, in our case the values from celsius_q and fahrenheit_a at a specific index, like (22, 72)

# Create the model via a Dense Network using a single layer and a single neuron
# Build a layer
#       input_shape=[1] - this specifies that the input to this layer is a singular value. the shape is a one-dimensional array with one number. since this is the first and only layer, the input shape is the input shape of the entire model, the isngle value is a floating point number, representing degrees Celsius
#       units=1         - this specifies the number of neurons in the layer; the number of neurons defines how many internal variables the layer has to try to learn how to solve the problem; since this is the final layer, it is also the size of the model's output - a single gloat value represnting degrees fahrenheit
#               in a multi-layered network, the size and shape of the layer would need to match the input_shape of the next layer

l0 = tf.keras.layers.Dense(units = 1, input_shape=[1])
print("Dense Layer Created for Network\n")

# Assemble layers into the model
#       Once layers are defined, they need to be assembled into a model. The sequential model definition takes a list of layers as an argument, specifying the calcualtion order from teh input to the output. Our model in this exercise has a single layer l0

model = tf.keras.Sequential([l0])
print("Dense Layer Added to Sequential Model\n")

# Compile the model, with loss and optimizer functions
#       Loss Function      - a way of measuring how far off predictions are from the desired outcome
#       Optimizer Function - a way of adjusting internal values in order to reduce the loss

model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Model Training Complete\n")

# Add MatLab package to plot Epochs and Loss Magnitude
import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

# Print Model Prediction for 100 Degrees Celsius
print("Print Model Prediction for 100 Degrees Celsius")
print(model.predict([100.0]))

# Print Internal Variables of the Dense Layer
print("Dense Layer Variables: {}".format(l0.get_weights()))
