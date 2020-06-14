import tensorflow as tf
import cv2
import numpy as np
import keras
from keras.datasets import mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images= training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
training_images=training_images / 255.0
test_images= test_images / 255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=1)
test_loss, test_acc = model.evaluate(test_images, test_labels)

# predicting images
img= cv2.imread('download.png', 0)
img = np.array(img)
img = img.reshape(1,28,28,1)
img = img/255.0

#predicting the class
res = model.predict([img])[0]
digit, accuracy= np.argmax(res), max(res)
print(digit, accuracy)