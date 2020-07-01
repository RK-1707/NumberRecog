import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('mnist.h5')

# predicting images
img= cv2.imread('imgfile', 0) 
#resize image to 28x28 pixels
print(img.size)
img_resized = img.resize(28,28)
img_resized = np.array(img)
print(img.size)
img_reshaped = img_resized.reshape(1,28,28,1)
img_reshaped = img_reshaped/255.0

#predicting the class
res = model.predict([img_reshaped])[0]
digit, accuracy= np.argmax(res), max(res)
print('digit= ' + str(digit), 'and certainty= ' + str(accuracy))
