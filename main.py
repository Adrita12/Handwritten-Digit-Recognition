import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#load data and then split it into train and test data
# mnist = tf.keras.datasets.mnist #load data
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# #normalize:scaling it down so that every value is between 0 and 1
# #we normalize the pixels here
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)
#
# model = tf.keras.models.Sequential() #basic sequential model
# model.add(tf.keras.layers.Flatten()) #faltten layer converts it into a 784 single layer instead of a 28 * 28 grid
# model.add(tf.keras.layers.Dense(128, activation='relu'))  #add a dense layer- each neuron of one layer connected to other neurons of other layer
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax')) #output layers 10 units represents individual 10 digits
# #softmax - makes sure all the outputs add up to 1 . It gives probablity of each digit to be the right answer
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# #fit/train the model
# model.fit(x_train,y_train, epochs=10)
#
# model.save('handwritten_digit.model')

model = tf.keras.models.load_model('handwritten_digit.model')

# #evaluate the model
# loss, accuracy = model.evaluate(x_test, y_test)
#
# print(loss)
# print(accuracy)

image_number = 1
while os.path.isfile(f'digits/digit{image_number}.jpg'):
    try:
        img = cv2.imread(f'digits/digit{image_number}.jpg')[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1

