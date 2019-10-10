"""
Created on Fri Mar 29 10:35:44 2019

@author: Kritika Nayyar
"""

# Importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.datasets import mnist
import matplotlib.pyplot as plt


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
from keras.utils.np_utils import to_categorical
Y_train = to_categorical(Y_train, num_classes = 10)
Y_test = to_categorical(Y_test, num_classes=10)

# Set the random seed
random_seed = 2

# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

''''
to gnerate a sample img

g = plt.imshow(X_train[0][:,:,0])
'''


"""
CNN Model
"""

# Initialising the CNN
classifier = Sequential()

#first Convolution layer

classifier.add(Conv2D(28, (3, 3), input_shape = (28, 28, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#second convolution layer
classifier.add(Conv2D(28, (3, 3), activation = 'softmax'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(units=128, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=10, activation='softmax'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                   metrics=['accuracy'])
# Fitting the CNN
classifier.evaluate(X_test, Y_test, verbose=0)
classifier.fit(X_train,Y_train, batch_size = 128, epochs = 10, verbose = 1, validation_data = (X_val,Y_val))

#prediction
image_index = 4444
plt.imshow(X_test[image_index].reshape(28, 28),cmap='Greys')
pred = classifier.predict(X_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())
