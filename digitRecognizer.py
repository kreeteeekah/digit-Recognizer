"""
Created on Fri Mar 29 10:35:44 2019

@author: Kritika Nayyar
"""

# Importing the Keras libraries and packages

# For CNN Model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# For Managing CSV Data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import pandas as pd

'''
Part 1: Managing CSV files
'''

# Load Data
train = pd.read_csv("F:\\kaggle competitions\\train.csv")   # write the file path of train.csv
test = pd.read_csv("F:\\kaggle competitions\\test.csv")

Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 

# free some space
del train

g = sns.countplot(Y_train)
Y_train.value_counts()

# Normalize the data
X_train = X_train / 255.0
test = test / 255.0

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
from keras.utils.np_utils import to_categorical
Y_train = to_categorical(Y_train, num_classes = 10)

# Set the random seed
random_seed = 2

# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

'''
to gnerate a sample img

g = plt.imshow(X_train[0][:,:,0])
'''


'''
CNN Model
'''

# Initialising the CNN
classifier = Sequential()

#first Convolution layer
classifier.add(Conv2D(32, (5, 5), input_shape = (28, 28, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#second convolution layer
classifier.add(Conv2D(32, (5, 5), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Dropout 
classifier.add(Dropout(0.25))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units =10,activation='softmax'))


# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])

# Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   rotation_range = 10 )

test_datagen = ImageDataGenerator(rescale = 1./255)

classifier.fit(X_train,Y_train,
                         batch_size = 128,
                         epochs = 12,
                         verbose = 1,
                         validation_data = (X_val,Y_val))