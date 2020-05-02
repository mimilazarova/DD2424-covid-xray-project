from keras.models import Sequential
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
# https://keras.io/getting-started/sequential-model-guide/
# https://keras.io/layers/core/#dense
# https://keras.io/models/sequential/

class Classifier:
    @staticmethod
    def build(width, height, depth, classes):
        
        model = Sequential()
        inputShape = (height, width, depth) # if using Tensorflow
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width) # if using Theano
        
        model.add(Conv2D(16, (3, 3), strides=(1, 1), input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        model.add(Conv2D(32, (3, 3), strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        model.add(Conv2D(64, (3, 3), strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))
        
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        #print("summary")
        #print(model.summary())
        
        return model
    
    

