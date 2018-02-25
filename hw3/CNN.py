import argparse
import pickle
import gzip
from collections import Counter, defaultdict
import keras

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPool2D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.core import Reshape


class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # Load the dataset
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)
        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set

class CNN:
    '''
    CNN classifier
    '''
    def __init__(self, train_x, train_y, test_x, test_y, epoches = 15, batch_size=128):
        '''
        initialize CNN classifier
        '''
        self.batch_size = batch_size
        self.epoches = epoches

        # TODO: reshape train_x and test_x
        # NOTE: I am referencing info from
        # https://elitedatascience.com/keras-tutorial-deep-learning-in-python
        # reshape our data from (n, length) to (n, width, height, 1) which width*height = length
        self.train_x = train_x.reshape(train_x.shape[0],28,28,1)
        self.test_x = test_x.reshape(test_x.shape[0],28,28,1)

        # normalize data to range [0, 1]
        self.train_x /= 255
        self.test_x /= 255

        # TODO: one hot encoding for train_y and test_y
        self.train_y = np_utils.to_categorical(train_y, 10)
        self.test_y = np_utils.to_categorical(test_y, 10)
        
        # TODO: build you CNN model
        self.model = Sequential()
        
        self.model.add(Conv2D(32, 3, 3,
            activation='relu', 
            input_shape=(28,28,1)))
        self.model.add(Conv2D(32, 3, 3, activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2,2)))
        
        self.model.add(Conv2D(32, 3, 3, activation='relu'))
		
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))
        
        self.model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        
    def train(self):
        '''
        train CNN classifier with training data
        :param x: training data input
        :param y: training label input
        :return:
        '''
        # TODO: fit in training data
        self.model.fit(self.train_x,
            self.train_y,
            epochs = self.epoches,
            batch_size = self.batch_size)
        pass

    def evaluate(self):
        '''
        test CNN classifier and get accuracy
        :return: accuracy
        '''
        acc = self.model.evaluate(self.test_x, self.test_y)
        return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Restrict training to this many examples')
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")


    cnn = CNN(data.train_x[:args.limit], data.train_y[:args.limit], data.test_x, data.test_y)
    cnn.train()
    acc = cnn.evaluate()
    print(acc)
