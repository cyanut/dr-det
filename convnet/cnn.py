import numpy as np
from lasagne.nonlinearities import softmax
from lasagne import layers
from lasagne.updates import adagrad
from nolearn.lasagne import NeuralNet, BatchIterator
import glob
from itertools import cycle
from scipy.misc import imread, imresize

#for replication
random = np.random.RandomState(59410)


class ImageBatchIterator(BatchIterator):

    def __init__(self, batch_size, image_size, epoch_shuffle = True):
        '''
            batch_size: int
            image_size: tuple of (int, int). Images will be force resized to image_size.
            epoch_shuffle: bool. whether to shuffle data at each epoch.
        '''
        super(ImageBatchIterator, self).__init__(batch_size)
        self.image_size = image_size
        self.epoch_shuffle = epoch_shuffle

    def __call__(self, X, y=None):
        if self.epoch_shuffle:
            #shuffle X and y in unison: zip, shuffle, then unzip
            data = list(zip(X, y))
            random.shuffle(data)
            X, y = zip(*data)

        return super(ImageBatchIterator, self).__call__(X, y)


    def __iter__(self):
        n_samples = len(self.X)
        bs = self.batch_size
        for i in range((n_samples + bs - 1) // bs):
            sl = slice(i * bs, (i + 1) * bs)
            Xb = self.X[sl]
            if self.y is not None:
                yb = self.y[sl]
            else:
                yb = None
            yield self.transform(Xb, yb)

    def transform(self, Xb, yb):
        '''
            Xb: list of image file paths.
            yb: numpy array of labels.
        '''
        Xb = np.array([imresize(imread(x), self.image_size).transpose(2,0,1) for x in Xb])
        return Xb, yb


def cnn():
    cnn = NeuralNet(
        layers = [
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('hidden1', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 28, 28),
        conv1_num_filters=20, conv1_filter_size=(5,5), pool1_ds=(2,2),
        conv2_num_filters=50, conv2_filter_size=(5,5), pool2_ds=(2,2),
        hidden1_num_units=500, 
        output_num_units=10, 
        output_nonlinearity=softmax,

        update_learning_rate=0.01,
        #update_momentum=0.9,
        update = adagrad,

        regression=False,
        max_epochs=1000,
        verbose=1,
        batch_iterator_train=ImageBatchIterator(batch_size=128, image_size=(28,28)),
        batch_iterator_test=ImageBatchIterator(batch_size=128, image_size=(28,28)),
        )
    return cnn



if __name__ == "__main__":

    import pickle
    import glob
    x = glob.glob('./mnist/*.png')
    x.sort()
    x = np.array(x)
    print(x.dtype)
    y = pickle.load(open('./mnist/mnist_label.pkl'))
    lenet5 = cnn()
    lenet5.fit(x, y)
