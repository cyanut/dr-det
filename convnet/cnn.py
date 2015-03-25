import numpy as np
from lasagne.nonlinearities import softmax
from lasagne import layers
from lasagne.updates import nesterov_momentum,adagrad
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
        #print(Xb)
        Xb = np.array([imresize(imread(x), self.image_size).transpose(2,0,1)[:3] for x in Xb], dtype='float32')/255.0
        Xb -= np.mean(Xb, axis=(2,3))[...,None,None]
        return Xb, yb


def cnn():
    image_size = (372, 372)
    batch_size = 20
    cnn = NeuralNet(
        layers = [
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('dropout2', layers.DropoutLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('dropout3', layers.DropoutLayer),
            ('conv4', layers.Conv2DLayer),
            ('pool4', layers.MaxPool2DLayer),
            ('dropout4', layers.DropoutLayer),
            ('hidden1', layers.DenseLayer),
            ('dropout5', layers.DropoutLayer),
            ('hidden2', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, image_size[0], image_size[1]),
        conv1_num_filters=16, conv1_filter_size=(5,5), pool1_ds=(4,4),
        dropout1_p=0.1,
        conv2_num_filters=48, conv2_filter_size=(5,5), pool2_ds=(4,4),
        dropout2_p=0.1,
        conv3_num_filters=144, conv3_filter_size=(3,3), pool3_ds=(2,2),
        dropout3_p=0.2,
        conv4_num_filters=432, conv4_filter_size=(3,3), pool4_ds=(2,2),
        dropout4_p=0.3,
        #shape after conv layers: 432*4*4
        hidden1_num_units=500, 
        dropout5_p=0.5,
        hidden2_num_units=50,
        output_num_units=3, 
        output_nonlinearity=softmax,

        update_learning_rate=0.01,
        update_momentum=0.6,
        update = nesterov_momentum,
        #update = adagrad,

        regression=False,
        max_epochs=240,
        verbose=1,
        batch_iterator_train=ImageBatchIterator(batch_size=batch_size, image_size=image_size),
        batch_iterator_test=ImageBatchIterator(batch_size=batch_size, image_size=image_size),
        )
    return cnn



if __name__ == "__main__":

    import pickle
    import glob
    import csv
    import re
    import sys

    prefix = '/tmp/train-372-classified/'
    postfix = '.png'
    '''
    train_x = np.array(glob.glob('{}*{}'.format(prefix, postfix)))
    label_file = csv.reader(open('../../trainLabels.csv','r'))
    #skip title
    label_file.next()
    label_dic = {x[0]:int(x[1]) for x in label_file}
    train_x_fmt = [x[len(prefix):-len(postfix)] for x in train_x]
    train_y = np.array([label_dic[x] for x in train_x_fmt]).astype("int32")
    print(train_y)
    '''
    n_per_category = 2400
    x = []
    y = []
    for i in range(3):
        x_cat = glob.glob(prefix + str(i) + "/*.png")
        random.shuffle(x_cat)
        x_cat = x_cat[:n_per_category]
        y += [i] * len(x_cat)
        x += x_cat
    x = np.array(x)
    y = np.array(y, dtype="int32")
    net = cnn()
    if len(sys.argv) > 1:
        net.load_weights_from(sys.argv[1])
    net.fit(x, y)
    print("saving weights ...")
    net.save_weights_to("./2015-03-24-equal.weights")
