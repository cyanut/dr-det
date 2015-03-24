from cnn import ImageBatchIterator, random, cnn
import numpy as np

from lasagne.nonlinearities import softmax
from lasagne import layers
from lasagne.updates import adagrad
from nolearn.lasagne import NeuralNet, BatchIterator
import glob
from itertools import cycle
from scipy.misc import imread, imresize


def test_iterator():
    import sys
    import pickle
    import glob
    import matplotlib.pyplot as plt
    labels = pickle.load(open('../../mnist/mnist_label.pkl','rb'))
    img_glob_list = ['../../mnist/*.png']
    img_file_list = sorted(sum([glob.glob(x) for x in img_glob_list],[]))
    label_dic = dict(zip(img_file_list, labels))
    data_iter = ImageBatchIterator(batch_size=30, image_size=(50,50))
    data_iter = data_iter(img_file_list, labels)

    d_iter = data_iter.__iter__()
    data = next(d_iter)
    for i,d in enumerate(zip(*data)):
        plt.subplot(5, 6, i+1)
        plt.imshow(d[0].transpose(1,2,0))
        plt.title(d[1])
    plt.show()

    data_iter = data_iter(img_file_list, labels)
    data = next(d_iter)
    for i,d in enumerate(zip(*data)):
        plt.subplot(5, 6, i+1)
        plt.imshow(d[0].transpose(1,2,0))
        plt.title(d[1])
    plt.show()
        
def lenet5():    
    lenet5 = NeuralNet(
        layers = [
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('hidden1', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 37, 37),
        conv1_num_filters=20, conv1_filter_size=(5,5), pool1_ds=(2,2),
        conv2_num_filters=50, conv2_filter_size=(5,5), pool2_ds=(2,2),
        hidden1_num_units=500, 
        output_num_units=10, 
        output_nonlinearity=softmax,

        update_learning_rate=0.01,
        update = adagrad,
        regression=False,
        max_epochs=1000,
        verbose=1,
        batch_iterator_train=ImageBatchIterator(batch_size=5, image_size=(37, 37)),
        batch_iterator_test=ImageBatchIterator(batch_size=5, image_size=(37,37)),
        )
    return lenet5


if __name__ == "__main__":

    import pickle
    import glob
    x = glob.glob('../../mnist/*.png')
    x.sort()
    x = np.array(x)[:1000]
    print(x.dtype)
    y = pickle.load(open('../../mnist/mnist_label.pkl')).astype("int32")[:1000]
    lenet = cnn()
    lenet.fit(x, y)

