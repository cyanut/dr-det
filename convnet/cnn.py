from __future__ import print_function
import numpy as np

#for replication
np.random.seed(59411)

from lasagne.nonlinearities import softmax, tanh
from lasagne import layers
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer

from lasagne.updates import nesterov_momentum,adagrad
from dr_iterator import ImageBatchIterator
from dr_iterator import img_load, img_flip, img_rot90, img_resize, img_rotate
from dr_network import DRNeuralNet
from dr_network import qkappa, confusion_matrix
from dr_network import get_data_by_class, get_dr_data, split_data
import sys

   



def cnn(train_iterator, test_iterator, batch_size, image_size, save_network_to=None):
    cnn = DRNeuralNet(
        save_network_to=save_network_to,
        layers = [
            ('input', layers.InputLayer),
            ('conv1', Conv2DLayer),
            ('pool1', MaxPool2DLayer),
            #('dropout1', layers.DropoutLayer),
            ('conv2', Conv2DLayer),
            ('pool2', MaxPool2DLayer),
            #('dropout2', layers.DropoutLayer),
            ('conv3', Conv2DLayer),
            ('pool3', MaxPool2DLayer),
            #('dropout3', layers.DropoutLayer),
            ('conv4', Conv2DLayer),
            ('pool4', MaxPool2DLayer),
            #('dropout4', layers.DropoutLayer),
            ('hidden1', layers.DenseLayer),
            #('dropout5', layers.DropoutLayer),
            #('hidden2', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, image_size[0], image_size[1]),
        conv1_num_filters=16, conv1_filter_size=(5,5), pool1_ds=(4,4),
        #conv1_nonlinearity=tanh,
        #dropout1_p=0.1,
        conv2_num_filters=64, conv2_filter_size=(5,5), pool2_ds=(4,4),
        #conv2_nonlinearity=tanh,
        #dropout2_p=0.1,
        conv3_num_filters=64, conv3_filter_size=(3,3), pool3_ds=(2,2),
        #conv3_nonlinearity=tanh,
        #dropout3_p=0.2,
        conv4_num_filters=64, conv4_filter_size=(3,3), pool4_ds=(2,2),
        #conv4_nonlinearity=tanh,
        #dropout4_p=0.3,
        #shape after conv layers: 128*4*4
        hidden1_num_units=200, 
        #hidden1_nonlinearity=tanh,
        #dropout5_p=0.5,
        #hidden2_num_units=50,
        #output_num_units=1,
        output_num_units=5,
        output_nonlinearity = softmax,
        #output_nonlinearity=None,

        update_learning_rate=0.01,
        #update_momentum=0.6,
        #update = nesterov_momentum,
        update = adagrad,

        #regression=True,
        regression=False,
        max_epochs=240,
        verbose=1,
        batch_iterator_train=train_iterator,
        batch_iterator_test=test_iterator,
        report_funcs={"qkappa":qkappa, "confusion_matrix":confusion_matrix},
        select_weight_by="qkappa",
        )
    return cnn


if __name__ == "__main__":

    data = get_dr_data("../../train-400/*.png", "../../trainLabels.csv")
    for x in data:
        np.random.shuffle(x)
    val_data_size = 0.05
    train_data, val_data = split_data(data, val_data_size)
    img_size = (372, 372)
    batch_size = 20
    train_iter = ImageBatchIterator(\
            train_data,
            n_per_category = 2000,
            batch_size = batch_size,
            image_size = img_size,
            img_transform_funcs = [img_load,
                                   img_rotate(),
                                   img_flip(),
                                   img_resize(output_size=img_size)],
            batch_transform_funcs = [],
            is_parallel = True,
            )

    val_iter = ImageBatchIterator(\
            val_data,
            n_per_category = 1.0,
            batch_size = batch_size,
            image_size = img_size,
            img_transform_funcs = [img_load,
                                   img_resize(output_size=img_size)],
            batch_transform_funcs=[],
            )
    

    net = cnn(train_iter, val_iter, batch_size, img_size, save_network_to="/tmp/test.cnn")
    if len(sys.argv) > 1:
        net.load_weights_from(sys.argv[1])
    net.fit()
    print("saving weights ...")
    net.save_network()
