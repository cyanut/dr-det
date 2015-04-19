from scipy.misc import imread
from scipy.ndimage.interpolation import rotate
import cv2
import matplotlib.pyplot as plt
import numpy as np
from dr_network import get_data_by_class, get_dr_data, split_data
from dr_iterator import ImageBatchIterator, img_load, img_resize, img_flip, img_rotate, rgb_shift, zmuv_normalization, img_rot90
from dr_network import DRNeuralNet, qkappa
from cnn import random
from lasagne import layers
from lasagne.updates import adagrad, nesterov_momentum
from lasagne.nonlinearities import tanh, softmax


def test_iterator(imgs_by_class, image_size=(372, 372)):
    
    data_iter = ImageBatchIterator(\
            imgs_by_class, 
            n_per_category=30, 
            batch_size=30, 
            image_size=image_size, 
            img_transform_funcs=[img_load,
                             img_rot90(),
                             img_flip(),
                             img_resize(output_size=image_size), 
                             ],
            batch_transform_funcs=[]
            )
    data_iter = data_iter()
    data = next(data_iter)
    data = next(data_iter)
    for i, d in enumerate(data_iter):
        #print(i)
        data = d
        #print(data[0].shape)
    #data_iter.iter_process.terminate()
    #data_iter.iter_process.join()

    for i,d in enumerate(zip(*data)):
        d = list(d)
        d[0] -= np.min(d[0])
        d[0] /= np.max(d[0])
        d[0] *= 255
        d[0] = d[0].clip(0, 255)
        d[0] = np.uint8(d[0])
        plt.subplot(5, 6, i+1)
        plt.imshow(d[0].transpose(1,2,0))
        plt.title(d[1])
        plt.axis('off')
    plt.show()

def get_mnist_data():
    import sys
    import pickle
    import glob
    import matplotlib.pyplot as plt
    labels = pickle.load(open('../../mnist/mnist_label.pkl','rb'))
    img_glob_list = ['../../mnist/*.png']
    img_file_list = sorted(sum([glob.glob(x) for x in img_glob_list],[]))
    #img_file_list = img_file_list[:18]
    label_dic = dict(zip(img_file_list, labels))
    return get_data_by_class(label_dic)

def lenet5(train_iter, val_iter, save_network_to =None):    
    lenet5 = DRNeuralNet(
        layers = [
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('hidden1', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 24, 24),
        conv1_num_filters=20, conv1_filter_size=(5,5), pool1_ds=(2,2),
        conv1_nonlinearity = tanh,
        conv2_num_filters=50, conv2_filter_size=(5,5), pool2_ds=(2,2),
        conv2_nonlinearity = tanh,
        hidden1_num_units=400, 
        hidden1_nonlinearity = tanh,
        output_num_units=10, 
        output_nonlinearity=softmax,

        update_learning_rate=0.01,
        #update = nesterov_momentum,
        update = adagrad,
        #update_momentum = 0.9,
        regression=False,
        max_epochs=1000,
        verbose=1,
        batch_iterator_train=train_iter,
        batch_iterator_test=val_iter,
        report_funcs = {"qkappa":qkappa},
        save_network_to=save_network_to,
        )
    return lenet5

def test_network():
    mnist_data = get_mnist_data()
    train, val = split_data(mnist_data, 50)
    img_size = (24,24)

    train_iter = ImageBatchIterator(\
            train,
            n_per_category = 2000,
            batch_size = 30,
            image_size = img_size,
            img_transform_funcs = [img_load, img_resize(output_size=img_size)],
            batch_transform_funcs = [],
            is_parallel = False,
            rand_state = random,
            )
            

    val_iter = ImageBatchIterator(\
            val,
            n_per_category = 50,
            batch_size = 30,
            image_size = (24,24),
            img_transform_funcs = [img_load, img_resize(output_size=img_size)],
            batch_transform_funcs = [],
            is_parallel = False,
            rand_state = random,
            )

    net = lenet5(train_iter, val_iter, save_network_to="/tmp/test.weights")
    net.fit()
            

if __name__ == "__main__":
    #test_iterator(get_dr_data('../../train-372/3[456]?_*.png', "../../trainLabels.csv"))
    #test_iterator(get_dr_data('/tmp/train-400/*.png', "../../trainLabels.csv"))
    #test_iterator(get_mnist_data(), image_size=(25,25))
    test_network()
