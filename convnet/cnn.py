import numpy as np
from lasagne.nonlinearities import softmax
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet


if __name__ == "__main__":
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
        input_shape=(None, 1, 28, 28),
        conv1_num_filters=20, conv1_filter_size=(5,5), pool1_ds=(2,2),
        conv2_num_filters=50, conv2_filter_size=(5,5), pool2_ds=(2,2),
        hidden1_num_units=500, 
        output_num_units=10, 
        output_nonlinearity=softmax,

        update_learning_rate=0.01,
        update_momentum=0.9,

        regression=False,
        max_epochs=1000,
        verbose=1
        )

    import pickle
    data = pickle.load(open('./mnist.pkl','rb'))
    x_train, y_train = data[0]
    y_train = y_train.astype("int32")
    x_train = x_train.reshape(-1, 1, 28, 28)
    '''
    y_train_real = np.zeros([y_train.shape[0], 10])
    for i, ind in enumerate(y_train):
        y_train_real[i, ind] = 1
    '''
    lenet5.fit(x_train, y_train)