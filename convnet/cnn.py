import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
import time
import pickle
import sys

theano.config.floatX = 'float32'
rng = np.random.RandomState(23455)


class ConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, pool_size):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type pool_size: tuple or list of length 2
        :param pool_size: the downsampling (pooling) factor (#rows, #cols)
        """

        print(image_shape, filter_shape)
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.pool_size = pool_size

        fan_in = np.prod(filter_shape[1:])

        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(pool_size))

        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = T.nnet.conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=pool_size,
            ignore_border=True
        )

        # add the bias term. 
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]

class HiddenLayer(object):
    '''Fully connected layer of NN'''
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
            activation=T.nnet.sigmoid):
        self.input = input
        if W is None:
            W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]



class LogisticRegression(object):

    def __init__(self, input, n_in, n_out, W=None, b=None):
        if W is None:
            W = theano.shared(value=np.zeros((n_in, n_out),
                                             dtype=theano.config.floatX),
                                                name='W', borrow=True)
        if b is None:
            b = theano.shared(value=np.zeros((n_out,),
                                             dtype=theano.config.floatX),
                                                name='b', borrow=True)
        self.W = W
        self.b = b

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.output = self.p_y_given_x
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.stack(
                    T.mean(T.neq(self.y_pred, y) * y),
                    T.mean(T.neq(self.y_pred, y) * (1-y)),
                    T.mean(T.neq(self.y_pred, y))
                    )
        else:
            raise NotImplementedError()



class ConvNet(object):
    def __init__(self, rng, image_dim, batch_size, layers_def, l2=0, lr=0.01, momentum=0):
        n_input = np.array([batch_size, 1, image_dim[0], image_dim[1]])
        self.params = []
        self.input = T.tensor4('x') 
        self._targets = T.lvector('y')
        self.layers = []
        self.n_input = n_input
        self.batch_size = batch_size
        self.rng = rng
        for n in layers_def:
            if len(self.layers) < 1:
                layer_input = self.input

            if n["type"] == "convpool":
                filter_shape = n["filter_shape"]
                print(filter_shape, n_input[1])
                filter_shape.insert(1, n_input[1])
                filter_shape = np.array(filter_shape)
                pool_size = n["pool_size"]

                self.layers.append(ConvPoolLayer(\
                        rng = self.rng,
                        input=layer_input,
                        filter_shape= filter_shape,
                        image_shape = n_input,
                        pool_size = pool_size))
                
                #filter_shape = [batch_size, n_kernel, w_kernel, h_kernel]
                #output shape: 
                #[batch_size, n_kernel, (w-w_kernel+1)/pool_w, (h-h_kernel+1)/pool_h]
                n["size"] = np.array([n_input[0], filter_shape[0], (n_input[2] - filter_shape[2] + 1) / pool_size[0], (n_input[3] - filter_shape[3] + 1) / pool_size[1]])

            else:
                if not type(n_input) == int:
                    #previous layer is convpool layer, flatten output
                    n_input = np.prod(n_input[1:])
                    layer_input = layer_input.flatten(2)

                if n["type"] == "sigmoid":
                    self.layers.append(HiddenLayer(rng=rng, 
                                                   input=layer_input, 
                                                   n_in = n_input,
                                                   n_out = n["size"],
                                                   activation = T.nnet.sigmoid
                                                   ))
                elif n["type"] == "softmax":
                    self.layers.append(LogisticRegression(input=layer_input,
                                                n_in = n_input,
                                                n_out = n["size"]))

            self.params += self.layers[-1].params
            layer_input = self.layers[-1].output
            #set output shape as input shape for next layer
            n_input = n["size"]
        

        self._L2_sqr = T.sum([(l.W ** 2).sum() for l in self.layers])
        self.nll = self.layers[-1].negative_log_likelihood
        self._prob = self.layers[-1].p_y_given_x
        self._pred = self.layers[-1].y_pred
        self._err = self.layers[-1].errors
         
        self.train_model = None
        self.validation_model = None
        self.best_params = None
        self.predict_model = theano.function(inputs=[self.input],
            outputs=self._prob,
        )
        self._cost = self.nll(self._targets) + l2 * self._L2_sqr

        self.gparams = []
        for p in self.params:
            gp = T.grad(self._cost, p)
            self.gparams.append(gp)

        self.updates = []
        for param, grad in zip(self.params, self.gparams):
            new_param = param - lr * grad + momentum * param
            self.updates.append((param, new_param))
        

    def load_weights(self, weights):
        n_layers = len(self.layers)
        assert len(weights) == 2 * n_layers, "weights does not match layer number"
        for l, w, b in zip(self.layers, weights[::2], weights[1::2]):
            l.W.set_value(w)
            l.b.set_value(b)


    def train(self, datasets, n_epochs =1000, bulk_size=10000, valid_set_size=10000, test_set_size=10000, low_mem = True):
        #datasets = [(train_X, train_y),(valid_X, valid_y), (test_X, test_y)]
        p_train_pos = np.sum(datasets[0][1]) / 1.0 / datasets[0][1].size
        p_train_neg = 1 - p_train_pos
        print "p train pos:", p_train_pos, "p train neg:", p_train_neg
        p_valid_pos = np.sum(datasets[1][1]) / 1.0 / datasets[1][1].size
        p_valid_neg = 1 - p_valid_pos
        print "p valid pos:", p_valid_pos, "p valid neg:", p_valid_neg


        train_set, valid_set = datasets
        n_train = train_set[1].size
        
        batch_size = self.batch_size
        n_bulk = n_train / bulk_size
        if n_bulk < 1:
            n_bulk = 1
            bulk_size = n_train
        
        index = T.lscalar()

        print "start"
        print(train_set[0].shape, train_set[1].shape)

        train_set_x = theano.shared(train_set[0][:bulk_size], borrow=True)
        train_set_y = theano.shared(train_set[1][:bulk_size], borrow=True)
        valid_set_x = theano.shared(valid_set[0][:valid_set_size], borrow=True)
        valid_set_y = theano.shared(valid_set[1][:valid_set_size], borrow=True)
        print "off to gpu"

        print(index, index*batch_size, batch_size, train_set_x[:2,:3])
        self.train_model = theano.function(
            inputs=[index], 
            outputs=self._cost,
            updates = self.updates,
            givens = {
                self.input: train_set_x[index*batch_size : (index+1)*batch_size], 
                self._targets: train_set_y[index*batch_size : (index+1)*batch_size] 
            }
        )

        self.validation_model = theano.function(
            inputs=[],
            outputs=self._err(self._targets),
            givens = {
                self.input: valid_set_x, 
                self._targets: valid_set_y, 
            }
        )

        print '... training'

        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = bulk_size / batch_size

        best_params = None
        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()

        epoch = 0
        done_looping = False


        iter = 0
        valid_counter = validation_frequency
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for bulk in range(n_bulk):
                print bulk
                if bulk > 0:
                    train_set_x.set_value(train_set[0][bulk * bulk_size:(bulk+1) * bulk_size])
                    train_set_y.set_value(train_set[1][bulk * bulk_size:(bulk+1) * bulk_size])
                n_train_batches = min(n_train - bulk * bulk_size, bulk_size) / batch_size

                for minibatch_index in xrange(n_train_batches):
                    minibatch_avg_cost = self.train_model(minibatch_index)
                    iter += 1
                    if (iter + 1) > valid_counter:
                        valid_counter += validation_frequency

                        validation_res = self.validation_model()
                        pos_err = validation_res[0]
                        neg_err = validation_res[1]
                        this_validation_loss = validation_res[2]

                        print('epoch %i, cost %.6f, minibatch %i/%i, validation error %f %%, pos_err %f %%, neg_err %f %%' 
                                % (epoch, minibatch_avg_cost, n_train_batches * bulk + minibatch_index + 1, n_train_batches * n_bulk, this_validation_loss * 100., pos_err / p_valid_pos * 100., neg_err /p_valid_neg * 100.))
                        print >> sys.stderr, ('epoch %i, v_err %f %%' %(epoch, this_validation_loss * 100))
                        print "patience:", patience, "iter", iter

                        if this_validation_loss < best_validation_loss:
                            patience = max(patience, iter * patience_increase)

                            best_validation_loss = this_validation_loss
                            best_iter = iter
                            best_params = [np.asarray(x.eval()) for x in self.params]
                            '''
                            valid_losses = self.validation_model()

                            print(('     epoch %i, minibatch %i/%i, test error of best model %f %%, pos_err %f %%, neg_err %f %%') %
                                  (epoch, minibatch_index + 1, n_train_batches, valid_losses[2] * 100., valid_losses[0] * 100./p_valid_pos, valid_losses[1] * 100./p_valid_neg))
                            '''

                        if patience <= iter:
                                done_looping = True
                                break

        end_time = time.clock()
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
        return best_params


    def test(self, X):
        return np.asarray(self.predict_model(X))


def nnet_from_param_file(fname):
    param_dic = pickle.load(open(fname))
    layer_def = param_dic['layer_def']
    print layer_def
    x_dim = param_dic['params'][0].shape[0]
    print param_dic
    rng = np.random.RandomState(int(param_dic['rng']))
    nnet = NNet(rng, x_dim, layer_def)
    nnet.load_weights(param_dic['params'])
    return nnet


if __name__ == "__main__":
    import pickle
    data = pickle.load(open("./mnist.pkl","rb"))
    data = [(x[0].reshape([-1,1,28,28]), x[1]) for x in data]
    print(data[0][0].shape)
    test_net_shape = [{"type":"convpool", "filter_shape":[20, 5, 5], "pool_size":[2,2]},
            {"type":"convpool", "filter_shape":[50, 5, 5], "pool_size":[2,2]},
                    {"type":"sigmoid", "size": 500},
                    {"type":"softmax", "size": 10}]
    convnet = ConvNet(rng, data[0][0].shape[2:], 500 , test_net_shape, lr=0.1)
    convnet.train(data[:2], bulk_size=50000)

