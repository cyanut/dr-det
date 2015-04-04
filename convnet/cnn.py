import numpy as np
from lasagne.nonlinearities import softmax
from lasagne import layers
from lasagne.updates import nesterov_momentum,adagrad
from nolearn.lasagne import NeuralNet, BatchIterator
import glob
from itertools import cycle
from scipy.misc import imread, imresize
from scipy.linalg import svd
from scipy.ndimage.interpolation import rotate
import multiprocessing
import os.path
import csv

#for replication
random = np.random.RandomState(59410)


class ImageBatchIterator(BatchIterator):


    def __init__(self, images_by_class, n_per_category, image_size, batch_size):
        '''
            images_by_class: a list of list of image paths arranged by class, so that images_by_class[n] contains all images of class n
            n_per_category: number of samples per category after augmentation in each epoch
            image_size: tuple of (int, int). Images will be force resized to image_size.
            batch_size: int
        '''
        super(ImageBatchIterator, self).__init__(batch_size)
        self.image_size = image_size
        self.data_queue = multiprocessing.Queue()
        self.iter_process = None
        self.images_by_class = images_by_class  
        self.n_per_category = n_per_category

    def __call__(self, X=None, y=None):
        #Create equal samples across class for the epoch
        #parameters are useless, only there to make nolearn happy
        data = []
        for img_class, img_list in enumerate(self.images_by_class):
            n_img = len(img_list)
            n_round = self.n_per_category // n_img
            n_remainder = self.n_per_category % n_img
            this_img_list = img_list * n_round + \
                list(random.choice(img_list, n_remainder, replace=False))
            data += [(x, img_class) for x in this_img_list]


        random.shuffle(data)
        X, y = zip(*data)
        #y=y[...,None]
        self.iter_process = multiprocessing.Process(target=self._iter, args=(X, y))
        self.iter_process.daemon = True
        
        self.iter_process.start()

        self.X = X
        self.y = y

        return self


    def _iter(self, X, y):
        n_samples = len(X)
        bs = self.batch_size
        for i in range((n_samples + bs - 1) // bs):
            sl = slice(i * bs, (i + 1) * bs)
            Xb = X[sl]
            if y is not None:
                yb = y[sl]
            else:
                yb = None
            self.data_queue.put(self.transform(Xb, yb))
        self.data_queue.join()

    def __iter__(self):
        yield self.data_queue.get()
        


    def transform(self, Xb, yb):
        '''
            Xb: list of image file paths.
            yb: numpy array of labels.
        '''
        Xb = np.array([imresize(imread(x), self.image_size).transpose(2,0,1)[:3] for x in Xb], dtype='float32')

        for i in range(Xb.shape[0]):
            is_flip = bool(random.choice(2))
            if is_flip:
                Xb[i] = Xb[i][:,::-1,...]
            angle = 180 * random.rand()
            Xb[i] = rotate(Xb[i], angle, axes=(2,1), reshape = False)

        #probably not very useful after normalization
        rgb_shift = 128
        delta = random.rand(Xb.shape[0],3,1,1) * rgb_shift
        Xb += delta
        Xb = Xb.clip(0,255)


        Xb -= Xb.mean(axis=(2,3))[...,None,None]
        Xb /= Xb.std(axis=(2,3))[...,None,None]
        
        '''
        Xb = imgs.astype("uint8")
        for i in range(Xb.shape[0]):
            mask = cv2.Canny(Xb[i], 50, 150)
            mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1,cv2.BORDER_REPLICATE)
            ret, rect = cv2.floodFill(Xb[i], mask, (0,0), 0)
            ret, rect = cv2.floodFill(Xb[i], mask, (0,imgs.shape[2]-1), 0)
            ret, rect = cv2.floodFill(Xb[i], mask, (imgs.shape[1]-1,0), 0)
            ret, rect = cv2.floodFill(Xb[i], mask, (imgs.shape[1]-1,imgs.shape[2]-1), 0)
        '''
        return Xb, yb


def cnn(train_iterator, test_iterator):
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
        output_num_units=3,#1 
        output_nonlinearity=softmax,#None,

        update_learning_rate=0.01,
        update_momentum=0.6,
        update = nesterov_momentum,
        #update = adagrad,

        #regression=True,
        regression=False,
        max_epochs=240,
        verbose=1,
        batch_iterator_train=ImageBatchIterator(batch_size=batch_size, image_size=image_size),
        batch_iterator_test=ImageBatchIterator(batch_size=batch_size, image_size=image_size),
        )
    return cnn


def get_data_by_class(label_dic):
    '''
    label_dic: a dictionary of {image_path: label}, where labels are int in range(n_class).
    return: a list of lists, where each list contains all the images for one class.
    '''
    n_class = int(max(label_dic.values()) + 1)
    imgs_by_class = []
    for i in range(n_class):
        imgs_by_class.append([])
    for f, l in label_dic.items():
        imgs_by_class[l].append(f)
    return imgs_by_class


def get_dr_data(img_glob, csv_path):
    prefix = os.path.dirname(img_glob)
    postfix = os.path.splitext(img_glob)[1]
    label_file = csv.reader(open(csv_path,'r'))
    #skip csv title
    label_file.next()
    label_dic = {os.path.join(prefix, "{}{}".format(x[0],postfix)):int(x[1]) for x in label_file}
    label_dic = {x: label_dic[x] for x in glob.glob(img_glob)}

    return get_data_by_class(label_dic)


if __name__ == "__main__":


    net = cnn()
    if len(sys.argv) > 1:
        net.load_weights_from(sys.argv[1])
    net.fit(x, y)
    print("saving weights ...")
    net.save_weights_to("./2015-03-25-equal-regression.weights")
    '''
