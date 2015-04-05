from __future__ import print_function
import numpy as np
from lasagne.nonlinearities import softmax
from lasagne import layers
from lasagne.updates import nesterov_momentum,adagrad
from nolearn.lasagne import NeuralNet, BatchIterator
import glob
from itertools import cycle
from scipy.misc import imread
from scipy.linalg import svd
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import zoom
import multiprocessing
import os.path
import csv
import time
import struct
try:
    import cPickle as pickle
except:
    import pickle

#for replication
random = np.random.RandomState(59410)

def fun(f,q_in,q_out):
    while True:
        i,x = q_in.get()
        if i is None:
            break
        q_out.put((i,f(x)))

def parmap(f, X, nprocs = multiprocessing.cpu_count()):
    q_in   = multiprocessing.Queue(1)
    q_out  = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun,args=(f,q_in,q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i,x)) for i,x in enumerate(X)]
    [q_in.put((None,None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i,x in sorted(res)]


class ImageBatchIterator(BatchIterator):


    def __init__(self, images_by_class, n_per_category, image_size, batch_size, img_transform_funcs=[], batch_transform_funcs=[], is_parallel=False):
        '''
            images_by_class: a list of list of image paths arranged by class, so that images_by_class[n] contains all images of class n
            n_per_category: number of samples per category after augmentation in each epoch
            image_size: tuple of (int, int). Images will be force resized to image_size.
            batch_size: int
            img_transform_funcs: a list of transform functions performed on each image, function takes arguments of (img, random) where img is a numpy array of (x, y, n_channel), random is an instance of np.random.RandomState.
            batch_transform_funcs: a list of transform functions performed on each minibatch. The function takes arguments of (Xb, yb).
            is_parallel: Whether img_transform_funcs are performed in parallel. May increase performance if transform functions take more than 0.1s to complete per image.
        '''
        super(ImageBatchIterator, self).__init__(batch_size)
        self.image_size = image_size
        self.data_queue = multiprocessing.Queue()
        self.iter_process = None
        self.images_by_class = images_by_class  
        self.n_per_category = n_per_category
        self.batch_transform_funcs = batch_transform_funcs
        self.img_transform_funcs = img_transform_funcs
        self.is_parallel = is_parallel

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
        #self.iter_process.daemon = True
        
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
        def _mp_transform(img):
            random = np.random.RandomState(struct.unpack('I',os.urandom(4))[0])
            
            for i, f in enumerate(self.img_transform_funcs):
                t = time.time()
                img = f(img, random)
                print('step', i, ":", time.time() -t)
            return img
        
        t = time.time()
        if self.is_parallel:
            Xb = parmap(_mp_transform, Xb)
        else:
            Xb = map(_mp_transform, Xb)
        print("map time", time.time() -t)
        t1 = time.time()
        Xb = np.array(Xb).transpose(0,3,1,2)
        print("transpose time", time.time() -t1)
        for f in self.batch_transform_funcs:
            Xb, yb = f(Xb, yb)

        print("minibatch process time:", time.time() -t)
        return Xb, yb


#transform functions and functionals
def img_load(img, random):
    return imread(img).astype('float32')


def img_resize(output_size, transform_type="crop"):
    '''
        A functional returning a size transform function. 
        output_size: tuple of (int, int). The resulting size of the image.
        transform_type: "crop" or "resize". 
    '''
    def _img_resize(img, random):
        input_size = img.shape
        if transform_type == "crop":
            size_diff = [x - y for x,y in zip(input_size, output_size)]
            start_coord = [random.randint(x) for x in size_diff]
            end_coord = [x + y for x,y in zip(start_coord, output_size)]
            img = img[start_coord[0]:end_coord[0], start_coord[1]:end_coord[1]]
        elif transform_type == "resize":
            zoom_factors = [1,1] + [float(x) / y for x,y in zip(input_size, output_size)]
            img = zoom(Xb, zoom_factors)
        return img
    return _img_resize

def img_flip(chance=0.5):
    def _img_flip(img, random):
        if bool(random.choice(2, p=[1-chance, chance])):
            return np.flipud(img)
        else:
            return img
    return _img_flip

def img_rotate(angle=360):
    def _img_rotate(img, random):
        rot_angle = angle * random.rand()
        return rotate(img, rot_angle, axes=(1,0), reshape = False, cval=img[0,0,0])
    return _img_rotate

def img_rot90(chance=[.25,.25,.25,.25]):
    def _img_rot90(img, random):
        k = random.choice(4, p=chance)
        return np.rot90(img, k)
    return _img_rot90


def rgb_shift(Xb, yb):
    rgb_shift = 128
    delta = random.rand(Xb.shape[0],Xb.shape[1],1,1) * rgb_shift
    Xb += delta
    Xb = Xb.clip(0,255)
    return Xb, yb

def zmuv_normalization(Xb, yb):
    Xb -= Xb.mean(axis=(2,3))[...,None,None]
    Xb /= Xb.std(axis=(2,3))[...,None,None]
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

    data = get_dr_data("/tmp/train-400/*.png", "../../trainLabels.csv")
    for x in data:
        random.shuffle(x)
    val_data_size = 50
    val_data = [x[:val_data_size] for x in data]
    train_data = [x[val_data_size:] for x in data]
    img_size = (372, 372)
    train_iter = ImageBatchIterator(\
            train_data,
            n_per_category = 2000,
            batch_size = 30,
            image_size = img_size,
            img_transform_funcs = [img_load,
                                   img_rot90(),
                                   img_flip(),
                                   img_resize(output_size=img_size)],
            batch_transform_funcs = [],
            )

    val_iter = ImageBatchIterator(\
            val_data,
            n_per_category = val_data_size,
            batch_size = 30,
            image_size = img_size,
            img_transform_funcs = [img_load,
                                   img_resize(output_size=img_size)]
            batch_transform_funcs=[],
            )
    

    net = cnn(train_iter, val_iter)
    if len(sys.argv) > 1:
        net.load_weights_from(sys.argv[1])
    net.fit(x, y)
    print("saving weights ...")
    pickle.dump(open("test.pkl", cnn))
