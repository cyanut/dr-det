from __future__ import print_function

import numpy as np
import multiprocessing 
import threading
import Queue
from scipy.misc import imread, imresize
from scipy.linalg import svd
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import zoom
import os
import csv
import struct
from nolearn.lasagne import BatchIterator
import time

#the below two are funcs for parallel processing since multiprocessing.Pool.map doesn't take anonymous function

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
            n_per_category: if int, number of samples per category after augmentation in each epoch, else if float, percent of sample in each category
            image_size: tuple of (int, int). Images will be force resized to image_size.
            batch_size: int
            img_transform_funcs: a list of transform functions performed on each image, function takes arguments img: a numpy array of (x, y, n_channel).
            batch_transform_funcs: a list of transform functions performed on each minibatch. The function takes arguments of (Xb, yb).
            is_parallel: Whether img_transform_funcs are performed in parallel. May increase performance if transform functions take more than 0.1s to complete per image.
        '''
        super(ImageBatchIterator, self).__init__(batch_size)
        self.image_size = image_size
        #self.data_queue = multiprocessing.Queue(20)
        self.data_queue = Queue.Queue(20)
        self.iter_process = None
        self.images_by_class = images_by_class  
        self.n_per_category = n_per_category
        self.batch_transform_funcs = batch_transform_funcs
        self.img_transform_funcs = img_transform_funcs
        self.is_parallel = is_parallel

    def __call__(self):
        #Create equal samples across class for the epoch
        data = []

        for img_class, img_list in enumerate(self.images_by_class):
            n_img = len(img_list)
            if type(self.n_per_category) is int:
                n_round = self.n_per_category // n_img
                n_remainder = self.n_per_category % n_img
            elif type(self.n_per_category) is float:
                n_round = int(self.n_per_category)
                n_remainder = int(len(img_list) * (self.n_per_category - n_round))

            this_img_list = img_list * n_round + \
                list(np.random.choice(img_list, n_remainder, replace=False))

            data += [(x, img_class) for x in this_img_list]


        np.random.shuffle(data)
        X, y = zip(*data)
        #y=y[...,None]
        if self.iter_process is not None:
            self.iter_process.join()
        #self.iter_process = multiprocessing.Process(target=self._iter, args=(X, y))
        self.iter_process = threading.Thread(target=self._iter, args=(X,y))
        self.iter_process.setDaemon(True)
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
        self.data_queue.put((None, None))
        #print("hi from child, q =", self.data_queue.qsize())
        #self.data_queue.close()
        #print("queue close")
        #self.data_queue.join_thread()
        #print("join thread")
        return True

    def __iter__(self):
        return self

    def next(self):
        #print("iter!")
        res = self.data_queue.get()
        #print(self.data_queue.qsize())
        if res[0] is None and res[1] is None:
            #print("hi")
            self.iter_process.join()
            raise StopIteration
        else:
            return [np.array(res[0]), np.array(res[1], dtype='int32')]
        
    
    def transform(self, Xb, yb):
        '''
            Xb: list of image file paths.
            yb: numpy array of labels.
        '''
        def _mp_transform(args):
            #maybe in separate process, re-seed
            img, rand_seed = args
            if not rand_seed:
                rand_seed = struct.unpack('I',os.urandom(4))[0]
            np.random.seed(rand_seed)

            for i, f in enumerate(self.img_transform_funcs):
                t = time.time()
                img = f(img)
                #print('step', i, ":", time.time() -t)
            return img
        
        t = time.time()
        nprocs = multiprocessing.cpu_count()
        rand_seeds = np.random.randint(np.iinfo(np.int32).max, size=len(Xb))
        args = zip(Xb, rand_seeds)
        if self.is_parallel:
            Xb = parmap(_mp_transform, args)
        else:
            Xb = map(_mp_transform, args)
        #print("map time", time.time() -t)
        t1 = time.time()
        Xb = np.array(Xb).transpose(0,3,1,2)
        #print("transpose time", time.time() -t1)
        for f in self.batch_transform_funcs:
            Xb, yb = f(Xb, yb)

        #print("minibatch process time:", time.time() -t)
        return Xb, yb



## transform functions and functionals ##
def img_load(img):
    return imread(img).astype('float32')


def img_resize(output_size, transform_type="crop"):
    '''
        A functional returning a size transform function. 
        output_size: tuple of (int, int). The resulting size of the image.
        transform_type: "crop" or "resize". 
    '''
    def _img_resize(img):
        input_size = img.shape
        if transform_type == "crop":
            size_diff = [x - y for x,y in zip(input_size, output_size)]
            start_coord = [np.random.randint(x) for x in size_diff]
            end_coord = [x + y for x,y in zip(start_coord, output_size)]
            img = img[start_coord[0]:end_coord[0], start_coord[1]:end_coord[1]]
        elif transform_type == "resize":
            #zoom_factors = [1,1] + [float(x) / y for x,y in zip(input_size, output_size)]
            #img = zoom(img, zoom_factors)
            img = imresize(img, output_size)
        return img
    return _img_resize

def img_flip(chance=0.5):
    def _img_flip(img):
        if bool(np.random.choice(2, p=[1-chance, chance])):
            return np.flipud(img)
        else:
            return img
    return _img_flip

def img_rotate(angle=360):
    def _img_rotate(img):
        rot_angle = angle * np.random.rand()
        return rotate(img, rot_angle, axes=(1,0), reshape = False, cval=img[0,0,0])
    return _img_rotate

def img_rot90(chance=[.25,.25,.25,.25]):
    def _img_rot90(img):
        k = np.random.choice(4, p=chance)
        return np.rot90(img, k)
    return _img_rot90


def rgb_shift(Xb, yb):
    rgb_shift = 128
    delta = np.random.rand(Xb.shape[0],Xb.shape[1],1,1) * rgb_shift
    Xb += delta
    Xb = Xb.clip(0,255)
    return Xb, yb

def zmuv_normalization(Xb, yb):
    Xb -= Xb.mean(axis=(2,3))[...,None,None]
    Xb /= Xb.std(axis=(2,3))[...,None,None]
    return Xb, yb
## end of transform funcs ##


