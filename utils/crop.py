from __future__ import print_function
import numpy as np
from scipy.misc import imread, imsave, imresize
import cv2

def is_bg(i):
    #return np.sum(x) < 1000 or np.max(i) < 10
    #return np.max(i) < 10
    return np.sum(i) / i.size < 5

def crop_img(im_pair, size=(400,400)):
    im_name_in, im_name_out = im_pair
    img = imread(im_name_in)
    l=0
    r=-1
    try:
        while is_bg(img[:,l]):
            l += 1
    except IndexError:
        print()
        print("error over crop", im_name_in)
        print()
        return False

    while is_bg(img[:,r]):
        r -= 1
    img_w = img.shape[1] - l + r
    if img_w < img.shape[1] / 3:
        print()
        print("warning: over crop at", im_name_in)
        print()
    if img.shape[0] < img_w:
        w = img_w - img.shape[0]
        img = np.pad(img, ((int(w/2), w-int(w/2)), (0,0), (0,0)), mode="constant")
    else:
        w = img.shape[0] - img_w
        img = img[int(w/2):-(w-int(w/2))]
    try:
        img = img[:, l:r]
        if size is not None:
            img = imresize(img, size)
        imsave(im_name_out, img)
    except:
        print()
        print("save error", im_name_in)
        print()
    return True


if __name__ == "__main__":
    import sys
    import glob
    import os
    import multiprocessing
    import time

    #glob_in = sys.argv[1:-1]
    f_in = sys.argv[1:-1]
    dir_out = sys.argv[-1]
    #f_in = glob.glob(glob_in)
    f_out = [os.path.join(dir_out, os.path.basename(x).replace("jpeg", "png")) for x in f_in]
    f_list = zip(f_in, f_out)
    f_list = filter(lambda x: not os.path.exists(x[1]), f_list)
    p = multiprocessing.Pool(8)
    chunk = 10
    total = len(f_list)

    m = p.map_async(crop_img, f_list, chunksize=chunk)
    t_start = time.time()
    n_chunk = int(float(total) / chunk)
    est = "N/A"
    while not m.ready():
        done = (n_chunk - m._number_left) * chunk
        t = time.time() - t_start
        if done > 0:
            est = t / done * (total - done)
            est = int(est)
        t = int(t)
        sys.stderr.write("\r{} of {} done in {} seconds. Estimated {} seconds left".format(done, total, t, est))
        time.sleep(1)
    p.close()
    p.join()
    print()
        
