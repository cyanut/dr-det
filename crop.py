import numpy as np
from scipy.misc import imread, imsave
import cv2

def is_bg(x, i):
    return np.sum(x) < 2000 or np.max(i) < 20

if __name__ == "__main__":
    import sys
    img = imread(sys.argv[1])
    edges = cv2.Canny(img, 30, 120)
    #edges = img * edges[...,None]
    t=0
    b=-1
    l=0
    r=-1
    while is_bg(edges[:,l], img[:,l]):
        l += 1
    while is_bg(edges[:,r], img[:,r]):
        r -= 1
    #print(t,b,l,r, np.sum(edges[t]), np.sum(edges[b]), np.sum(edges[:,l]), np.sum(edges[:,r]))
    img_w = img.shape[1] - l + r
    if img.shape[0] < img_w:
        w = img_w - img.shape[0]
        img = np.pad(img, ((int(w/2), w-int(w/2)), (0,0), (0,0)), mode="constant")
    else:
        w = img.shape[0] - img_w
        img = img[int(w/2):-(w-int(w/2))]

    imsave(sys.argv[2], img[:, l:r])
