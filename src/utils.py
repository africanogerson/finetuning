import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import imageio
import os
from numpy.random import random_sample
from skimage.exposure import equalize_adapthist
from sklearn.metrics import roc_auc_score
import itertools

def elapsed_time(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    et_str = "{:0>2}h {:0>2}m {:05.2f}s".format(int(hours), int(minutes), seconds)
    return et_str


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def join_yaml_fn(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


def read_image_png(file_name):
    image = np.array(imageio.imread(file_name), dtype=np.float32)
    image = (image - np.amin(image)) / (np.amax(image) - np.amin(image))
    return image


def imclahe(im, clip_limit=0.01):
    if im.dtype is not np.float32:
        im = np.asarray(im, dtype=np.float32)
        im = (im - np.amin(im)) / (np.amax(im) - np.amin(im))
    im_clahe = equalize_adapthist(im, clip_limit=clip_limit)
    im_clahe = np.asarray(im_clahe, dtype=np.float32)
    return im_clahe


def random_rotation(img, angle):

    def random_angle(_angle):
        return (2 * _angle) * random_sample() - _angle

    num_rows, num_cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(
        (num_cols / 2, num_rows / 2),
        random_angle(angle),
        1
    )
    img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
    return img_rotation

def hyper_par(lr, batch):
    lear_r = 10**(-lr)
    h = itertools.product(lear_r, batch)
    return list(h)

# def cross_val_auc(raiz, prefixpath):
#     t_0 = pd.read_csv(raiz+prefixpath+'val_fold0.csv')
#     # t_1 = pd.read_csv(raiz+prefixpath+'val_fold1.csv')
#     # t_2 = pd.read_csv(raiz+prefixpath+'val_fold2.csv')
#     # t_3 = pd.read_csv(raiz+prefixpath+'val_fold3.csv')
#     # t_4 = pd.read_csv(raiz+prefixpath+'val_fold4.csv')
#     # frames = [t_0, t_1, t_2, t_3, t_4]
#     frames = [t_0, t_0]
#     complete = pd.concat(frames)
#     return roc_auc_score(complete.targets, complete.scores)

def cross_val_auc(raiz, prefixpath):
    t_0 = pd.read_csv(raiz+prefixpath+'val_fold0.csv')
    t_1 = pd.read_csv(raiz+prefixpath+'val_fold1.csv')
    t_2 = pd.read_csv(raiz+prefixpath+'val_fold2.csv')
    t_3 = pd.read_csv(raiz+prefixpath+'val_fold3.csv')
    t_4 = pd.read_csv(raiz+prefixpath+'val_fold4.csv')
    frames = [t_0, t_1, t_2, t_3, t_4]

    complete = pd.concat(frames)
    return roc_auc_score(complete.targets, complete.scores)
def cross_test_auc(raiz, prefixpath):
    t_0 = pd.read_csv(raiz+prefixpath+'test_fold0.csv')
    t_1 = pd.read_csv(raiz+prefixpath+'test_fold1.csv')
    t_2 = pd.read_csv(raiz+prefixpath+'test_fold2.csv')
    t_3 = pd.read_csv(raiz+prefixpath+'test_fold3.csv')
    t_4 = pd.read_csv(raiz+prefixpath+'test_fold4.csv')
    frames = [t_0, t_1, t_2, t_3, t_4]
    complete = pd.concat(frames)
    return roc_auc_score(complete.targets, complete.scores)
def cross_train_auc(raiz, prefixpath):
    t_0 = pd.read_csv(raiz+prefixpath+'train_fold0.csv')
    t_1 = pd.read_csv(raiz+prefixpath+'train_fold1.csv')
    t_2 = pd.read_csv(raiz+prefixpath+'train_fold2.csv')
    t_3 = pd.read_csv(raiz+prefixpath+'train_fold3.csv')
    t_4 = pd.read_csv(raiz+prefixpath+'train_fold4.csv')
    frames = [t_0, t_1, t_2, t_3, t_4]
    complete = pd.concat(frames)
    return roc_auc_score(complete.targets, complete.scores)
def cross_train_auc_optimization(raiz, prefixpath):
    # this fucntion was created to evaluate the performance in the training set of the system
    t_0 = pd.read_csv(raiz+'train_folds/'+prefixpath+'train_fold0.csv')
    t_1 = pd.read_csv(raiz+'train_folds/'+prefixpath+'train_fold1.csv')
    t_2 = pd.read_csv(raiz+'train_folds/'+prefixpath+'train_fold2.csv')
    t_3 = pd.read_csv(raiz+'train_folds/'+prefixpath+'train_fold3.csv')
    t_4 = pd.read_csv(raiz+'train_folds/'+prefixpath+'train_fold4.csv')
    frames = [t_0, t_1, t_2, t_3, t_4]
    complete = pd.concat(frames)
    return roc_auc_score(complete.targets, complete.scores)

def cross_t(raiz, prefixpath):
    t_0 = pd.read_csv('/home/bioimag/Documentos/gafricano/code/dataset creators/deepdatset/my_results/cross_val/heatmap/val_fold0.csv')
    t_1 = pd.read_csv('/home/bioimag/Documentos/gafricano/code/dataset creators/deepdatset/my_results/cross_val/heatmap/val_fold1.csv')
    t_2 = pd.read_csv('/home/bioimag/Documentos/gafricano/code/dataset creators/deepdatset/my_results/cross_val/heatmap/val_fold2.csv')
    t_3 = pd.read_csv('/home/bioimag/Documentos/gafricano/code/dataset creators/deepdatset/my_results/cross_val/heatmap/val_fold3.csv')
    t_4 = pd.read_csv('/home/bioimag/Documentos/gafricano/code/dataset creators/deepdatset/my_results/cross_val/heatmap/val_fold4.csv')

    frames = [t_0, t_1, t_2, t_3, t_4]
    complete = pd.concat(frames)
    return roc_auc_score(complete.targets, complete.scores)

def show_case(sample, pause=False, contour=False):
    ax = plt.subplot(221)
    ax.imshow(sample['L-CC'], 'gray')
    if contour:
        xs, ys = sample['con_L-CC'][:, 1], sample['con_L-CC'][:, 0]
        xs = (xs + 1) * 0.5 * sample['L-CC'].shape[1]
        ys = (ys + 1) * 0.5 * sample['L-CC'].shape[0]
        ax.plot(xs, ys, linewidth=2)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.axis('off')
    plt.title('L-CC')

    ax = plt.subplot(222)
    ax.imshow(sample['R-CC'], 'gray')
    if contour:
        xs, ys = sample['con_R-CC'][:, 1], sample['con_R-CC'][:, 0]
        xs = (xs + 1) * 0.5 * sample['R-CC'].shape[1]
        ys = (ys + 1) * 0.5 * sample['R-CC'].shape[0]
        ax.plot(xs, ys, linewidth=2)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.axis('off')
    plt.title('R-CC')

    ax = plt.subplot(223)
    ax.imshow(sample['L-MLO'], 'gray')
    if contour:
        xs, ys = sample['con_L-MLO'][:, 1], sample['con_L-MLO'][:, 0]
        xs = (xs + 1) * 0.5 * sample['L-MLO'].shape[1]
        ys = (ys + 1) * 0.5 * sample['L-MLO'].shape[0]
        ax.plot(xs, ys, linewidth=2)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.axis('off')
    plt.title('L-MLO')

    ax = plt.subplot(224)
    ax.imshow(sample['R-MLO'], 'gray')
    if contour:
        xs, ys = sample['con_R-MLO'][:, 1], sample['con_R-MLO'][:, 0]
        xs = (xs + 1) * 0.5 * sample['R-MLO'].shape[1]
        ys = (ys + 1) * 0.5 * sample['R-MLO'].shape[0]
        ax.plot(xs, ys, linewidth=2)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.axis('off')
    plt.title('R-MLO')
    plt.show()
    if pause:
        plt.pause(0.5)
        plt.clf()


def imhist(img):
    plt.hist(img.ravel(), 256)
    plt.show()

