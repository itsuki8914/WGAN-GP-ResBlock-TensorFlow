import os,sys,shutil
import tensorflow as tf
import numpy as np
import argparse
import cv2,math,glob,random,time
import time
import matplotlib.pyplot as plt
from model import *
from btgen import BatchGenerator

DATASET_DIR = "mix"
SAVE_DIR = "model"
SVIM_DIR = "generated"


def tileImage(imgs):
    d = int(math.sqrt(imgs.shape[0]-1))+1
    h = imgs[0].shape[0]
    w = imgs[0].shape[1]
    r = np.zeros((h*d,w*d,3),dtype=np.float32)
    for idx,img in enumerate(imgs):
        idx_y = int(idx/d)
        idx_x = idx-idx_y*d
        r[idx_y*h:(idx_y+1)*h,idx_x*w:(idx_x+1)*w,:] = img
    return r

def foloderLength(folder):
    dir = folder
    paths = os.listdir(dir)
    return len(paths)

def printParam(scope):
    total_parameters = 0
    for variable in tf.trainable_variables(scope=scope):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("{} has {} parameters".format(scope, total_parameters))

def main():
    start = time.time()
    if not os.path.exists(SVIM_DIR):
        os.mkdir(SVIM_DIR)

    img_size = 128
    bs = 36
    z_dim = 64
    critic = 5
    lmd = 10
    seed = 1
    np.random.seed(seed)

    z = tf.placeholder(tf.float32, [bs, z_dim])
    X_real = tf.placeholder(tf.float32, [bs, img_size, img_size, 3])

    X_fake = buildGenerator(z,z_dim=z_dim,img_size=img_size,nBatch=bs)

    sess =tf.Session()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(SAVE_DIR)

    if ckpt: # checkpointがある場合
        last_model = ckpt.model_checkpoint_path # 最後に保存したmodelへのパス
        print ("load " + last_model)
        saver.restore(sess, last_model) # 変数データの読み込み
        print("succeed restore model")
    else:
        print("models were not found")
        init = tf.global_variables_initializer()
        sess.run(init)

    print("%.4e sec took initializing"%(time.time()-start))

    for i in range(100):
        batch_z = np.random.uniform(-1.,+1.,[bs,z_dim]).astype(np.float32)
        g_image = sess.run(X_fake,feed_dict={
                z:batch_z})
        cv2.imwrite(os.path.join(SVIM_DIR,"img_%d-%d.png"%(seed,i)),tileImage(g_image)*127.+127.5)

if __name__ == '__main__':
    main()
