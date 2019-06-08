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
SVIM_DIR = "sample"


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
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    if not os.path.exists(SVIM_DIR):
        os.mkdir(SVIM_DIR)

    img_size = 128
    bs = 32
    z_dim = 64
    critic = 3
    lmd = 10

    datalen = foloderLength(DATASET_DIR)

    # loading images on training
    batch = BatchGenerator(img_size=img_size,imgdir=DATASET_DIR)

    id = np.random.choice(range(datalen),bs)

    IN_ = batch.getBatch(bs,id)[:4]
    IN_ = (IN_ + 1)*127.5
    IN_ =tileImage(IN_)

    cv2.imwrite("{}/input.png".format(SVIM_DIR),IN_)

    z = tf.placeholder(tf.float32, [bs, z_dim])
    X_real = tf.placeholder(tf.float32, [bs, img_size, img_size, 3])

    X_fake = buildGenerator(z,z_dim=z_dim,img_size=img_size,nBatch=bs)
    fake_y = buildDiscriminator(y=X_fake,nBatch=bs,isTraining=True)
    real_y = buildDiscriminator(y=X_real,nBatch=bs,reuse=True,isTraining=True)
    d_loss_real = -tf.reduce_mean(real_y)
    d_loss_fake = tf.reduce_mean(fake_y)
    g_loss      = -tf.reduce_mean(fake_y)

    epsilon = tf.random_uniform(shape=[bs, 1, 1, 1],minval=0.,maxval=1.)
    X_hat = X_real + epsilon * (X_fake - X_real)
    D_X_hat = buildDiscriminator(X_hat,nBatch=bs,reuse=True,isTraining=False)
    grad_D_X_hat = tf.gradients(D_X_hat, [X_hat])[0]

    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat),axis=[1,2,3]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)


    wd_g = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope="Generator")
    wd_d = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope="Discriminator")

    wd_g = tf.reduce_sum(wd_g)
    wd_d = tf.reduce_sum(wd_d)

    d_loss = d_loss_real + d_loss_fake + lmd * gradient_penalty + wd_d
    d_loss += 0.001 * tf.reduce_mean(tf.square(d_loss_real - 0.0))
    g_loss = g_loss + wd_g

    g_opt = tf.train.AdamOptimizer(2e-4,beta1=0.5).minimize(g_loss, var_list=[x for x in tf.trainable_variables() if "Generator"     in x.name])
    d_opt = tf.train.AdamOptimizer(2e-4,beta1=0.5).minimize(d_loss, var_list=[x for x in tf.trainable_variables() if "Discriminator" in x.name])

    
    printParam(scope="Generator")
    printParam(scope="Discriminator")

    start = time.time()

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.66))

    sess =tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    summary = tf.summary.merge_all()

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
    g_hist = []
    d_hist = []

    start = time.time()
    stable    = np.random.uniform(-1.,+1.,[bs,z_dim]).astype(np.float32)
    for i in range(100001):
        # loading images on training

        for c in range(critic):
                id = np.random.choice(range(datalen),bs)
                batch_images= batch.getBatch(bs,id)
                batch_z        = np.random.uniform(-1.,+1.,[bs,z_dim]).astype(np.float32)
                _,dis_loss = sess.run([d_opt,d_loss],feed_dict={
                            z:batch_z, X_real:batch_images})

        id = np.random.choice(range(datalen),bs)
        batch_images_x = batch.getBatch(bs,id)
        batch_z        = np.random.uniform(-1.,+1.,[bs,z_dim]).astype(np.float32)
        _,gen_loss     = sess.run([g_opt,g_loss],feed_dict={
                        z:batch_z, X_real:batch_images})

        print("in step %s, dis_loss = %.4e, gen_loss = %.4e" %(i,dis_loss, gen_loss))
        g_hist.append(gen_loss)
        d_hist.append(dis_loss)

        if i %100 ==0:
            batch_z = np.random.uniform(-1.,+1.,[bs,z_dim]).astype(np.float32)
            g_image = sess.run(X_fake,feed_dict={
                    z:batch_z})
            cv2.imwrite(os.path.join(SVIM_DIR,"img_%d_fake.png"%i),tileImage(g_image)*127.+127.5)
            g_image = sess.run(X_fake,feed_dict={
                    z:stable})
            cv2.imwrite(os.path.join(SVIM_DIR,"imgst_%d_fake.png"%i),tileImage(g_image)*127.+127.5)

            fig = plt.figure(figsize=(8,6), dpi=128)
            ax = fig.add_subplot(111)
            plt.title("Loss")
            plt.grid(which="both")
            #plt.yscale("log")
            ax.plot(g_hist,label="gen_loss", linewidth = 0.25)
            ax.plot(d_hist,label="dis_loss", linewidth = 0.25)
            plt.xlabel('step', fontsize = 16)
            plt.ylabel('loss', fontsize = 16)
            plt.legend(loc = 'upper right')
            plt.savefig("hist.png")
            plt.close()

            print("%.4e sec took 100steps" %(time.time()-start))
            start = time.time()

        if i%1000==0 :
            saver.save(sess,os.path.join(SAVE_DIR,"model.ckpt"),i)

if __name__ == '__main__':
    main()
