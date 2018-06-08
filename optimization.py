import os
import imageio
import subprocess
import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import timeit
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import Dataset
from model import ConvNet
from matplotlib.collections import PatchCollection

#######################################
# MODEL TRAINING AND OPTIMIZATION     #
#######################################

with tf.device('/cpu:0'):
    tf.reset_default_graph()
    sess = tf.Session()
    model = ConvNet(360, 490, sess, 1e-3)
    sess.run(tf.global_variables_initializer())

epochs = 800
losses = []
acc_val = []
train_set = Dataset(name='hdf5/training_5k.hdf5', shuffle=True, batch_size=100)
val_set = Dataset(name='hdf5/validation_1k.hdf5', shuffle=True)

for ep in range(epochs):
    loss = 0
    with tf.device('/device:GPU:0'):
        for t0, (x_batch, y_batch) in enumerate(train_set):
            feed_dict={model.image: x_batch, model.labels: y_batch, model.is_training: True}
            loss_t, _ = sess.run([model.loss, model.train_step], feed_dict)
            loss += loss_t

    accuracy = 0
    for t1, (xval_batch, yval_batch) in enumerate(val_set):
        feed_dict = {model.image: xval_batch, model.labels: yval_batch, model.is_training: False}
        accuracy += sess.run(model.accuracy, feed_dict)
        if t1 == 2: break

    losses.append(loss / (t0+1))
    acc_val.append(accuracy / (t1+1))
    if ep % 10 == 0:
        print("Epoch %d, loss: %f, val acc: %f" % (ep, loss / (t0+1), accuracy / (t1+1)))

#######################################
# VISUALIZING RESULTS                 #
#######################################

plt.plot(losses)

def transparent_cmap(cmap, N=255):
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.7, N+4)
    return mycmap

def plot_heatmap(img, label):
    probs = tf.sigmoid(model.scores)
    heats = sess.run(probs, {model.image: img, model.labels: label, model.is_training: False})[0]
    heats = heats * (heats > 0.5)

    mycmap = transparent_cmap(plt.cm.Reds)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(np.abs(img[0]-255))
    cb = ax.contourf(heats[:,:,0], 12, cmap=mycmap)
    plt.colorbar(cb)

img = train_set.X[3050][None,:,:,:]
label = train_set.Y[3050][None,:,:,:]
plot_heatmap(img, label)

show_img = np.abs(255-img[0])
plt.figure()
plt.imshow(show_img*label[0])
plt.figure()
plt.imshow(show_img)

#######################################
# EVALUATION AND METRICS              #
#######################################
acc = [0, 0]
prec = [0, 0]
rec = [0, 0]
iou = [0, 0]

for t1, (x_batch, y_batch) in enumerate(train_set):
    feed_dict = {model.image: x_batch, model.labels: y_batch, model.is_training: False}
    (prec_t, rec_t, iou_t) = sess.run(model.metrics, feed_dict)
    acc[0] += sess.run(model.accuracy, feed_dict)
    prec[0] += prec_t
    rec[0] += rec_t
    iou[0] += iou_t
    if t1 % 10 == 0:
        print("train set t=%d" % t1)


for t2, (x_batch, y_batch) in enumerate(val_set):
    feed_dict = {model.image: x_batch, model.labels: y_batch, model.is_training: False}
    (prec_t, rec_t, iou_t) = sess.run(model.metrics, feed_dict)
    acc[1] += sess.run(model.accuracy, feed_dict)
    prec[1] += prec_t
    rec[1] += rec_t
    iou[1] += iou_t
    if t2 % 10 == 0:
        print("dev set t=%d" % t2)

for x in (acc, prec, rec, iou):
    x[0] /= (t1+1)
    x[1] /= (t2+1)

print([acc, prec, rec, iou])

#######################################
# NETWORK VISUALIZATION               #
#######################################
def compute_saliency_maps(X, y, model):
    correct_scores = tf.multiply(model.scores, model.labels)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones(y.shape), logits=correct_scores))
    gradient_tensor = tf.gradients(loss, model.image)
    feed_dict = {model.labels: y, model.image: X, model.is_training: False}
    gradient = sess.run(gradient_tensor, feed_dict=feed_dict)[0]

    saliency = np.max(np.abs(gradient), axis=3)
    return saliency

def show_saliency_maps(X, y, mask):
    mask = np.asarray(mask)
    Xm = X[mask]
    ym = y[mask]

    saliency = compute_saliency_maps(Xm, ym, model)

    for i in range(mask.size):
        plt.subplot(2, mask.size, i + 1)
        plt.imshow(np.abs(Xm[i]-255))
        plt.axis('off')
        plt.title("Saliency")
        plt.subplot(2, mask.size, mask.size + i + 1)
        plt.title(mask[i])
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(10, 4)
    plt.show()

X_sal = np.zeros((5, 360, 490, 3))
Y_sal = np.zeros((5, 360, 490, 1))
j = 555
for i in range(5):
    X_sal[i] = train_set.X[i*1000+j]
    Y_sal[i] = train_set.Y[i*1000+j]

mask = np.arange(5)
show_saliency_maps(X_sal, Y_sal, mask)
