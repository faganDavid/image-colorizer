import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,Input,BatchNormalization,LeakyReLU,Conv2DTranspose,Dense,Softmax
from keras.layers.merge import concatenate
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import set_random_seed
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # Filter out annoying WARNING deprecated method logs

set_random_seed(123)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

#global graph
graph = tf.get_default_graph()
sess = tf.Session(graph=graph, config=session_conf)

from tensorflow import keras
from keras import backend as K
import sklearn.neighbors as nn
import random

tf.keras.backend.set_session(sess)
set_random_seed(2)
np.random.seed(1)

channel = 3
epsilon = 1e-8
T = 0.38
img_rows=256
img_cols =256
nb_neighbors = 5

h, w = img_rows // 4, img_cols // 4

q_ab = np.load("npy/pts_in_hull.npy")
nb_q = q_ab.shape[0]

nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)
modelT = keras.models.load_model('keras_zhang_model.hdf5')
image_folder_pred = 'upload'
image_folder_pred_saved = 'upload/prediction'

#https://stackoverflow.com/a/50941282
#with graph.as_default():
def doPrediction(image_name):
    filename = os.path.join(image_folder_pred, image_name)
    print('filename', filename)
    # b: 0 <=b<=255, g: 0 <=g<=255, r: 0 <=r<=255.
    bgr = cv2.imread(filename)
    if bgr is None:
        print('image not found')
        return

    print('Start processing image: {}'.format(filename))
    gray = cv2.imread(filename, 0)
    bgr = cv2.resize(bgr, (img_rows, img_cols), cv2.INTER_CUBIC)
    gray = cv2.resize(gray, (img_rows, img_cols), cv2.INTER_CUBIC)
    # L: 0 <=L<= 255, a: 42 <=a<= 226, b: 20 <=b<= 223.
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    x_test = np.empty((1, img_rows, img_cols, 1), dtype=np.float32)
    x_test[0, :, :, 0] = gray / 255.

    # L: 0 <=L<= 255, a: 42 <=a<= 226, b: 20 <=b<= 223.
    X_colorized = modelT.predict(x_test)

    X_colorized = X_colorized.reshape((h * w, nb_q))

    # Reweight probas
    X_colorized = np.exp(np.log(X_colorized + epsilon) / T)
    X_colorized = X_colorized / np.sum(X_colorized, 1)[:, np.newaxis]

    # Reweighted
    q_a = q_ab[:, 0].reshape((1, 313))
    q_b = q_ab[:, 1].reshape((1, 313))

    X_a = np.sum(X_colorized * q_a, 1).reshape((h, w))
    X_b = np.sum(X_colorized * q_b, 1).reshape((h, w))

    X_a = cv2.resize(X_a, (img_rows, img_cols), cv2.INTER_CUBIC)
    X_b = cv2.resize(X_b, (img_rows, img_cols), cv2.INTER_CUBIC)

    X_a = X_a + 128
    X_b = X_b + 128


    out_lab = np.zeros((img_rows, img_cols, 3), dtype=np.int32)
    out_lab[:, :, 0] = lab[:, :, 0]
    out_lab[:, :, 1] = X_a
    out_lab[:, :, 2] = X_b
    out_L = out_lab[:, :, 0]
    out_a = out_lab[:, :, 1]
    out_b = out_lab[:, :, 2]

    out_lab = out_lab.astype(np.uint8)
    out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)

    out_bgr = out_bgr.astype(np.uint8)

    cv2.imwrite(image_folder_pred_saved + '/{}_image.png'.format(i), gray)
    cv2.imwrite(image_folder_pred_saved + '/{}_gt.png'.format(i), bgr)
    cv2.imwrite(image_folder_pred_saved + '/{}_out.png'.format(i), out_bgr)

K.clear_session()