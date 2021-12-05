import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.self. pd.read_csv)
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
#from tensorflow.keras.optimizers import Adam


from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,Input,BatchNormalization,LeakyReLU,Conv2DTranspose,Dense,Softmax
from keras.layers.merge import concatenate
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import set_random_seed
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # Filter out annoying WARNING deprecated method logs

from tensorflow import keras

import sklearn.neighbors as nn
import random

np.random.seed(1)

class Predictor:
  def __init__(self):
    self.graph = tf.get_default_graph()
    set_random_seed(2)
    self.session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    self.sess = tf.Session(graph=self.graph, config=self.session_conf)
    

    self.channel = 3
    self.epsilon = 1e-8
    self.T = 0.38
    self.img_rows=256
    self.img_cols =256
    self.nb_neighbors = 5

    self.h, self.w = self.img_rows // 4, self.img_cols // 4

    self.q_ab = np.load("npy/pts_in_hull.npy")
    self.nb_q = self.q_ab.shape[0]

    self.nn_finder = nn.NearestNeighbors(n_neighbors=self.nb_neighbors, algorithm='ball_tree').fit(self.q_ab)
    self.modelT = keras.models.load_model('keras_zhang_model.hdf5')
    self.image_folder_pred = 'upload'
    self.image_folder_pred_saved = 'upload/prediction'
    

#https://stackoverflow.com/a/50941282
def doPrediction(p, image_name):
    with p.graph.as_default(), tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        filename = os.path.join(p.image_folder_pred, image_name)
        print('filename', filename)
        # b: 0 <=b<=255, g: 0 <=g<=255, r: 0 <=r<=255.
        bgr = cv2.imread(filename)
        if bgr is None:
            print(filename, 'image not found')
            return

        print('Start processing image: {}'.format(filename))
        gray = cv2.imread(filename, 0)
        bgr = cv2.resize(bgr, (p.img_rows, p.img_cols), cv2.INTER_CUBIC)
        gray = cv2.resize(gray, (p.img_rows, p.img_cols), cv2.INTER_CUBIC)
        # L: 0 <=L<= 255, a: 42 <=a<= 226, b: 20 <=b<= 223.
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        a = lab[:, :, 1]
        b = lab[:, :, 2]

        x_test = np.empty((1, p.img_rows, p.img_cols, 1), dtype=np.float32)
        x_test[0, :, :, 0] = gray / 255.

        # L: 0 <=L<= 255, a: 42 <=a<= 226, b: 20 <=b<= 223.
        #if not hasattr(p, 'modelT'):
        #    p.modelT = keras.models.load_model('keras_zhang_model.hdf5')
        X_colorized = p.modelT.predict(x_test)

        X_colorized = X_colorized.reshape((p.h * p.w, p.nb_q))

        # Reweight probas
        X_colorized = np.exp(np.log(X_colorized + p.epsilon) / p.T)
        X_colorized = X_colorized / np.sum(X_colorized, 1)[:, np.newaxis]

        # Reweighted
        q_a = p.q_ab[:, 0].reshape((1, 313))
        q_b = p.q_ab[:, 1].reshape((1, 313))

        X_a = np.sum(X_colorized * q_a, 1).reshape((p.h, p.w))
        X_b = np.sum(X_colorized * q_b, 1).reshape((p.h, p.w))

        X_a = cv2.resize(X_a, (p.img_rows, p.img_cols), cv2.INTER_CUBIC)
        X_b = cv2.resize(X_b, (p.img_rows, p.img_cols), cv2.INTER_CUBIC)

        X_a = X_a + 128
        X_b = X_b + 128


        out_lab = np.zeros((p.img_rows, p.img_cols, 3), dtype=np.int32)
        out_lab[:, :, 0] = lab[:, :, 0]
        out_lab[:, :, 1] = X_a
        out_lab[:, :, 2] = X_b
        out_L = out_lab[:, :, 0]
        out_a = out_lab[:, :, 1]
        out_b = out_lab[:, :, 2]

        out_lab = out_lab.astype(np.uint8)
        out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)

        out_bgr = out_bgr.astype(np.uint8)

        #cv2.imwrite(image_folder_pred_saved + '/{}_image.png'.format(image_name), gray)
        #cv2.imwrite(image_folder_pred_saved + '/{}_gt.png'.format(image_name), bgr)

        img_pred_saved=p.image_folder_pred_saved + '/{}_out.png'.format(image_name)
        cv2.imwrite(img_pred_saved, out_bgr)
        return img_pred_saved
    #K.clear_session()