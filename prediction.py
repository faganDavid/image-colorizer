import numpy as np
np.random.seed(1)
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2

from tensorflow import set_random_seed
import tensorflow as tf

from tensorflow import keras
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
tf.get_logger().setLevel('ERROR') # Filter out annoying WARNING deprecated method logs

# https://github.com/tensorflow/tensorflow/issues/28287#issuecomment-495005162
tf_config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

sess = tf.Session(config=tf_config)
graph = tf.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
# Otherwise, their weights will be unavailable in the threads after the session there has been set
set_session(sess)
set_random_seed(2)
model = keras.models.load_model('keras_zhang_model_30K.hdf5')
model_10 = keras.models.load_model('keras_zhang_model.hdf5')

class Config():
    def __init__(self) -> None:
        
        self.epsilon = 1e-8
        self.q_ab = np.load("npy/pts_in_hull.npy")
        self.nb_q = self.q_ab.shape[0]
        self.T = 0.38
        self.img_rows=256
        self.img_cols =256
        self.h, self.w = self.img_rows // 4, self.img_cols // 4
        self.image_folder_pred = 'upload'
        self.image_folder_pred_saved = 'upload/prediction'

c=Config()    

#https://stackoverflow.com/a/50941282
def doPrediction(image_name, modelType):
    global sess
    global graph
    global model
    global c
    with graph.as_default():
        set_session(sess)
        filename = os.path.join(c.image_folder_pred, image_name)
        print('filename', filename)

        # b: 0 <=b<=255, g: 0 <=g<=255, r: 0 <=r<=255.
        bgr = cv2.imread(filename)
        if bgr is None:
            print(filename, 'image not found')
            return

        print('Start processing image: {}'.format(filename))
        gray = cv2.imread(filename, 0)
        bgr = cv2.resize(bgr, (c.img_rows, c.img_cols), cv2.INTER_CUBIC)
        gray = cv2.resize(gray, (c.img_rows, c.img_cols), cv2.INTER_CUBIC)
        
        # L: 0 <=L<= 255, a: 42 <=a<= 226, b: 20 <=b<= 223.
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        a = lab[:, :, 1]
        b = lab[:, :, 2]

        x_test = np.empty((1, c.img_rows, c.img_cols, 1), dtype=np.float32)
        x_test[0, :, :, 0] = gray / 255.

        # L: 0 <=L<= 255, a: 42 <=a<= 226, b: 20 <=b<= 223.
        X_colorized = model.predict(x_test)
        if modelType == '10':
            X_colorized = model_10.predict(x_test)
        else: # 30
            X_colorized = model.predict(x_test)

        X_colorized = X_colorized.reshape((c.h * c.w, c.nb_q))

        # Reweight probas
        X_colorized = np.exp(np.log(X_colorized + c.epsilon) / c.T)
        X_colorized = X_colorized / np.sum(X_colorized, 1)[:, np.newaxis]

        # Reweighted
        q_a = c.q_ab[:, 0].reshape((1, 313))
        q_b = c.q_ab[:, 1].reshape((1, 313))

        X_a = np.sum(X_colorized * q_a, 1).reshape((c.h, c.w))
        X_b = np.sum(X_colorized * q_b, 1).reshape((c.h, c.w))

        X_a = cv2.resize(X_a, (c.img_rows, c.img_cols), cv2.INTER_CUBIC)
        X_b = cv2.resize(X_b, (c.img_rows, c.img_cols), cv2.INTER_CUBIC)

        X_a = X_a + 128
        X_b = X_b + 128


        out_lab = np.zeros((c.img_rows, c.img_cols, 3), dtype=np.int32)
        out_lab[:, :, 0] = lab[:, :, 0]
        out_lab[:, :, 1] = X_a
        out_lab[:, :, 2] = X_b
        out_L = out_lab[:, :, 0]
        out_a = out_lab[:, :, 1]
        out_b = out_lab[:, :, 2]

        out_lab = out_lab.astype(np.uint8)
        out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)

        out_bgr = out_bgr.astype(np.uint8)

        img_pred_saved=c.image_folder_pred_saved + '/{}_out.png'.format(image_name)
        out_bgr= cv2.resize(out_bgr, (300, 300))
        cv2.imwrite(img_pred_saved, out_bgr)

        return img_pred_saved
