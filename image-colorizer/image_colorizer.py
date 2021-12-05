from tensorflow import keras
from keras.models import load_model


prediction_model = load_model("model/keras_zhang_model.hdf5")