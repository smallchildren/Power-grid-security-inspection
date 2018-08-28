#coding=utf8
import keras
from keras.models import load_weights
model = load_weights('/home/smallchild/keras-yolo3/model_data/trained_weights_final.h5')
model.summary()
