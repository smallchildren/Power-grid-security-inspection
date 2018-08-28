#coding=utf8
import keras
keras.__version__
# from keras.models import load_model
# from keras.models import model_save
# #model = model_save('trained_weights_final.h5')
# model = model.load_weights('trained_weights_final.h5')
# model.summary()  # As a reminder.

def create_model():
   model = Sequential()
   model.add(Dense(64, input_dim=14, init='uniform'))
   model.add(LeakyReLU(alpha=0.3))
   model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
   model.add(Dropout(0.5)) 
   model.add(Dense(64, init='uniform'))
   model.add(LeakyReLU(alpha=0.3))
   model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
   model.add(Dropout(0.5))
   model.add(Dense(2, init='uniform'))
   model.add(Activation('softmax'))
   return model

 def load_trained_model(weights_path):
   model = create_model()
   model.load_weights(weights_path)
   model.summary()
   
 load_trained_model('trained_weights_final.h5')