import gdown
import os
from zipfile import ZipFile

import numpy as np
import datetime
import os
import random, shutil
import glob

import warnings
warnings.simplefilter('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import pyplot
from matplotlib.image import imread

from os import makedirs,listdir
from shutil import copyfile
from random import seed
from random import random
import keras 
import tensorflow as tf
print(tf.__version__)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D, Input
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from sklearn import metrics
from keras.applications import vgg16
from keras.callbacks import LearningRateScheduler


sns.set_context('notebook')
sns.set_style('white')

def download_data(link,file_name):
    return gdown.download(link,file_name,quiet=False, fuzzy=True)


def eztraction(file_path,file_name):
    with ZipFile(os.path.join(file_path,file_name)) as file:
        return file.extractall()
    
def imageDataGeneration(path,val_split,batch_size,labels,img_rows): 
    train_path=path+'train'
    test_path=path+'test'

    train_datagen = ImageDataGenerator(
    validation_split = val_split,
    rescale=1.0/255.0,
	width_shift_range=0.1, 
    height_shift_range=0.1, 
    horizontal_flip=True
    )
    train_generator = train_datagen.flow_from_directory(
    directory = train_path,
    classes = labels,
    seed = seed,
    batch_size = batch_size, 
    class_mode='binary',
    shuffle = True,
    target_size=(img_rows, img_rows),
    subset = 'training'
)

    val_datagen = ImageDataGenerator(
    validation_split = val_split,
    rescale=1.0/255.0,
	width_shift_range=0.1, 
    height_shift_range=0.1, 
    horizontal_flip=True
    )
    val_generator = val_datagen.flow_from_directory(
    directory = path,
    classes = labels,
    seed = seed,
    batch_size = batch_size, 
    class_mode='binary',
    shuffle = True,
    target_size=(img_rows, img_rows),
    subset = 'validation'
)

    test_datagen = ImageDataGenerator(
    rescale=1.0/255.0
    )   

    test_generator = test_datagen.flow_from_directory(
    directory = test_path,
    classes = labels,
    class_mode='binary',
    seed = seed,
    batch_size = batch_size, 
    shuffle = True,
    target_size=(img_rows, img_rows)
)
    return train_generator,val_generator,test_generator

def vgg16_model(input_shape,weights):
    vgg=vgg16.VGG16(include_top=False, weights=weights, input_shape=input_shape)
    return vgg

def base_model(model):
    output=model.layers[-1].output
    output=keras.layers.Flatten()(output)
    return Model(model.input, output)

def Denselayers(model,input_shape):
    [layer.name for layer in base_model.layers]
    model.trainable=True
    set_trainable=False

    for layer in model.layers:
        if layer.name in ['block5_conv1', 'block4_conv1']:
            set_trainable=True
        if set_trainable:
            layer.trainable=True
        else:
            layer.trainable=False
    model=Sequential()
    model.add(base_model)
    model.add(Dense(512,activation='relu',input_dim=input_shape))
    model.add(Dropout(0.25))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1,activation='sigmoid')) 
    return model

          
    

class LossHistory(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.losses=[]
    self.lr=[]
    def on_epoch_end(self,batch,logs={}):
      self.losses.append(logs.get('loss'))
      self.lr.append(exp_decay(len(self.losses)))
      print('lr:', exp_decay(len(self.losses)))
def exp_decay(epoch):
  initial_lrate=0.1
  k=0.1
  lrate=initial_lrate*np.exp(-k*epoch)
  return lrate
exp_decay=exp_decay()

def model_train(model,train_generator,val_generator,checkpoint_path): 
    loss_=LossHistory()
    lrate_=LearningRateScheduler(exp_decay)

    keras_callbacks=[EarlyStopping(monitor='loss', patience=5,mode='min',min_delta=0.01), ModelCheckpoint(checkpoint_path, monitor='loss', save_best_only=True, mode='min')]

    callbacks_list_=[loss_,lrate_,keras_callbacks]

    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-5), metrics=['accuracy'])

    return model.fit(train_generator, steps_per_epoch=10,epochs=100, callbacks=callbacks_list_, validation_data=val_generator,validation_steps=10,verbose=1)

def img_to_tensor_(img_dim,test_file_path):
    test_file=glob.glob(test_file_path)
    test_imgs=[keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(img,target_size=img_dim)) for img in test_file]
    test_imgs=np.array(test_imgs)

    test_img_scaled=test_imgs.astype('float32')
    test_img_scaled/=255
    return test_img_scaled
