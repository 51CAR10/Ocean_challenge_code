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


sns.set_context('notebook')
sns.set_style('white')

def download_data(link,file_name):
    return gdown.download(link,file_name,quiet=False, fuzzy=True)


def eztraction(file_path,file_name):
    with ZipFile(os.path.join(file_path,file_name)) as file:
        return file.extractall()
    
def imageDataGeneration(path,val_split,batch_size,labels): 
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




