# Reference:
# https://github.com/zhixuhao/unet/blob/master/model.py
#
# Changelog:
# 1. Switched from standalone keras to tf.keras
#

import os

import tensorflow as tf

from model import unet
from data import trainGenerator, testGenerator, saveResult


os.environ["TF_CPP_MIN_LOG_LEVEL"]="2" # This is to filter out TensorFlow INFO and WARNING logs
#os.environ["CUDA_VISIBLE_DEVICES"]="0" # Make 1 GPU visible for training


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGene = trainGenerator(batch_size=2, train_path='data/membrane/train',
                        image_folder='image', mask_folder='label',
                        aug_dict=data_gen_args, save_to_dir=None)

model = unet()
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=2000, epochs=5, callbacks=[model_checkpoint])

testGene = testGenerator(test_path='data/membrane/test')
results = model.predict_generator(testGene, 30, verbose=1)
saveResult(save_path='data/membrane/test', npyfile=results)
