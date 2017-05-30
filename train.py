import numpy as np
import keras.keras as Keras
from keras.keras.preprocessing.image import ImageDataGenerator
from skimage.util import random_noise
import skimage
import skimage.io
import skimage.util.dtype
import matplotlib.pyplot as plt
import os

from LRNN import gen_model
from data.gen_patch import gen_patch

np.random.seed(0x5EED)

model = gen_model((96, 96, 3), 'parallel')
model.compile(optimizer='Adamax', loss=Keras.losses.mse)
# Keras.utils.plot_model(model, to_file='model.png')

datadir = './data/train2014'

batch_size = 32
patchsize = 96
gauss_var = 0.001
n_epoch = 30

def datagen(pathnames):
    while True:
        np.random.shuffle(pathnames)
        for iter in range(0, len(pathnames), batch_size):
            images = []
            for f in pathnames[iter : iter+batch_size]:
                img = skimage.io.imread(f)
                img = skimage.util.dtype.convert(img, np.float32)
                if img.ndim == 2:
                    img = skimage.color.gray2rgb(img)
                patch = gen_patch(img, patchsize)
                images.append(patch)
            images = np.array(images)
            noisy_img = np.array([random_noise(f, var=gauss_var) for f in images])
            yield (noisy_img, images)

pathnames = [os.path.join(datadir, f) for f in os.listdir(datadir)]
pathnames = pathnames
n_sample = len(pathnames)
for epoch in range(n_epoch):
    model.fit_generator(
        datagen(pathnames),
        steps_per_epoch=int((n_sample + batch_size-1) / batch_size),
        max_q_size=1024,
        initial_epoch=epoch,
        epochs=epoch+1
    )
    model.save_weights('models/model-parallel-denoise-weights-%d.h5' % (epoch+1))
