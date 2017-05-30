import numpy as np
import keras.keras as Keras
import skimage
import skimage.io
from skimage.util import random_noise
import matplotlib.pyplot as plt
import os

from LRNN import gen_model

np.random.seed(0x5EED)

model = gen_model((None, None, 3), 'parallel')
model.load_weights('models/model-parallel-denoise-weights-10.h5')

"""
pathname = os.path.join('./data/test2014', filename)
"""
pathnames = ["org_0020_crop.png", "data/test2014/COCO_test2014_000000002418.jpg"]

gauss_var = 0.001
for pathname in pathnames:
    img = skimage.io.imread(pathname)
    img = skimage.img_as_float(img)
    noisy_img = random_noise(img, var=gauss_var)
    denoise_img = model.predict(np.array([noisy_img]))
    denoise_img = denoise_img[0]
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(noisy_img)
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(denoise_img)
plt.show()
