import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from numpy.core.fromnumeric import reshape
folder = 'C:/Users/Admin/Desktop/photos'
i = np.random.randint(0,1024)
img = Image.open(os.path.join(folder,os.listdir(folder)[i]))
plt.imshow(img)
plt.show()
img = np.array(img)
res = cv2.resize(img,dsize=(600,400),interpolation=cv2.INTER_CUBIC)
img = reshape(res,(240000,3))
channel_red = []
channel_green = []
channel_blue = []
channel = []
for pixel in img:
    channel_red.append(pixel[0])
    channel_green.append(pixel[1])
    channel_blue.append(pixel[2])
channel.append(channel_blue)
channel.append(channel_red)
channel.append(channel_green)
plt.hist(channel)
plt.show()