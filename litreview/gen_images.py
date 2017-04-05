
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import colorspacious as cspace
from skimage.color import rgba2rgb

# In[2]:

data = plt.imread('calf.png')
rainbow = plt.imread('calf_rainbow_img.png')
rainbow = rgba2rgb(rainbow)
cvd1 = {"name": "sRGB1+CVD", "cvd_type": "deuteranomaly", "severity": 100}
cvd2 = {"name": "sRGB1+CVD", "cvd_type": "protanomaly", "severity": 100}
cvd3 = {"name": "sRGB1+CVD", "cvd_type": "tritanomaly", "severity": 100}

def colormaps():
    plt.imshow(data)
    plt.show()
    
    plt.imshow(data[:,:,0],cmap='jet')
    plt.colorbar()
    plt.show()
    
    plt.imshow(data[:,:,0],cmap='viridis')
    plt.colorbar()
    plt.show()
    
    # use to show differences
    plt.imshow(data[:,:,0],cmap='RdBu')
    plt.colorbar()
    plt.show()
    
    # a very good sequential plot
    plt.imshow(data[:,:,0],cmap='afmhot')
    plt.colorbar()
    plt.show()

def colorblind_rainbow(cvd_type):
    colorblind_rainbow = cspace.cspace_convert(rainbow, cvd_type, "sRGB1")
    # colorblind original
    plt.imshow(np.column_stack((rainbow,np.clip(colorblind_rainbow,0,1))))
    plt.show()

def colorblind_calf(cvd_type):
    colorblind_data = cspace.cspace_convert(data, cvd_type, "sRGB1")
    plt.imshow(np.column_stack((data,np.clip(colorblind_data,0,1))))
    plt.show()

#colormaps()
colorblind_calf(cvd1)
colorblind_calf(cvd2)
colorblind_calf(cvd3)
colorblind_rainbow(cvd1)
colorblind_rainbow(cvd2)
colorblind_rainbow(cvd3)
