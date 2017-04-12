
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
#import colorspacious as cspace
#from skimage.color import rgba2rgb
#
## In[2]:
#
data = plt.imread('calf.png')
rainbow = plt.imread('calf_rainbow_img.png')
#rainbow = rgba2rgb(rainbow)
#cvd1 = {"name": "sRGB1+CVD", "cvd_type": "deuteranomaly", "severity": 100}
#cvd2 = {"name": "sRGB1+CVD", "cvd_type": "protanomaly", "severity": 100}
#cvd3 = {"name": "sRGB1+CVD", "cvd_type": "tritanomaly", "severity": 100}

def colormaps():
    plt.imshow(data)
    plt.show()
    
    plt.imshow(data[:,:,0],cmap='summer')
    plt.colorbar()
    plt.show()
    
    plt.imshow(data[:,:,0],cmap='winter')
    plt.colorbar()
    plt.show()
    
    # use to show differences
    plt.imshow(data[:,:,0],cmap='autumn')
    plt.colorbar()
    plt.show()
    
    # a very good sequential plot
    plt.imshow(data[:,:,0],cmap='gist_stern')
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

def simultaneouscontrast():
    plt.figure()
    plt.xlim(0,5)
    plt.ylim(0,5)
    for i in range(0,5):
        gry = i*.2 + .1
        plt.axvspan(i,i+1,facecolor=str(gry),alpha=.5)
        #plt.axhspan(i,i+1,facecolor=str(gry),alpha=.5)
        plt.show()
    plt.scatter(i+.5,2.5,s=300,c='grey')

def handout():
    x = np.linspace(0,4,1000)
    y = np.linspace(0,4,1000)
    X,Y = np.meshgrid(x,y)
    Z = X**2 + Y**2 + np.random.randn(1000,1000)
    plt.pcolormesh(X,Y,Z)
    plt.show()
    plt.pcolormesh(X,Y,Z,cmap='viridis')
    plt.show()

#if __name__ == "__main__":
    #colormaps()
    #colorblind_calf(cvd1)
    #colorblind_calf(cvd2)
    #colorblind_calf(cvd3)
    #colorblind_rainbow(cvd1)
    #colorblind_rainbow(cvd2)
    #colorblind_rainbow(cvd3)
    #simultaneouscontrast()
