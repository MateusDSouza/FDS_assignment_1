import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2
from skimage import color



def gauss(sigma):

    _3sigma = 3*sigma
    ceiling = math.floor(_3sigma)
    floor = math.ceil(-_3sigma)
    x = np.arange(floor, ceiling+1, 1)
    #Link below for the simple explanation.
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
    Gx = np.array([((1/(math.sqrt(2*math.pi*sigma)))*math.exp(-(i**2)/(2*sigma**2))) for i in x])

    return Gx, x

def gaussianfilter(img, sigma):
    
    # This is the formula for the 2D gaussian 
    # is the for loop correct?
    Gx, x = gauss(sigma)
    y = x
    Gxy =  np.array([((1/((2*math.pi*sigma)))*math.exp(-((i**2)+(j**2))/(2*sigma**2))) for i in x for j in y])
    kernel = np.reshape(Gxy, (len(x),len(x)))
    smooth_img = conv2(img, kernel, mode= "same")

    return smooth_img

def show_image(image, title='Image', cmap_type='gray'):
	#Display an image using matplotlib
	plt.imshow(image, cmap = cmap_type)
	plt.title(title)
	plt.axis('off')
	plt.show()

# that s the image - CHANGE the directory 
image = plt.imread("C:/Users/Guillaume/Desktop/Fundamentals of data science/Assignment1/Filtering/graf.png")
# Turn it gray 
image = color.rgb2gray(image)
show_image(image)

# a Higher sigma will make it more blurry
test = gaussianfilter(image, 5)
show_image(test)
