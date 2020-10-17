import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2
from skimage import color



def gauss(sigma):

	# probably better code than this for the output of the x vector of integers
    if sigma < 1:
    	_3sigma = 3*sigma
    	ceiling = math.ceil(_3sigma)
    	floor = math.floor(-(_3sigma)+1)
    	x = np.array([x for x in range(floor, ceiling)])
    if sigma >= 1:
    	x = np.array([x for x in range(-3*sigma, 3*sigma+1)])

    # Applying the function given to get a 1D gaussian. 

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

    smooth_img = conv2(kernel, img, mode= "same")

    return smooth_img


# that s the image
image = plt.imread("C:/Users/Guillaume/Desktop/Fundamentals of data science/Assignment1/Filtering/graf.png")
# Turn it gray 
image = color.rgb2gray(image)

test = gaussianfilter(image, 0.5)

