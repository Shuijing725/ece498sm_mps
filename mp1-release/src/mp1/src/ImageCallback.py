# import numpy as np
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
#from ipywidgets import interact, interactive, fixed
#import ipywidgets as widgets
#from moviepy.editor import VideoFileClip
#from IPython.display import HTML


window_width = 30
window_height = 100 # Break image into 9 vertical layers since image height is 720
margin = 80 # How much to slide left and right for searching
window = np.ones(window_width) # Create our window template that we will use for convolutions
convThres = 100 # filtered out back ground noise

#Parameters for finding the start points
verticle_ratio = 1.0/2
horizontal_ratio = 1.0/4

def warpImage(img):
#(1) use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
#(2) use cv2.warpPerspective to generate warped image in bird view,
    ## TODO

    ####
    return warpedImg, M, Minv

def laplacian(img, thresh_min=0, thresh_max=255): #after you find the correct values, use them as default values
    #1. convert the image to gray scale
    #2. Gaussian blur the image
    #3. implement laplacian edge detetion using openCV
    #4. using numpy to scale the result to uint8(0-255)
    #5. using the threshold to create a binary image and return it

    ## TODO

    ####

def sobel(img, thresh_min=40, thresh_max=200):
    #1. convert the image to gray scale
    #2. Gaussian blur the image
    #3. Use Sobel to find derievatives for both X and Y Axis
    #4. Use addweight to combine the results
    #5. Convert each pixel to unint8, then apply threshold to get binary image

    ## TODO

    ####



def colorThres(img, thresh_min1=0,thresh_max1=0,thresh_min2=0,thresh_max2=0,thresh_min3=0,thresh_max3=0):
    #when tuning, uncomment the min and max inputs
    #1. Gaussian blur the image (cv2.GaussianBlur())
    #2. Choose the color space that you like, convert image to it (cv2.cvtColor())
    #3. Apply threshold on the image to create binary image output
    #4. Combine results from different color spaces

    ##TODO

    ####
    return binary_output

def combinedEdgeDetection(img):

    ##TODO
    # choose whatever you want, you just need to uncommet each line that you need.
#     LapOutput = laplacian(img, 22, 131)
#    SobelOutput = sobel(img, 27, 90)
#    ColorOutput = colorThres(img)
    #####

    binaryImage = np.zeros_like(ColorOutput)
    #here you can use as many methods as you want.
    binaryImage[(ColorOutput==1)|(SobelOutput==1)] = 1
    return binaryImage



def findConvCenter(image):

    ## TODO


    ####

    #return two lists of tuples
    return leftCentroids, rightCentroids

def window_mask(width, height, img_ref, centerX, centerY):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-centerY-height/2):int(img_ref.shape[0]-centerY+height/2),max(0,int(centerX-width/2)):min(int(centerX+width/2),img_ref.shape[1])] = 1
    return output

def plotWindows(warped):
    leftCentroids, rightCentroids= findConvCenter(warped)

    # Points used to draw all the left and right windows
    r_points = np.zeros_like(warped)
    l_points = np.zeros_like(warped)

    if len(leftCentroids) > 0 and len(rightCentroids) > 0:
        if len(leftCentroids) > 0:
            # Go through each level and draw the windows
            for level in range(0,len(leftCentroids)):
                # Window_mask is a function to draw window areas
                l_mask = window_mask(window_width,window_height,warped,leftCentroids[level][0],leftCentroids[level][1])
                # Add graphic points from window mask here to total pixels found
                l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255

        if len(rightCentroids) > 0:
            # Go through each level and draw the windows
            for level in range(0,len(rightCentroids)):
                # Window_mask is a function to draw window areas
                r_mask = window_mask(window_width,window_height,warped,rightCentroids[level][0],rightCentroids[level][1])
                # Add graphic points from window mask here to total pixels found
                r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage= np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
    #If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)
    return output

def drawPoly(originImg, warpImg, leftCentroids, rightCentroids, Minv):

    ##TODO
    #Generate polynomial functions

    ####
    
    #return the new image and the coefficients of two polynomial equations
    return newImg, leftPlotX, rightPlotX

def Image_Callback(img):
    #img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #undistortedImg = cv2.undistort(img, mtx, dist, None, mtx)
    warpedImg, M, Minv = warpImage(img)
    edgeDetectedImg = combinedEdgeDetection(warpedImg)
    #plotedLinesImg = plotWindows(edgeDetectedImg)
    leftCentroids, rightCentroids = findConvCenter(edgeDetectedImg)
    plotedLinesImg, leftX, rightX = drawPoly(img, warpedImg, leftCentroids, rightCentroids, Minv)

    return plotedLinesImg
