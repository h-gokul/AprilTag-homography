import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, ifft
import math
# import moviepy.editor as mpy

prev_AR_block = None

def videoRead(path = './Data/Ball_travel_10fps.mp4'):
    imgs = []
    cap = cv2.VideoCapture(path)
    
    while(True):
        ret, frame = cap.read()
        if ret:
            frame  = cv2.resize(frame, (800,640))
            imgs.append(frame)
        else:
            break
    cap.release()
    
    return imgs

def Warp(src, H , dst):
    """ To perform warping,
    Apply H on every pixel location from src (a,b ,c=1) to find destination location x,y,z
    since z doesnt exist, do x = x/z, y = y/z , so z = 1
    
    if x and y locations are within the boundaries of dst image shape,
    paste the value from src [a,b] at dst [x,y]
    """
    im = cv2.transpose(src)
    height,width = im.shape[:2]
    h_limit, w_limit = dst.shape[:2]
    for a in range(width):
        for b in range(height):
            ab1 = np.array([a ,b, 1])
            x,y,z = H.dot(ab1)
            x,y = int(x/z), int(y/z)
            if (x >= 0 and x < w_limit) and (y >= 0 and y < h_limit) :
                    dst[int(y),int(x)] = im[a ,b]
    
    return dst

def Homography(pts1,pts2):
    """ To compute Homography matrix:
    1) Compute the A matrix with 8x9 dimensions. Need to solve for Ax = 0 
    2) perform Singular Value decomposition on A matrix to find the solution x 
    3) reshape x into a 3x3 homography matrix
    """
    pts1,pts2 = pts1.squeeze(),pts2.squeeze()
    
    X,Y = pts1[:,0],pts1[:,1]
    Xp,Yp = pts2[:,0],pts2[:,1]

    ############################ Generate A matrix ############################
    startFlag=1

    for (x,y,xp,yp) in zip(X,Y,Xp,Yp):

        if (startFlag == 1) :
            A = np.array([[-x,-y,-1,0,0,0, x*xp, y*xp,xp], [0,0,0,-x,-y,-1, x*yp, y*yp, yp]])
        else:
            tmp = np.array([[-x,-y,-1,0,0,0, x*xp, y*xp,xp], [0,0,0,-x,-y,-1, x*yp, y*yp, yp]])
            A = np.vstack((A, tmp))

        startFlag+=1    

    U,S,Vt = np.linalg.svd(A.astype(np.float32))

    H_ = Vt[8,:]/Vt[8][8]
    H_ = H_.reshape(3,3)
    
    return H_

def flip(im):
    return cv2.flip(im,-1)

def getCircularMask(shape, radius):
    """
    get a circular black mask at the center of the frame with given radius
    """
    thickness = -1
    color = 0
    
    centre = (shape[0]//2,shape[1]//2)
    mask = np.ones(shape, dtype=np.float32)
    cv2.circle(mask, centre, radius, color, thickness)
    mask = mask/255.0
    
    return mask


def crop_AR(AR_block):

    """
    For a given AR tag, crop the black region
    
    """
    global prev_AR_block
    Xdistribution = np.sum(AR_block,axis=0)
    Ydistribution = np.sum(AR_block,axis=1)
    
    mdpt = len(Xdistribution)//2
    left_Xdistribution = Xdistribution[:mdpt]
    right_Xdistribution = Xdistribution[mdpt:]
    
    leftx,rightx,topx,topy = -1,-1,-1,-1
    for i in range(len(left_Xdistribution)):
        if left_Xdistribution[i] > 2000:
            leftx = i
            break

    for i in range(len(right_Xdistribution)):
        if right_Xdistribution[i] < 2000:
            rightx = i
            rightx+=mdpt
            break
    

    top_Ydistribution = Ydistribution[:mdpt]
    bottom_Ydistribution = Ydistribution[mdpt:]

    for i in range(len(top_Ydistribution)):
        if top_Ydistribution[i] > 2000:
            topy = i
            break

    for i in range(len(bottom_Ydistribution)):
        if bottom_Ydistribution[i] < 2000:
            bottomy = i
            bottomy+=mdpt
            break

    cropped_AR_block = AR_block[topy:bottomy,leftx:rightx]
    
    if (leftx < 0 )or(rightx < 0)or(topy < 0 )or(bottomy < 0):
        cropped_AR_block  = prev_AR_block
        print('bad tag found')
    else:
        prev_AR_block = cropped_AR_block        
        
    return cropped_AR_block

def getOrientation_2(AR_block, margin = 10, decode = True):
    """
    LHS : where the key white block is,
    RHS : corresponding angle by which image needs to be super imposed 

    UL = 180
    UR = -90
    LL = 90
    LR = 0
    -------------------------------------------------------------------------
    To estimate Orientation:
    Crop the black paddings out -> Find the orientation pattern -> Compute the orientation angle
    
    To decode:
    Rotate the tag as per the orientation -> Find the bits in clockwise order -> perform decimal conversion
    
    """
    AR_block = AR_block[margin:-margin,margin:-margin]
    _, AR_block = cv2.threshold(cv2.cvtColor(AR_block, cv2.COLOR_BGR2GRAY), 240, 255, cv2.THRESH_BINARY) # only threshold
    cropped_AR_block = crop_AR(AR_block)
    cropped_AR_block  = cv2.resize(cropped_AR_block, (64,64))

    lowerright = cropped_AR_block[48:64,48:64]
    lowerleft = cropped_AR_block[48:64,0:16]

    upperright = cropped_AR_block[0:16,48:64]
    upperleft = cropped_AR_block[0:16,0:16]

    UL,UR,LL,LR = np.int(np.median(upperleft)), np.int(np.median(upperright)), np.int(np.median(lowerleft)), np.int(np.median(lowerright))

    AR_orientationPattern = [UL,UR,LL,LR]
    orientations = [180,-90,90,0]

    index = np.argmax(AR_orientationPattern)

    orientation = orientations[index]

    if decode ==  True:
        rotated_AR_block = RotatebyOrientation(cropped_AR_block, orientation)
        
        block1 = rotated_AR_block[16:32,16:32]
        block2 = rotated_AR_block[16:32,32:48]
        block3 = rotated_AR_block[32:48,32:48]
        block4 = rotated_AR_block[32:48, 16:32]

        bit1 = np.median(block1)/255
        bit2 = np.median(block2)/255
        bit3 = np.median(block3)/255
        bit4 = np.median(block4)/255

#         print("Bit Value: ",bit1,bit2,bit3,bit4 )
        decodedValue = bit1*1 + bit2*2 + bit3*4 + bit4*8

    else:
        decodedValue = None
    return orientation, decodedValue, rotated_AR_block

def RotatebyOrientation(Block, orientation):
    
    # rotateBlock by orientation degree
    if orientation == 90:
#         print("Rotated anticlckwise 90")
        Block = cv2.rotate(Block, cv2.cv2.ROTATE_90_CLOCKWISE) 

    elif orientation == -90:
#         print("Rotated clckwise 90")
        Block = cv2.rotate(Block, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

    elif orientation == 180:
#         print("Rotated 180")
        Block = cv2.rotate(Block, cv2.cv2.ROTATE_180) 
    return Block


