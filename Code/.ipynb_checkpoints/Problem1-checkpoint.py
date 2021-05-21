import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, ifft
import math
import moviepy.editor as mpy
from helper_functions import *
import argparse

def imRead(path = './Data/Tag0.mp4'):
    startFlag = True
    counter = 1
    cap = cv2.VideoCapture(path)    
    while(True):
        if startFlag==True:
            ret, frame = cap.read()
            if ret:
                frame  = cv2.resize(frame, (512,512))
                startFlag = False
                break
        else:
            counter+=1
            if counter ==10:
                break
    cap.release()
    
    return frame

def drawGrids(block, step = 8):
    """
    ref: http://study.marearts.com/2018/11/python-opencv-draw-grid-example-source.html
    """
    
    block  = cv2.resize(block, (512,512))
    h,w = block.shape[:2]
    
    x = np.linspace(0, w, step).astype(np.int32)
    y = np.linspace(0, h, step).astype(np.int32)

    v_lines = []
    h_lines = []
    for i in range(step):
        v_lines.append( [x[i], 0, x[i], w-1] )
        h_lines.append( [0, int(y[i]), h-1, int(y[i])] )


    for i in range(step):
        [vx1, vy1, vx2, vy2] = v_lines[i]
        [hx1, hy1, hx2, hy2] = h_lines[i]

        block = cv2.line(block, (vx1,vy1), (vx2, vy2), (0,255,255),1 )
        block = cv2.line(block, (hx1,hy1), (hx2, hy2), (0,255,255),1 )
        
    return block


def Tag_using_FFT(im_org):
    """
    Find a tag from a imageframe using FFT pattern noise removal methods.
    
    """
    # convert to grayscale and blur
    im = cv2.cvtColor(im_org, cv2.COLOR_BGR2GRAY)
    im = cv2.GaussianBlur(im,(3,3),0.2)
    
    # find FFT of the image and shift it.
    dft = cv2.dft(np.float32(im),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # find the magnitude spectrum to visualise it.
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    
    # get a circular mask to remove the central spot 
    mask = getCircularMask(dft_shift.shape, radius=30)
    # apply the mask on FFT 
    fshift = dft_shift*mask
    
    # find the magnitude spectrum to visualise masked FFT.
    magnitude_spectrum_lp = 20*np.log(cv2.magnitude(fshift[:,:,0],fshift[:,:,1]))
    
    # inverse shift the FFT
    f_ishift = np.fft.ifftshift(fshift)
    # find magnitude of inverse FFT
    im_ = cv2.idft(f_ishift)
    im_ = cv2.magnitude(im_[:,:,0],im_[:,:,1])
    # scale the inverse FFT to 0-255 to reconstruct the filtered output
    im_ = cv2.normalize(im_, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # perform thresholding to obtain binary map of the image
    _, binary = cv2.threshold(im_, 20, 255, cv2.THRESH_BINARY) 
    kernel = np.ones((3,3),np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # find the contour of the largest area in the image and extract the region where White space of Tag is present. 
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    Area_List = []
    for c in contours:
        Area_List.append(cv2.contourArea(c))
    i = np.argmax(np.array(Area_List))
    rect = cv2.boundingRect(contours[i])
    x,y,w,h = rect
    margin = 10
    # get the AR Tag with white space with 10pixel pad margin
    AR_tag = im_org[y-margin:y+h+margin,x-margin:x+w+margin]  
    
    
    # find the contour of the AR block inside the Tag with  white space 
    _, binary = cv2.threshold(cv2.cvtColor(AR_tag, cv2.COLOR_BGR2GRAY), 240, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    AR_contour = contours[1] # get the AR's contour
    # get the corresponding corner.
    AR_corners = cv2.approxPolyDP(AR_contour,0.11*cv2.arcLength(AR_contour,True),True)
    

    size = 20
    
    reference_corners = np.array([[0,0],[size,0],[size,size],[0,size]]).reshape(-1,1,2)
    H = Homography(np.float32(AR_corners), np.float32(reference_corners))
    
    # warp the AR from the ground to an arbitrarily chosen plane of dimensions size x size 
    imOut = np.zeros((size,size,3),dtype = np.uint8)
    warped_AR_tag = Warp(AR_tag, H, imOut)
    
    warped_AR_tag  = cv2.resize(warped_AR_tag, (128,128)) # resizing to larger size to perform interpolation  
    
    warped_AR_tag = warped_AR_tag[margin:-margin,margin:-margin]
    
    return magnitude_spectrum, magnitude_spectrum_lp, im_, AR_tag, warped_AR_tag


def getOrientation_2(AR_block, margin = 10, decode = True):
    """
    LHS : where the key white block is,
    RHS : corresponding angle by which image needs to be super imposed 

    UL = 180
    UR = -90
    LL = 90
    LR = 0

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

        print("Bit Value: ",bit1,bit2,bit3,bit4 )
        decodedValue = bit1*8 + bit2*4 + bit3*2 + bit4
        print("Decoded Result :", decodedValue)
    else:
        decodedValue = None
    return orientation, decodedValue, rotated_AR_block


def main():
    
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Name', default='Tag1', help='Video input name , Default:Tag0')
    Parser.add_argument('--VideoPath', default='../Data/Tag1.mp4', help='Video path , Default:../Data/Tag0.mp4')
    Parser.add_argument('--ReferencePath', default='../Data/ref_marker.png', help='Template Path , Default: ../Data/template.png')
    Parser.add_argument('--SavePath', default='../Outputs/Problem1/', help='Template Path , Default: ../Data/template.png')
    
    Args = Parser.parse_args()
    VideoPath = Args.VideoPath
    ReferencePath = Args.ReferencePath
    SavePath = Args.SavePath
    
    if(not (os.path.isdir(SavePath))):
        os.makedirs(SavePath)
        
    ###############################################################################################
    ######################################### Problem 1 A #########################################
    ###############################################################################################
    
    im_org = imRead(VideoPath)
    magnitude_spectrum, magnitude_spectrum_hp, im_, AR_tag, AR_block = Tag_using_FFT(im_org)
        
    ## Print the images:
    fig,plts = plt.subplots(2,3,figsize = (15,10))
    plts[0][0].imshow(im_org,'gray')
    plts[0][0].axis('off')
    plts[0][0].title.set_text('a) Input image Frame')
    
    plts[0][1].imshow(magnitude_spectrum,'gray')
    plts[0][1].axis('off')
    plts[0][1].title.set_text('b) FFT magnitude of image')

    plts[0][2].imshow(magnitude_spectrum_hp,'gray')
    plts[0][2].axis('off')
    plts[0][2].title.set_text('c) FFT magnitude after blocking low frequencies ')

    plts[1][0].imshow(im_,'gray')
    plts[1][0].axis('off')
    plts[1][0].title.set_text('d) Detected Edges')
    
    plts[1][1].imshow(AR_tag,'gray')
    plts[1][1].axis('off')
    plts[1][1].title.set_text('e) AR Tag in image plane')
    
    plts[1][2].imshow(AR_block)
    plts[1][2].axis('off')
    plts[1][2].title.set_text('f) AR Tag world plane ')
    
    if(not (os.path.isdir(SavePath))):
        os.makedirs(SavePath)
        
    fig.savefig(SavePath+'ARDetectionUsingFFt.png')
    print("Check ", SavePath," for Problem1a Results")
    
    ###############################################################################################
    ######################################### Problem 1 B #########################################
    ###############################################################################################

    ref_AR_block = cv2.imread(ReferencePath)
    ref_AR_block  = cv2.resize(ref_AR_block, (64,64))
    grid_AR_block = drawGrids(ref_AR_block,8)
    orientation,decodedValue,rotated_AR_block = getOrientation_2(ref_AR_block,decode=True)        

    fig,plts = plt.subplots(1,3,figsize = (15,5))
    plts[0].imshow(ref_AR_block)
    plts[0].axis('off')
    plts[0].title.set_text('a) Input Reference AR Tag')
    
    plts[1].imshow(grid_AR_block)
    plts[1].axis('off')
    plts[1].title.set_text('b) AR tag with a grid')
    
    plts[2].imshow(rotated_AR_block, cmap='gray')
    plts[2].axis('off')
    plts[2].text(24,54,"Decoded Value: "+str(decodedValue))
    plts[2].title.set_text('c) Rotated by '+str(orientation)+" degrees")
    
    fig.savefig(SavePath+'AR_Decoded.png')
    print("Check ", SavePath," for Problem1b Results")
if __name__ == '__main__':
    main()
    
    
