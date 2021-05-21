import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, ifft
import math
# import moviepy.editor as mpy ## use moviepy package to write a smoother video
from helper_functions import *
import argparse

## global variables
prev_AR_contours = None

K = np.array([[1406.08415449821,2.20679787308599,1014.13643417416],
            [0.0000000000, 1417.99930662800,566.347754321696],
            [0.0000000000, 0.0000000000,1.0000000000]])


###############################################                                   ###############################################
###############################################   Define General Functions     ###############################################
###############################################                                   ###############################################

def ProjectionMatrix(H):
    global K
    h1, h2, h3 = H[:,0], H[:,1], H[:,2]
    K_inv = np.linalg.inv(K) 
    lamda = 2/(np.linalg.norm(K_inv.dot(h1)) + np.linalg.norm(K_inv.dot(h2)) )
    
    B_ = lamda*K_inv.dot(H)

    if np.linalg.det(B_) > 0 :
        B = B_
    else:
        B = - B_

    r1, r2, r3 = B[:,0], B[:,1], np.cross(B[:,0], B[:,1])
    t = B[:,2]

    RTmatrix = np.dstack((r1,r2,r3,t)).squeeze()
    P = K.dot(RTmatrix)
    return P


def getCubeCoordinates(P,cube_size = 128):

    x1,y1,z1 = P.dot([0,0,0,1])
    x2,y2,z2 = P.dot([0,cube_size,0,1])
    x3,y3,z3 = P.dot([cube_size,0,0,1])
    x4,y4,z4 = P.dot([cube_size,cube_size,0,1])

    x5,y5,z5 = P.dot([0,0,-cube_size,1])
    x6,y6,z6 = P.dot([0,cube_size,-cube_size,1])
    x7,y7,z7 = P.dot([cube_size,0,-cube_size,1])
    x8,y8,z8 = P.dot([cube_size,cube_size,-cube_size,1])

    X = [x1/z1 ,x2/z2 ,x3/z3 ,x4/z4 ,x5/z5 ,x6/z6 ,x7/z7 ,x8/z8] 
    Y = [y1/z1 ,y2/z2 ,y3/z3 ,y4/z4 ,y5/z5 ,y6/z6 ,y7/z7 ,y8/z8] 
    XY = np.dstack((X,Y)).squeeze().astype(np.int32)
    
    return XY

        
def drawCube(im_org, XY):
    im_print = im_org.copy()
    for xy_pts in XY:
        x,y = xy_pts
        cv2.circle(im_print,(x,y), 3, (0,0,255), -1)

    im_print = cv2.line(im_print,tuple(XY[0]),tuple(XY[1]), (0,255,255), 2)
    im_print = cv2.line(im_print,tuple(XY[0]),tuple(XY[2]), (0,255,255), 2)
    im_print = cv2.line(im_print,tuple(XY[0]),tuple(XY[4]), (0,255,255), 2)
    im_print = cv2.line(im_print,tuple(XY[1]),tuple(XY[3]), (0,225,255), 2)
    im_print = cv2.line(im_print,tuple(XY[1]),tuple(XY[5]), (0,225,255), 2)
    im_print = cv2.line(im_print,tuple(XY[2]),tuple(XY[6]), (0,200,255), 2)
    im_print = cv2.line(im_print,tuple(XY[2]),tuple(XY[3]), (0,200,255), 2)
    im_print = cv2.line(im_print,tuple(XY[3]),tuple(XY[7]), (0,175,255), 2)
    im_print = cv2.line(im_print,tuple(XY[4]),tuple(XY[5]), (0,150,255), 2)
    im_print = cv2.line(im_print,tuple(XY[4]),tuple(XY[6]), (0,150,255), 2)
    im_print = cv2.line(im_print,tuple(XY[5]),tuple(XY[7]), (0,125,255), 2)
    im_print = cv2.line(im_print,tuple(XY[6]),tuple(XY[7]), (0,100,255), 2)

    return im_print

# def getCorners(c):
    
#     extLeft = tuple(c[c[:, :, 0].argmin()][0])
#     extRight = tuple(c[c[:, :, 0].argmax()][0])
#     extTop = tuple(c[c[:, :, 1].argmin()][0])
#     extBot = tuple(c[c[:, :, 1].argmax()][0])
#     return np.array([[np.array(extLeft), np.array(extRight) , np.array(extTop) , np.array(extBot)]]).reshape(-1,1,2)


###############################################                                   ###############################################
###############################################   Defining Pipeline Function      ###############################################
###############################################                                   ###############################################

def processAR(im_org,size = 128):
    global prev_AR_contours
    imOut = im_org.copy()
    
    _, binary = cv2.threshold(cv2.cvtColor(im_org, cv2.COLOR_BGR2GRAY), 240, 255, cv2.THRESH_BINARY) 
    kernel = np.ones((3,3),np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    AR_contours = []
    parentFlag = -1
    childFlag = 0
    hierarchy = hierarchy[0]

    ## Find the AR contours
    for i,h in enumerate(hierarchy):
        nxt,prev,child, parent =  h    

        if (parent == -1) and (child > -1) :  # find a parent
            parentFlag = 1
            continue

        if (parent > -1) and (child > -1) and (parentFlag == 1) : # find the first child - has a parent(whiteRegion) ,a child(inner artifacts) 
            parentFlag = 0 # after finding the first child, we set this to ignore the inner children
            AR_contours.append(contours[i])

    # if outliers are present/ nothing is found, use previous set of contours 
    if (len(AR_contours)<0) or (len(AR_contours)>4) :
        AR_contours = prev_AR_contours
    
    #for every contour
    if (len(AR_contours)>0) and (len(AR_contours)<4) :

        prev_AR_contours = AR_contours
        # get corners for every contour        
        for contour in AR_contours:
            

            AR_corners = cv2.approxPolyDP(contour,0.11*cv2.arcLength(contour,True),True) 
            if (len(AR_corners) == 4) :
                
                reference_corners = np.array([[0,0],[0,size],[size,size],[size,0]]).reshape(-1,1,2)
                H = Homography(np.float32(AR_corners), np.float32(reference_corners))   # custom HomoGraphy
                
                # find the projection Matrix
                P = ProjectionMatrix(np.linalg.inv(H))
                # find cube coordinates
                XY = getCubeCoordinates(P,cube_size = size)
                # draw the cube
                imOut = drawCube(imOut, XY)
            
    return imOut


def ProcessVideo(Videopath = '../Data/Tag0.mp4'):
    imgs = []
    cap = cv2.VideoCapture(Videopath)
    outputs = []
    startFlag = True
    counter = 0
    while(True):
        ret, frame = cap.read()
        if ret:
            im_org  = cv2.resize(frame, (800,640))
            imOut = processAR(im_org,size = 128)
            outputs.append(imOut)
            
            
            counter +=1
            if counter%100 == 0:
                print(" Running Frame:", counter)                
        else:
            print("Done Processing.....")
            break        
    cap.release()
                
    return outputs

###############################################                                   ###############################################
############################################### Calling Pipeline in main function ###############################################
###############################################                                   ###############################################

def main():
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--SaveName', default='Cube0', help='Video input name , Default: Tag0')
    Parser.add_argument('--VideoPath', default='../Data/Tag0.mp4', help='Video path , Default:../Data/Tag0.mp4')
    Parser.add_argument('--SavePath', default='../Outputs/Cube/', help='Save Path , Default: ../Outputs/Cube/')
    
    Args = Parser.parse_args()
    SaveName = Args.SaveName
    VideoPath = Args.VideoPath
    SavePath = Args.SavePath
    if(not (os.path.isdir(SavePath))):
        os.makedirs(SavePath)
    
    outputs= ProcessVideo(VideoPath)
    
#     outVideo = mpy.ImageSequenceClip(outputs, fps=25)
#     outVideo.write_videofile(SavePath+str(SaveName)+'_output.mp4')

    out = cv2.VideoWriter(SavePath+str(SaveName)+'_output.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, (800,640))
    for i in range(len(outputs)):
        out.write(outputs[i])
    out.release()
                    
if __name__ == '__main__':
    main()


# references:
# For drawing the cube:
# https://github.com/nalindas9/enpm673/blob/master/AR-tag-detection-and-tracking/Code/cube.py

