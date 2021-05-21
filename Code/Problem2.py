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

###############################################                                   ###############################################
###############################################   Define General Functions        ###############################################
###############################################                                   ###############################################

def RunoneFrame(im_org, testudoBlock):
    global prev_AR_contours
    """
    run the first frame to, find the orientation of the AR, align the testudo in the correct orientation.
    decode the AR and return the decodedvalue
    """

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


    testudoBlocks,decodedValues,rotated_AR_blocks,warpedTags = [],[],[],[]
    
    multiTagFlag = False
    if (len(AR_contours)>0) and (len(AR_contours)<4) :
        prev_AR_contours  = AR_contours
        if len(AR_contours) == 3:
            multiTagFlag = True
            print("---------------------------------------------------------------------")
            print("Multiple Tags found in contour processing: even cv2.warpPerspective function cannot decode all tags accurately")            
            print("Wrong decoding will be resulted in most frames, Decoding AR correctly in Last Frame....")
            print("Orientation of the Testudo Template might not be accurate.....")
            print("---------------------------------------------------------------------")
            
        for contour in AR_contours: # get corners for every contour
            AR_corners = cv2.approxPolyDP(contour,0.11*cv2.arcLength(contour,True),True) 
            if (len(AR_corners) == 4) :                
                      ### predefined functions
#                     reference_corners = np.array([[0,0],[0,64],[64,64],[64,0]]).reshape(-1,1,2)
#                     H = cv2.getPerspectiveTransform(np.float32(AR_corners), np.float32(reference_corners))
#                     warped_AR_tag = cv2.warpPerspective(im_org, H, dsize =(64,64))

                ### custom functions:
                size = 22            
                reference_corners = np.array([[0,0],[0,size],[size,size],[size,0]]).reshape(-1,1,2)
                H = Homography(np.float32(AR_corners), np.float32(reference_corners))
                imOut = np.zeros((size,size,3),dtype = np.uint8)
                warped_AR_tag = Warp(im_org, H, imOut)
                warped_AR_tag  = cv2.resize(warped_AR_tag, (64,64)) # linear interpolation                     
                
                orientation, decodedValue,_ = getOrientation_2(warped_AR_tag,margin=10, decode = True)
                testudoBlock = RotatebyOrientation(testudoBlock, orientation)
                print("Decoded Result :", decodedValue)
                                
    return multiTagFlag, testudoBlock, decodedValue


def RunLastFrame(im_org, testudoBlock):
     
    """
    run the Last frame of MultiTags to, find the orientation of the AR, align the testudo in the correct orientation.
    decode the AR and return the decodedvalue    
    """
    
    print("Decoding Last Frame.... ")
    

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

    ## to print the contours
    im_contours = im_org.copy()
    im_contours = cv2.drawContours(im_contours, AR_contours, -1, (205, 0, 250), 4)


    testudoBlock = cv2.flip(testudoBlock, -1)
    if (len(AR_contours)>0) and (len(AR_contours)<4) :
        # get corners for every contour
        for contour in AR_contours:

            AR_corners = cv2.approxPolyDP(contour,0.11*cv2.arcLength(contour,True),True) 
            if (len(AR_corners) == 4) :
                size = testudoBlock.shape[0]
                testudoBlock,decodedValue = TagRotateDecode(im_org,AR_corners,testudoBlock,Tagsize = 22,out_size = size)
                print("Decoded Value: ", decodedValue)
                
                ## warp testudo to the AR location in video
                reference_corners = np.array([[0,0],[0,size],[size,size],[size,0]]).reshape(-1,1,2)
                H = Homography(np.float32(AR_corners), np.float32(reference_corners))   # custom HomoGraphy
                warpedTestudo = Warp(testudoBlock,np.linalg.inv(H), np.zeros_like(im_org))      # custom Warp
                
                ## paste Testudo in location
                _, warpedTestudo_mask = cv2.threshold(cv2.cvtColor(warpedTestudo, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY_INV)
                warpedTestudo_mask = np.dstack((warpedTestudo_mask,warpedTestudo_mask,warpedTestudo_mask))
                imOut = cv2.bitwise_and(imOut, warpedTestudo_mask)
                imOut = cv2.addWeighted(imOut, 1.0, warpedTestudo, 1.0, 0)
                text_location = tuple(AR_corners.squeeze()[0] + 10)
                text = "Tag:"+str(decodedValue)
                imOut = cv2.putText(imOut, text, text_location, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0) ,3, cv2.LINE_AA, False) 

            else:
                print("error could not decode Last image...")
    imOut = cv2.cvtColor(imOut, cv2.COLOR_BGR2RGB)
    return imOut

def TagRotateDecode(im_org,AR_corners,testudoBlock,Tagsize = 22,out_size = 128 ):

    reference_corners = np.array([[0,0],[0,Tagsize],[Tagsize,Tagsize],[Tagsize,0]]).reshape(-1,1,2)
    H = Homography(np.float32(AR_corners), np.float32(reference_corners))
    ### decode the AR and find the orientation to flip testudo
    imOut_AR = np.zeros((Tagsize,Tagsize,3),dtype = np.uint8)
    warped_AR_tag = Warp(im_org, H, imOut_AR)
    warped_AR_tag  = cv2.resize(warped_AR_tag, (128,128)) # linear interpolation
    orientation, decodedValue,_ = getOrientation_2(warped_AR_tag,margin=10, decode = True)
    testudoBlock = RotatebyOrientation(testudoBlock, orientation)

    return testudoBlock, decodedValue

###############################################                                   ###############################################
###############################################   Defining Pipeline Function      ###############################################
###############################################                                   ###############################################

def processAR(im_org, testudoBlock, decodedValue,MultiTag):
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


    if (len(AR_contours)<0) or (len(AR_contours)>4) :
        AR_contours = prev_AR_contours
        
    if (len(AR_contours)>0) and (len(AR_contours)<4) :
        prev_AR_contours = AR_contours
        
        # get corners for every contour
        for contour in AR_contours:
            ## find the corners using Doglas Peucker Algorithm
            AR_corners = cv2.approxPolyDP(contour,0.11*cv2.arcLength(contour,True),True) 
            if (len(AR_corners) == 4) :
                size = testudoBlock.shape[0]
                
                if MultiTag ==False:
                    testudoBlock,_ = TagRotateDecode(im_org,AR_corners,testudoBlock, Tagsize = 20, out_size = size)
                
                ## warp testudo to the AR location in video
                reference_corners = np.array([[0,0],[0,size],[size,size],[size,0]]).reshape(-1,1,2)
#                  H = cv2.getPerspectiveTransform(np.float32(AR_corners), np.float32(reference_corners)) # cv2 Homography
                H = Homography(np.float32(AR_corners), np.float32(reference_corners))   # custom HomoGraphy
#                  warpedTestudo = cv2.warpPerspective(testudoBlock, np.linalg.inv(H), dsize =(im_org.shape[1],im_org.shape[0]))
                warpedTestudo = Warp(testudoBlock,np.linalg.inv(H), np.zeros_like(im_org))      # custom Warp   
                
                ## get the mask of the warped testudo and paste it in the image
                _, warpedTestudo_mask = cv2.threshold(cv2.cvtColor(warpedTestudo, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY_INV)
                warpedTestudo_mask = np.dstack((warpedTestudo_mask,warpedTestudo_mask,warpedTestudo_mask))
                imOut = cv2.bitwise_and(imOut, warpedTestudo_mask)
                imOut = cv2.addWeighted(imOut, 1.0, warpedTestudo, 1.0, 0)
                text_location = tuple(AR_corners.squeeze()[0] + 10)
                text = str(decodedValue)
                imOut = cv2.putText(imOut, text, text_location, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0) ,3, cv2.LINE_AA, False)
        
    return imOut

def LoadTemplate(testudoPath, size=128):
    testudoBlock = cv2.imread(testudoPath)
    testudoBlock  = cv2.resize(testudoBlock, (size,size))
    testudoBlock = cv2.cvtColor(testudoBlock, cv2.COLOR_BGR2RGB)
    # flip the template here since orientation function for AR is returned in opposite direction
#     testudoBlock= cv2.transpose(testudoBlock)
    return testudoBlock

def ProcessVideo(Videopath = '../Data/Tag0.mp4', testudoPath = "../Data/testudo.png" ):

    cap = cv2.VideoCapture(Videopath)
    outputs = []
    startFlag = True
    counter = 0
    while(True):
        ret, frame = cap.read()
        if ret:
            im_org  = cv2.resize(frame, (800,640))
            counter +=1
            if counter%100 == 0:
                print(" Running Frame:", counter)
            
            if startFlag ==True:                
                testudoBlock = LoadTemplate(testudoPath,size = 128)
                ## run the first frame to decode the value and align testudo initially
                MultiTagFlag , testudoBlock, decodedValue  = RunoneFrame(im_org, testudoBlock)
                if MultiTagFlag:
                    decodedValue = ' '
                startFlag = False
                print("Begin Processing.....")

            imOut = processAR(im_org, testudoBlock, decodedValue, MultiTagFlag)
            imOut = cv2.cvtColor(imOut, cv2.COLOR_BGR2RGB)  ## uncomment when using cv2 videowriterr
            outputs.append(imOut)
        else:
            print("Done Processing.....")
            break
            
    cap.release()
                
    if MultiTagFlag: # if multitag is true, decode in last frame
        imOut  = RunLastFrame(im_org, testudoBlock)
        return outputs, imOut
    else:
        return outputs, None

###############################################                                   ###############################################
############################################### Calling Pipeline in main function ###############################################
###############################################                                   ###############################################
def main():
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--SaveName', default='Tag0', help='Video input name , Default:Tag0')
    Parser.add_argument('--VideoPath', default='../Data/Tag0.mp4', help='Video path , Default:../Data/Tag0.mp4')
    Parser.add_argument('--TemplatePath', default='../Data/testudo.png', help='Template Path , Default: ../Data/testudo.png')
    Parser.add_argument('--SavePath', default='../Outputs/Problem2a/', help='Template Path , Default: .../Outputs/Problem2a/')
    
    Args = Parser.parse_args()
    SaveName = Args.SaveName
    VideoPath = Args.VideoPath
    TemplatePath = Args.TemplatePath
    SavePath = Args.SavePath
    if(not (os.path.isdir(SavePath))):
        os.makedirs(SavePath)
    
    outputs, lastImage = ProcessVideo(VideoPath,TemplatePath)
    
    out = cv2.VideoWriter(SavePath+str(SaveName)+'_output.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, (800,640))
    for i in range(len(outputs)):
        out.write(outputs[i])
    out.release()
    
#     outVideo = mpy.ImageSequenceClip(outputs, fps=25)  ## use moviepy package to write a smoother video
#     outVideo.write_videofile(SavePath+str(SaveName)+'_output.mp4')
    
    if lastImage is not None:
        cv2.imwrite(SavePath+SaveName+'_result.png',lastImage)
        
        
if __name__ == '__main__':
    main()