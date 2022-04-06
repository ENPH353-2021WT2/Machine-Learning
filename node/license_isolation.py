#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import String
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# %tensorflow_version 1.14.0
# from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

sess1 = tf.Session()
graph1 = tf.get_default_graph()
set_session(sess1)

class plateFinder:
    """
    Class used to process image feed for license plates and analyze plate numbers.
    ...

    Attributes
    ----------
    DEBUG : Boolean
        Changes input from live driving feed to file directory of images
    counter : int
        For debugging, allows us to save photos with unique filenames
    timer : float
        To time how long license plate data has not been detected by robot 
    errorFolder : string
        Rejected license plates are saved to this folder
    bridge : CvBridge
        Used to convert cv2 images to/from imgmsg for pub/sub
    license_photo_pub: Publisher
        diagnostic feed for analyzed plates
    image_sub : Subscriber
        raw driving feed from robot in Gazebo
    images : list
        Debugging purposes, used to store photos from file directory
    folder : String
        Debugging, input folder for photos
    outFolder : String
        Debugging output folder for analysis
    currPlate : Image
        Fully cropped license plate taken from current frame of video feed.
    currThresh : Image
        Black-and-white version of currPlate with letters thresholded
    cntSort : List
        List of contours corresp. to currThresh sorted by area
    """
    DEBUG = False
    def __init__(self):
        """Sets up all instance variables, mainly pub/sub and timing.

        There is a time.sleep(1) to prevent any messages from being 
        published before being registered with the master node.

        If DEBUG, images come from a folder instead of ROS.
        """
        self.counter = 0
        self.last_license_time = time.time()
        self.published_plate = False
        self.plate_num = 1
        self.bridge = CvBridge()
        self.pub_str = ''
        self.conv_model = models.load_model('my_model1')
        print(self.conv_model.summary())
        self.errorFolder = "/home/fizzer/ros_ws/src/Machine-Learning/output_images/err"
        if not self.DEBUG: #typical analysis with photos from Gazebo
            rospy.init_node('license_plate_analysis')
            self.license_photo_pub = rospy.Publisher('R1/license_photo', Image, queue_size=1)
            self.license_pub = rospy.Publisher('/license_plate', String, queue_size=1)
            time.sleep(1)
            self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback)
            rospy.spin()
        
        elif self.DEBUG: #reads images from a folder
            self.images = []
            self.folder = "/home/fizzer/ros_ws/src/Machine-Learning/training_images"
            self.outFolder = "/home/fizzer/ros_ws/src/Machine-Learning/output_images"
            #https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
            for filename in os.listdir(self.folder):
                img = cv2.imread(os.path.join(self.folder,filename))
                if img is not None:
                    self.images.append([img, filename])

            for img in self.images:
                currPhoto = self.callback(img[0])
                    
                # cv2.imwrite(self.outFolder + "/out_" + img[1], currPhoto)
                # except UnboundLocalError:
                # print("No license plate found in " + img[1])

    def callback(self, data):
        """Callback function that analyzes each frame of video feed for processing.

        Parameters
        ----------
        data : imgmsg
            the photo (as imgmsg) passed by image_sub
        """
        #Convert from imgmsg
        if not self.DEBUG:
            raw_img = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            isolatedImg = self.plateIsolation(data)
        elif self.DEBUG:
            #data is already in CvImage format
            raw_img = data
            isolatedImg = self.plateIsolation(data)

        #Finding the largest continuous shape in image
        contours,hierarchy = cv2.findContours(isolatedImg, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        biggestContour=max(contours, key=cv2.contourArea)

        #Approximating contour into a shape
        approx = cv2.approxPolyDP(biggestContour, 0.009 * cv2.arcLength(biggestContour, True), True)

        #diagnostic drawing around largest shape
        color = (0,255,0)
        thickness = 1
        contourIndex = -1

        #Checking if approx is a quadrilateral
        if len(approx) == 4:
            xi = approx[0][0][0]
            xf = approx[2][0][0]
            
            yi = approx[0][0][1]
            yf = approx[2][0][1]
            outimg = self.shiftPerspective(approx, raw_img)

        #check for existence
        if 'outimg' not in locals():
            return

        # outimg = cv2.cvtColor(outimg, cv2.COLOR_HSV2RGB)

        if self.checkLicensePlate(outimg):
            letters = self.analyzePlate()
            if not self.DEBUG:
                # self.publishPlatePhoto(outimg)
                return
            elif self.DEBUG:
                return outimg

        # Checks if current plate was published already and if it is last image
        if self.published_plate == False and time.time() >= self.last_license_time + 1:
            self.license_pub.publish(self.pub_str)
            self.plate_num += 1
            self.published_plate = True

    def plateIsolation(self, imgmsg):
        """Processes driving feed to show just the big rectangles from the license rears of cars.

        Parameters
        ----------
        imgmsg : imgmsg
            a photo of the robot's driving feed

        Returns
        ----------
        Image of input feed thresholded to just show rears
        """
        #Convert from imgmsg
        if self.DEBUG:
            cv_image = imgmsg
        else:
            cv_image = self.bridge.imgmsg_to_cv2(imgmsg, desired_encoding='passthrough')
        width = len(cv_image[0])

        #convert photo to HSV and isolate for road with mask
        imHSV = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)
        darkestPlate = (0,0,95)
        lightestPlate = (0,0,210)
        mask = cv2.inRange(imHSV, darkestPlate, lightestPlate)
        imFiltered = cv2.bitwise_and(imHSV, imHSV, mask=mask)

        #convert to grayscale in 2 steps
        imFiltered_RGB = cv2.cvtColor(imFiltered, cv2.COLOR_HSV2RGB)
        imFiltered_gray = cv2.cvtColor(imFiltered_RGB, cv2.COLOR_RGB2GRAY)

        #change to pure black/white photo for maximum contrast
        threshVal = 0
        maxVal = 255
        ret,thresh = cv2.threshold(imFiltered_gray,threshVal,maxVal,cv2.THRESH_BINARY)

        outputImage = thresh
        return outputImage

    def shiftPerspective(self, approx, img):
        """Function used to 'unwarp' images for a straight perspective

        Parameters
        ----------
        approx : List
            List of four tuples corresponding to the four corners of
            the back of a car
        img : Image
        """
        #Computing perspectiveshift and warpperspective
        #https://theailearner.com/tag/cv2-warpperspective/
        #(x,y) pairs
        TL = approx[0][0]
        BL = approx[1][0]
        TR = approx[3][0]
        BR = approx[2][0]

        top_width = np.sqrt((TL[0] - TR[0])**2 + (TL[1] - TR[1])**2)
        bot_width = np.sqrt((BL[0] - BR[0])**2 + (BL[1] - BR[1])**2)
        max_width = max(int(top_width), int(bot_width))

        left_height = np.sqrt((TL[0] - BL[0])**2 + (TL[1] - BL[1])**2)
        right_height = np.sqrt((TR[0] - BR[0])**2 + (TR[1] - BR[1])**2)
        max_height = max(int(left_height), int(right_height))

        input_pts = np.float32([TL, BL, BR, TR])
        output_pts = np.float32([[0,0], 
                        [0,max_height - 1],
                        [max_width - 1, max_height - 1],
                        [max_width - 1, 0]])

        M = cv2.getPerspectiveTransform(input_pts, output_pts)
        # newImg = self.drawCorners(img, input_pts, (0,255,0))
        # newImg = self.drawCorners(newImg, output_pts, (255,0,0))
        fixedImg = cv2.warpPerspective(img,M,(max_width, max_height+50),flags=cv2.INTER_LINEAR)
        new_height = int(max_height * 1.25)

        #cropping photo to show just the license plate
        plateImg = fixedImg[max_height:new_height,:]
        return plateImg

    def checkLicensePlate(self, img):
        """Analyzes a photo to see if it is suitable to be sent to the NN

        Parameters
        ----------
        img : Image
            Photo that is cropped from our image feed.

        Returns
        ----------
        True if photo fits our definition of a plate
        False if photo does not fit our definition
        """

        #Image shape analysis. Note that 'perfect' plate has ratio of 2
        self.counter += 1
        largestRatio = 5
        lowestRatio = 2
        y_dim = len(img)
        x_dim = len(img[0])
        imgRatio = x_dim / y_dim

        if imgRatio >largestRatio:
            return False
        if imgRatio < lowestRatio:
            return False

        #Checking if 2 adjacent pixels are the same
        if np.all(img[0][0] == img[0][1]):
            return False

        #Checking if there are 4 blobs in the license plate
        if self.DEBUG:
            #cv2.imread() uses BGR
            plateHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif not self.DEBUG:
            #bridge.imgmsg_to_cv2 uses RGB
            plateHSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        highBlue = (182,255,227)
        lowBlue = (115,113,90)
        plateMask = cv2.inRange(plateHSV, lowBlue, highBlue)
        plateFiltered = cv2.bitwise_and(plateHSV, plateHSV, mask=plateMask)

        plate_RGB = cv2.cvtColor(plateFiltered, cv2.COLOR_HSV2RGB)
        plate_gray = cv2.cvtColor(plate_RGB, cv2.COLOR_RGB2GRAY)
        threshVal = 0
        maxVal = 255
        ret,thresh = cv2.threshold(plate_gray,threshVal,maxVal,cv2.THRESH_BINARY)

        contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        # out = cv2.drawContours(img, contours, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        if self.DEBUG:
            cv2.imshow('plate', thresh)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite(self.outFolder + "/diag/" + str(self.counter) + "plate.png", thresh)
        # if not self.DEBUG:
        #     self.publishPlatePhoto(thresh)
        # print("Contour Length: " + str(len(contours)))
        if len(contours) < 4:
            cv2.imwrite(self.errorFolder + "/" + str(self.counter) + "plate.png", img)
            return False

        cntSort = sorted(contours, key=cv2.contourArea, reverse=True)
        sumTopFour = 0
        for cnt in cntSort[:4]:
            sumTopFour += cv2.contourArea(cnt)
        # print(sumTopFour)
        if sumTopFour < 180:
            return False

        #If we're here, then we've passed all the tests and believe this is a valid plate.
        #Save the raw plate, threshold, and contours for analysis in another function.
        self.currPlate = img
        self.currThresh = thresh
        self.currContours = cntSort

        return True

    def analyzePlate(self):
        """Cuts up a photo into 4 letters, sends to NN and receives a response.
        This function analyzes class variables currPlate, currThresh, currContours.
        We do this to save time in recalculating contours and thresholding.

        This function should only be called when a new plate has been isolated.
        
        Returns
        ----------
        String containing the letters of the license plate.
        """
                #Draw a rectangle on the top four contours
        letters = [0,1,2,3]
        for i, cnt in enumerate(self.currContours[:4]):
            x,y,w,h = cv2.boundingRect(cnt)
            letters[i] = [x,y,w,h]

        #sort letters by x-dimension
        #https://stackoverflow.com/questions/3121979/how-to-sort-a-list-tuple-of-lists-tuples-by-the-element-at-a-given-index
        sortedLetters = sorted(letters, key=lambda tup:tup[0])
        # print(sortedLetters)

        #Extending dimensions
        for let in sortedLetters:
            let[0] = max(let[0] - 1, 0)
            let[1] = max(let[1] - 1, 0)
            let[2] = let[2] + 2
            let[3] = let[3] + 2

        #Drawing letters onto a blank canvas
        maxHeight = max(sortedLetters, key=lambda hei:hei[3])[-1]
        maxWidth = max(sortedLetters, key=lambda wid:wid[2])[-1]

        #Hard-coded dimension for resizing
        maxDim = 32
        testCanvas = np.ones((maxDim * 5, maxDim), np.uint8)
        testCanvas *= 255
        currHeight = 0
        letterImages = [0,1,2,3]

        for i, let in enumerate(sortedLetters):
            x = let[0]
            y = let[1]
            w = let[2]
            h = let[3]

            letterImages[i] = cv2.resize(self.currThresh[y:y+h,x:x+w], (maxDim, maxDim))
            # b, letterImages[i] = cv2.threshold(letterImages[i], 100, 255, cv2.THRESH_BINARY)
            testCanvas[currHeight:currHeight+maxDim,0:maxDim] = letterImages[i]
            currHeight += maxDim + 1

            
        if not self.DEBUG:
            self.publishPlatePhoto(testCanvas)

        #letterImages can now be sent to NN for analysis
        X_dataset_orig = np.array(letterImages)

        # Normalize dataset
        X_dataset_norm = X_dataset_orig/255.0
        X_dataset = np.expand_dims(X_dataset_norm, axis=-1)

        global sess1
        global graph1
        with graph1.as_default():
            set_session(sess1)
            pred = self.conv_model.predict(X_dataset)

            pred_num1 = np.argmax(pred[0])
            pred_num2 = np.argmax(pred[1])
            pred_num3 = np.argmax(pred[2])
            pred_num4 = np.argmax(pred[3])
            pred_char1 = self.num_to_char(pred_num1)
            pred_char2 = self.num_to_char(pred_num2)
            pred_char3 = self.num_to_char(pred_num3)
            pred_char4 = self.num_to_char(pred_num4)
            print(pred_char1)
            print(pred_char2)
            print(pred_char3)
            print(pred_char4)
            print(" ")

        cv2.imshow("X_dataset", X_dataset[0])
        cv2.waitKey(3)

        # String message to publish
        self.pub_str = 'TeamRed,multi21,' + str(self.plate_num) + ',' + str(pred_char1) + str(pred_char2) + str(pred_char3) + str(pred_char4)

        # Update last time a license plate was detected
        self.last_license_time = time.time()

        # Unables publishing again
        self.published_plate = False

        
    def num_to_char(self, num):
        if num <= 25:
            return chr(ord('A')+num)
        else:
            return chr(ord('0')+num-26)

    def publishPlatePhoto(self, isolatedPlates):
        """Publishes a diagnostic photo of the road with just license plate to
        /R1/road_image

        Parameters
        ----------
        isolatedRoad : Image
            Driving feed processed to just a road in black-and-white
        """
        image_message = self.bridge.cv2_to_imgmsg(isolatedPlates, encoding="passthrough")
        self.license_photo_pub.publish(image_message)

if __name__ == '__main__':
    plateFinder()