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


class plateFinder:
	DEBUG = 1
	def __init__(self):
		self.bridge = CvBridge()
		if self.DEBUG: #reads images from a folder
			self.images = []
			self.folder = "/home/fizzer/ros_ws/src/Machine-Learning/training_images"
			self.outFolder = "/home/fizzer/ros_ws/src/Machine-Learning/output_images"
			#https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
			for filename in os.listdir(self.folder):
				img = cv2.imread(os.path.join(self.folder,filename))
				if img is not None:
					self.images.append([img, filename])

			for img in self.images:
				currPhoto = self.plateIsolation(img[0])
				contours,hierarchy = cv2.findContours(currPhoto, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
				try:
					biggestContour=max(contours, key=cv2.contourArea)
					print(str(cv2.contourArea(biggestContour)) + " " + img[1])
					contourIndex = -1
					color = (0,255,0)
					thickness = 2
					#we need square brackets because drawContours expects a list of ndarrays
					# outimg = cv2.drawContours(img[0], [biggestContour], contourIndex, color,thickness)

					#approximating and detecting quadrilaterals
					approx = cv2.approxPolyDP(biggestContour, 0.009 * cv2.arcLength(biggestContour, True), True)
					outimg = cv2.drawContours(img[0], [approx], contourIndex, color,thickness)
					print(approx)
					print(approx.shape)
					if len(approx) == 4:
						xi = approx[0][0][0]
						xf = approx[2][0][0]
						
						yi = approx[0][0][1]
						yf = approx[2][0][1]
						yf_new = yf + int(0.3 * (yf-yi))
						# outimg = img[0][yi:yf_new, xi:xf]
						outimg = self.shiftPerspective(approx, outimg)

					cv2.imwrite(self.outFolder + "/out_" + img[1], outimg)
				except Exception,e:
					print str(e)
					contourIndex = -1
					color = (0,255,0)
					thickness = 2
					outimg = cv2.drawContours(img[0], contours, contourIndex, color, thickness)
					cv2.imwrite(self.outFolder + "/out_" + img[1], outimg)
					print("No contours in image")
					continue
		else: #typical analysis with photos from Gazebo
			rospy.init_node('license_plate_analysis')
			self.license_photo_pub = rospy.Publisher('R1/license_photo', Image, queue_size=1)
			time.sleep(1)
			self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback)
			rospy.spin()

	def shiftPerspective(self, approx, img):
		#CURRENTLY WIP. DOESN'T SHIFT PROPERLY, AND CUTS OFF PLATE!!
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
						[max_height - 1, max_width - 1],
						[max_width - 1, 0]])

		M = cv2.getPerspectiveTransform(input_pts, output_pts)
		newImg = cv2.warpPerspective(img,M,(max_width, max_height),flags=cv2.INTER_LINEAR)
		return newImg

	def callback(self, data):
		photo = self.plateIsolation(data)
		rospy.loginfo("asdf")
		self.publishPlatePhoto(photo)


	def plateIsolation(self, imgmsg):
		"""Processes driving feed to show just the big rectangles from the license rears of cars.

		Parameters
		----------
		imgmsg : imgmsg
			a photo of the robot's driving feed
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