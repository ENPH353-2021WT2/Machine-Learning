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


class plateFinder:
	DEBUG = 1
	def __init__(self):
		self.bridge = CvBridge()
		if self.DEBUG:
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
					outimg = cv2.drawContours(img[0], biggestContour, -1, (0,255,0), 3)
					cv2.imwrite(self.outFolder + "/out_" + img[1], outimg)
				except Exception,e:
					print str(e)
					outimg = cv2.drawContours(img[0], contours, -1, (0,255,0), 3)
					cv2.imwrite(self.outFolder + "/out_" + img[1], outimg)
					print("No contours in image")
					continue
		else:
			rospy.init_node('license_plate_analysis')
			self.license_photo_pub = rospy.Publisher('R1/license_photo', Image, queue_size=1)
			time.sleep(1)
			self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback)
			rospy.spin()

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