#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import String
import time

class Robot_Controller:
	# Set class constants here
	COMPETITION_TIME = 5

	def __init__(self):
		rospy.init_node('camera_interpreter')
		self.startup_flag = True
		self.stop_flag = False
		self.bridge = CvBridge()
		self.driving_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
		self.license_pub = rospy.Publisher('/license_plate', String, queue_size=1)
		self.processed_pub = rospy.Publisher('/R1/processed_image', Image, queue_size=1)
		time.sleep(1)
		self.startup_time = time.time()
		self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.linefind)
		rospy.spin()


	#Callback function receives image and proccesses it into a command
	def linefind(self, data):
		cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

		# If on startup, sends start timer command
		if(self.startup_flag):
			self.license_pub.publish(str('TeamRed,multi21,0,XR58'))
			self.startup_flag = False

		# After elapsed time, sends stop command
		if (time.time() > self.startup_time + self.COMPETITION_TIME) and (self.stop_flag == False):
			self.license_pub.publish(str('TeamRed,multi21,-1,XR58'))
			self.stop_flag = True

		#this is where the code from lab 2 begins#
		imgray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
		imgray_cropped = imgray[-100:-1] #y-values first

		ret,thresh = cv2.threshold(imgray_cropped,127,255,0)
		thresh_inverted = cv2.bitwise_not(thresh)

		#having issues doing decision making for when line is not detected
		contours,hierarchy = cv2.findContours(thresh_inverted, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
		try:
			biggestContour=max(contours, key=cv2.contourArea)
		except:
			forward = 0
			turn = 3
			self.sendDriveCommand(forward, turn)
		if cv2.contourArea(biggestContour) < 1000:
			forward = 0
			turn = 3
			self.sendDriveCommand(forward, turn)
			return

		moment_array=cv2.moments(biggestContour)
		try:
			cx = int(moment_array['m10']/moment_array['m00'])
		except ZeroDivisionError:
			#how do i do nothing
			print("divide by zero in moments")
		width = len(imgray[0])
		# rospy.loginfo("Width: " + str(width))

		#perhaps try implementing primitive PID
		if cx <= width / 2:
			forward = 0.5
			turn = 1
			self.sendDriveCommand(forward, turn)
		elif cx > width / 2:
			forward = 0.5
			turn = -1
			self.sendDriveCommand(forward, turn)

		# new stuff
		imHSV = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)
		darkerGray = (0,0,60)
		lighterGray = (0,0,150)

		mask = cv2.inRange(imHSV, darkerGray, lighterGray)

		imFiltered = cv2.bitwise_and(imHSV, imHSV, mask=mask)
		#crop relevant area

		imFiltered_RGB = cv2.cvtColor(imFiltered, cv2.COLOR_HSV2RGB)
		imFiltered_gray = cv2.cvtColor(imFiltered, cv2.COLOR_RGB2GRAY) #y-values first

		# print(imFiltered_gray[0][0].shape)

		threshVal = 0
		maxVal = 255
		ret,thresh = cv2.threshold(imFiltered_gray,threshVal,maxVal,cv2.THRESH_BINARY_INV)
		#having issues doing decision making for when line is not detected
		contours,hierarchy = cv2.findContours(thresh[-400:-1], cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]

		try:
			biggestContour=max(contours, key=cv2.contourArea)
		except:
			forward = 0
			turn = 3
			self.sendDriveCommand(forward, turn)
		if cv2.contourArea(biggestContour) < 1000: #why 1000
			forward = 0
			turn = 3
			self.sendDriveCommand(forward, turn)
			return

		moment_array=cv2.moments(biggestContour)
		try:
			cx = int(moment_array['m10']/moment_array['m00'])
			cy = int(moment_array['m01']/moment_array['m00'])
		except ZeroDivisionError:
			#how do i do nothing
			print("divide by zero in moments")
		width = len(imFiltered_gray[0])
		rospy.loginfo("Width: " + str(width))
		self.publishPhoto(thresh[-400:-1])
		# rospy.loginfo("Forward value: " + str(forward)  + "Turn value: " + str(turn))

	#called from lineDrive, publishes to cmd_vel so the car can drive.
	def sendDriveCommand(self, forwardAmount, turnAmount):
		# pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
		move = Twist()
		move.linear.x = forwardAmount
		move.angular.z = turnAmount
		self.driving_pub.publish(move)

	def scaleXCoord(self, coord, imWidth):
		return coord - imWidth // 3

	def turnRange(self, value, fromMin, fromMax, toMin, toMax):
		fromRange = fromMax - fromMin
		toRange = toMax - toMin

		mappingValue = toRange / fromRange
		output = value * mappingValue - 0.5
		return output 

	#http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
	def publishPhoto(self, processed_image):
		image_message = self.bridge.cv2_to_imgmsg(processed_image, encoding="passthrough")
		self.processed_pub.publish(image_message)


if __name__ == '__main__':
	Robot_Controller()